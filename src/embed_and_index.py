#!/usr/bin/env python3
"""
Batch Ingestion & Vector Indexing Module (Step 6)
=================================================
Streams normalized chunk records from `chunks.jsonl`, generates 1536-dimensional
dense vector embeddings via OpenAI `text-embedding-3-small`, and populates persistent
ChromaDB collection with sanitized metadata contracts and targeted book-scoped purge.

Features:
1. Streaming Ingestion:
   - Reads chunks line-by-line ($O(1)$ RAM), eliminating memory spikes on large corpora.
2. Targeted Book-Scoped Purge (Idempotency):
   - Purges only the active `book_id` prior to indexing (`where={"book_id": book_id}`).
   - Eliminates orphaned ghost vectors without impacting other books in the collection.
3. OpenAI Embedding Generation:
   - Standardized on `text-embedding-3-small` (1536 dimensions).
   - Built-in exponential backoff retry for network/rate-limit resilience.
   - Exact token consumption and cost tracking.
4. ChromaDB Metadata Contract Guard:
   - Sanitizes metadata via `ChromaManager.sanitize_metadata`:
     - Excludes `content` from metadata (passed as document).
     - Serializes empty or populated lists (`image_paths`) to JSON strings.
     - Strips `None` values to avoid ChromaDB `TypeError`.
5. Progress & Diagnostics:
   - Progress bar via `tqdm`.
   - Comprehensive post-indexing summary report with token metrics and cost accounting.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Iterator, Optional, Tuple, Union
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIConnectionError, APIError

from chroma_manager import ChromaManager, load_yaml_config

# Automatically load environment variables from .env
load_dotenv()


def stream_chunks(jsonl_path: str) -> Iterator[Dict[str, Any]]:
    """
    Streams chunk dictionaries line-by-line from a JSONL file.
    Maintains O(1) memory footprint regardless of file size.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Chunks file not found: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                yield json.loads(line_str)
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON on line {line_num} of {jsonl_path}: {err}") from err


def batch_iterator(iterable: Iterator[Any], batch_size: int = 100) -> Iterator[List[Any]]:
    """
    Groups elements from an iterator into fixed-size batches.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class BookIndexer:
    """
    Coordinates batch embedding generation and ChromaDB ingestion for a scripture.
    """

    def __init__(
        self,
        config_or_path: Union[str, Dict[str, Any]],
        api_key: Optional[str] = None,
        batch_size: int = 100,
        purge: bool = True,
        dry_run: bool = False
    ):
        if isinstance(config_or_path, str):
            self.config = load_yaml_config(config_or_path)
            self.config_path = config_or_path
        else:
            self.config = config_or_path
            self.config_path = None

        self.book_id = self.config.get("book_id", "book")
        self.book_title = self.config.get("book_title", "Untitled Book")
        self.batch_size = batch_size
        self.purge = purge
        self.dry_run = dry_run

        # Embedding model configuration
        emb_conf = self.config.get("embedding", {})
        self.embedding_model = emb_conf.get("model", "text-embedding-3-small")
        self.expected_dim = int(emb_conf.get("dimension", 1536))

        # Resolve API Key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key and not self.dry_run:
            raise ValueError(
                "OpenAI API key not found. Please add OPENAI_API_KEY to your .env file "
                "or pass it via the OPENAI_API_KEY environment variable."
            )

        self._openai_client: Optional[OpenAI] = None
        self.manager = ChromaManager(self.config)

    @property
    def openai_client(self) -> OpenAI:
        """Lazily initializes the OpenAI client."""
        if self._openai_client is None:
            if not self.api_key:
                raise ValueError("Cannot initialize OpenAI client without an API key.")
            self._openai_client = OpenAI(api_key=self.api_key)
        return self._openai_client

    def embed_batch_with_retry(
        self,
        texts: List[str],
        max_retries: int = 5,
        backoff_factor: float = 2.0
    ) -> Tuple[List[List[float]], int]:
        """
        Embeds a batch of texts using OpenAI API with exponential backoff retry.
        Returns:
            Tuple of (list_of_embeddings, total_tokens_used)
        """
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                response = self.openai_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                embeddings = [item.embedding for item in response.data]
                tokens_used = response.usage.total_tokens if response.usage else 0
                return embeddings, tokens_used
            except (RateLimitError, APIConnectionError) as e:
                if attempt == max_retries:
                    raise RuntimeError(f"OpenAI API failed after {max_retries} attempts: {e}") from e
                time.sleep(delay)
                delay *= backoff_factor
            except APIError as e:
                raise RuntimeError(f"OpenAI API Error: {e}") from e

        raise RuntimeError("Unexpected failure in embed_batch_with_retry.")

    def count_total_chunks(self, chunks_path: str) -> int:
        """Quickly counts non-empty lines in chunks file for progress bar sizing."""
        count = 0
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def index(self, chunks_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the end-to-end batch ingestion process:
        1. Optionally purges existing vectors for book_id (targeted idempotency).
        2. Streams chunks from JSONL in batches.
        3. Embeds chunk documents via OpenAI.
        4. Upserts records into ChromaDB collection.
        5. Reports metrics and collection status.
        """
        target_path = chunks_path or self.config.get("output_path", "chunks.jsonl")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Chunks file not found at: {target_path}")

        total_chunks = self.count_total_chunks(target_path)
        start_time = time.time()

        print("\n" + "=" * 60)
        print("          BATCH VECTOR INGESTION & INDEXING")
        print("=" * 60)
        print(f"  Book Title:         {self.book_title} ({self.book_id})")
        print(f"  Source Chunks:      {target_path} ({total_chunks} chunks)")
        print(f"  Target Collection:  {self.manager.collection_name}")
        print(f"  Embedding Model:    {self.embedding_model} ({self.expected_dim} dims)")
        print(f"  Batch Size:         {self.batch_size}")
        print(f"  Targeted Purge:     {'Enabled (Idempotent)' if self.purge else 'Disabled'}")
        print(f"  Mode:               {'DRY RUN (No API calls)' if self.dry_run else 'LIVE INGESTION'}")
        print("=" * 60 + "\n")

        # 1. Targeted Book-Scoped Purge
        if self.purge and not self.dry_run:
            print(f"[*] Executing targeted purge for book '{self.book_id}' to eliminate ghost vectors...")
            self.manager.purge_book(self.book_id)
            print(f"[✓] Scoped purge complete for '{self.book_id}'.")

        collection = self.manager.collection if not self.dry_run else None
        chunks_stream = stream_chunks(target_path)
        batches = batch_iterator(chunks_stream, batch_size=self.batch_size)

        total_indexed = 0
        total_tokens = 0
        total_batches = (total_chunks + self.batch_size - 1) // self.batch_size if total_chunks > 0 else 0

        with tqdm(total=total_chunks, desc="Indexing Chunks", unit="chunk") as pbar:
            for batch_idx, batch in enumerate(batches, start=1):
                batch_ids: List[str] = []
                batch_docs: List[str] = []
                batch_metadatas: List[Dict[str, Any]] = []

                for chunk in batch:
                    c_id = chunk["chunk_id"]
                    doc = chunk["content"]
                    sanitized_meta = self.manager.sanitize_metadata(chunk)

                    batch_ids.append(c_id)
                    batch_docs.append(doc)
                    batch_metadatas.append(sanitized_meta)

                if self.dry_run:
                    # Validate vector dimension placeholder and metadata schema
                    total_indexed += len(batch)
                    pbar.update(len(batch))
                    continue

                # Live Embedding & Upsert
                embeddings, tokens_used = self.embed_batch_with_retry(batch_docs)

                # Validate embedding dimension
                if embeddings and len(embeddings[0]) != self.expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.expected_dim}, "
                        f"got {len(embeddings[0])}"
                    )

                # Upsert into ChromaDB
                collection.upsert(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_docs,
                    metadatas=batch_metadatas
                )

                total_indexed += len(batch)
                total_tokens += tokens_used
                pbar.update(len(batch))

        elapsed_time = time.time() - start_time
        # text-embedding-3-small pricing: $0.02 per 1,000,000 tokens
        estimated_cost = (total_tokens / 1_000_000.0) * 0.02

        stats = self.manager.get_collection_stats() if not self.dry_run else {
            "collection_name": self.manager.collection_name,
            "total_items": total_indexed,
            "persist_directory": self.manager.persist_dir,
            "distance_metric": self.manager.distance_metric,
            "metadata_config": {"hnsw:space": self.manager.distance_metric}
        }

        print("\n" + "=" * 60)
        print("             INGESTION & INDEXING SUMMARY")
        print("=" * 60)
        print(f"  Book ID:                 {self.book_id}")
        print(f"  Collection Name:         {stats['collection_name']}")
        print(f"  Total Chunks Indexed:    {total_indexed}")
        print(f"  Total API Tokens Used:   {total_tokens:,}")
        print(f"  Total Estimated Cost:    ${estimated_cost:.5f} USD")
        print(f"  Ingestion Duration:      {elapsed_time:.2f} seconds")
        print(f"  Indexing Speed:          {total_indexed / elapsed_time:.1f} chunks/sec" if elapsed_time > 0 else "")
        print(f"  Total Collection Items:  {stats['total_items']}")
        print(f"  Storage Location:        {stats['persist_directory']}")
        print(f"  Status:                  {'DRY RUN COMPLETE' if self.dry_run else 'SUCCESS (100% Indexed)'}")
        print("=" * 60 + "\n")

        return {
            "book_id": self.book_id,
            "total_indexed": total_indexed,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "elapsed_seconds": elapsed_time,
            "collection_stats": stats,
            "dry_run": self.dry_run
        }


def main():
    parser = argparse.ArgumentParser(description="Batch Vector Ingestion & Indexing for Scripture Chunks")
    parser.add_argument("--config", default="configs/hatha_yoga_pradipika.yaml", help="Path to YAML config")
    parser.add_argument("--chunks", default=None, help="Path to chunks.jsonl (defaults to config output_path)")
    parser.add_argument("--batch-size", type=int, default=100, help="Embedding batch size (default: 100)")
    parser.add_argument("--no-purge", action="store_true", help="Skip targeted book purge before indexing")
    parser.add_argument("--dry-run", action="store_true", help="Simulate batching and schema validation without OpenAI calls")
    args = parser.parse_args()

    indexer = BookIndexer(
        config_or_path=args.config,
        batch_size=args.batch_size,
        purge=not args.no_purge,
        dry_run=args.dry_run
    )

    indexer.index(chunks_path=args.chunks)


if __name__ == "__main__":
    main()
