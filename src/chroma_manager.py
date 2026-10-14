#!/usr/bin/env python3
"""
ChromaDB Management & Collection Architecture Module (Step 5)
=============================================================
Provides persistent ChromaDB setup, collection management, metadata
sanitization, and book-scoped purge operations.

Features:
1. Persistent Local Setup:
   - Configures SQLite + HNSW storage via `chromadb.PersistentClient(path=...)`.
   - Sets collection metadata with cosine distance (`{"hnsw:space": "cosine"}`).
2. ChromaDB Metadata Guard:
   - Prevents ChromaDB `ValueError` on empty lists (`[]`) by JSON-stringifying `image_paths`.
   - Prevents ChromaDB `TypeError` on null values by safely filtering out `None` fields.
   - Extracts `content` into the ChromaDB `document` argument rather than metadata.
3. Targeted Book-Scoped Purge:
   - Eliminates stale / ghost sub-chunks when re-indexing via `collection.delete(where={"book_id": book_id})`.
4. Zero-API Dependency for Architecture Setup:
   - Does not require OpenAI API key or internet access to initialize or inspect the database.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional, Union
import yaml
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads declarative YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ChromaManager:
    """Manages persistent ChromaDB client, collections, and metadata contracts."""

    def __init__(self, config_or_path: Union[str, Dict[str, Any], None] = None):
        if isinstance(config_or_path, str):
            self.config = load_yaml_config(config_or_path)
        elif isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            self.config = {}

        v_conf = self.config.get("vector_db", {})
        self.persist_dir = v_conf.get("persist_directory", "chroma_db")
        self.collection_name = v_conf.get("collection_name", "yoga_scriptures")
        self.distance_metric = v_conf.get("distance_metric", "cosine")
        self.mode = v_conf.get("mode", "embedded")

        self._client: Optional[ClientAPI] = None
        self._collection: Optional[Collection] = None

    @property
    def client(self) -> ClientAPI:
        """Lazily initializes and returns the ChromaDB client."""
        if self._client is None:
            if self.mode == "embedded":
                os.makedirs(self.persist_dir, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            elif self.mode == "server":
                host = self.config.get("vector_db", {}).get("host", "localhost")
                port = int(self.config.get("vector_db", {}).get("port", 8000))
                self._client = chromadb.HttpClient(host=host, port=port)
            else:
                raise ValueError(f"Unsupported ChromaDB mode: '{self.mode}'. Use 'embedded' or 'server'.")
        return self._client

    def get_or_create_collection(self, name: Optional[str] = None) -> Collection:
        """
        Retrieves or creates the collection with cosine distance metadata.
        """
        col_name = name or self.collection_name
        self._collection = self.client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        return self._collection

    @property
    def collection(self) -> Collection:
        """Returns the active collection instance."""
        if self._collection is None:
            self.get_or_create_collection()
        return self._collection

    @staticmethod
    def sanitize_metadata(chunk_data: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Prepares a chunk dictionary for ChromaDB metadata storage:
        1. Excludes 'content' (passed as ChromaDB 'document').
        2. Serializes lists (like 'image_paths') to JSON strings ('[]') to satisfy ChromaDB constraints.
        3. Excludes None values to prevent ChromaDB TypeError.
        4. Enforces supported scalar types (str, int, float, bool).
        """
        metadata: Dict[str, Union[str, int, float, bool]] = {}

        for k, v in chunk_data.items():
            if k == "content":
                continue  # Content belongs in document, not metadata
            if v is None:
                continue  # ChromaDB rejects null/None metadata values

            if isinstance(v, list):
                # ChromaDB requires non-empty lists; empty lists raise ValueError.
                # Serializing to JSON string safely preserves empty or populated lists.
                metadata[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, (str, int, float, bool)):
                metadata[k] = v
            else:
                metadata[k] = str(v)

        return metadata

    @staticmethod
    def deserialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restores sanitized metadata fields (e.g. parsing JSON string lists back to Python lists).
        """
        deserialized = dict(metadata)
        if "image_paths" in deserialized and isinstance(deserialized["image_paths"], str):
            try:
                deserialized["image_paths"] = json.loads(deserialized["image_paths"])
            except json.JSONDecodeError:
                deserialized["image_paths"] = []
        return deserialized

    def purge_book(self, book_id: str) -> None:
        """
        Executes a targeted scoped purge for a given book_id to eliminate stale ghost vectors.
        Safe for multi-book collections; only deletes vectors matching `book_id`.
        """
        col = self.collection
        col.delete(where={"book_id": book_id})

    def get_collection_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics for the configured collection."""
        col = self.collection
        count = col.count()

        return {
            "collection_name": col.name,
            "total_items": count,
            "persist_directory": os.path.abspath(self.persist_dir) if self.mode == "embedded" else f"{self.mode}",
            "distance_metric": self.distance_metric,
            "metadata_config": col.metadata
        }

    def validate_chunks_file(self, jsonl_path: str) -> Dict[str, Any]:
        """
        Validates all chunks in a JSONL file against ChromaDB metadata rules
        without performing actual embedding or insertion.
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"File not found: {jsonl_path}")

        total_chunks = 0
        sanitization_errors = 0
        missing_content_errors = 0
        book_ids = set()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    total_chunks += 1
                    if "content" not in chunk or not chunk["content"]:
                        missing_content_errors += 1
                    if "book_id" in chunk:
                        book_ids.add(chunk["book_id"])

                    # Test metadata sanitization
                    sanitized = self.sanitize_metadata(chunk)

                    # Ensure all values are allowed primitive types
                    for k, val in sanitized.items():
                        if not isinstance(val, (str, int, float, bool)):
                            sanitization_errors += 1

                except Exception as e:
                    sanitization_errors += 1

        return {
            "total_chunks_validated": total_chunks,
            "valid": (sanitization_errors == 0 and missing_content_errors == 0),
            "sanitization_errors": sanitization_errors,
            "missing_content_errors": missing_content_errors,
            "book_ids": sorted(list(book_ids))
        }


def main():
    parser = argparse.ArgumentParser(description="ChromaDB Collection Architecture & Setup Manager")
    parser.add_argument("--config", default="configs/hatha_yoga_pradipika.yaml", help="Path to YAML config")
    parser.add_argument("--init", action="store_true", help="Initialize persistent database and collection")
    parser.add_argument("--status", action="store_true", help="Show current collection statistics")
    parser.add_argument("--purge", type=str, help="Execute targeted purge for a specific book_id")
    parser.add_argument("--validate-chunks", type=str, help="Validate a chunks.jsonl file against ChromaDB metadata rules")
    args = parser.parse_args()

    manager = ChromaManager(args.config)

    if args.init:
        col = manager.get_or_create_collection()
        print(f"[✓] Persistent ChromaDB initialized at: {os.path.abspath(manager.persist_dir)}")
        print(f"[✓] Collection '{col.name}' ready with distance metric: {manager.distance_metric}")

    if args.purge:
        print(f"[*] Executing targeted purge for book '{args.purge}'...")
        manager.purge_book(args.purge)
        print(f"[✓] Purge complete for '{args.purge}'.")

    if args.validate_chunks:
        print(f"[*] Validating metadata contracts for: {args.validate_chunks}...")
        report = manager.validate_chunks_file(args.validate_chunks)
        print("\n" + "=" * 50)
        print("          METADATA VALIDATION REPORT")
        print("=" * 50)
        print(f"  Total Chunks Tested:      {report['total_chunks_validated']}")
        print(f"  Books Detected:           {', '.join(report['book_ids'])}")
        print(f"  Sanitization Errors:      {report['sanitization_errors']}")
        print(f"  Missing Content Errors:   {report['missing_content_errors']}")
        print(f"  ChromaDB Schema Status:   {'PASSED (100% Compliant)' if report['valid'] else 'FAILED'}")
        print("=" * 50 + "\n")

    if args.status or (not args.init and not args.purge and not args.validate_chunks):
        stats = manager.get_collection_stats()
        print("\n" + "=" * 50)
        print("         CHROMADB COLLECTION STATUS")
        print("=" * 50)
        print(f"  Collection Name:    {stats['collection_name']}")
        print(f"  Total Indexed Items:{stats['total_items']}")
        print(f"  Distance Metric:    {stats['distance_metric']}")
        print(f"  Storage Location:   {stats['persist_directory']}")
        print(f"  Collection Metadata:{stats['metadata_config']}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
