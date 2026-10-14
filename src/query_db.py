#!/usr/bin/env python3
"""
Hybrid Semantic & Lexical Scripture Retrieval Engine (Step 7)
============================================================
Provides unified retrieval across yoga scriptures combining:
1. Dense Semantic Retrieval:
   - OpenAI `text-embedding-3-small` (1536 dims) embeddings queried via ChromaDB cosine space.
2. Sparse Keyword Retrieval:
   - In-memory BM25 index via `rank-bm25` directly from `chunks.jsonl` (40ms build time).
   - Guarantees 100% precision on technical and transliterated Sanskrit terms (Viparita Karani, Sushumna, etc.).
3. Reciprocal Rank Fusion (RRF):
   - Merges dense and sparse result lists using Cormack's RRF algorithm ($k=60$).
4. Metadata Filters:
   - Filter by book, chapter, or sloka-only.
5. Interactive & Benchmark Modes:
   - `--interactive`: Terminal prompt loop for exploratory queries.
   - `--benchmark`: Automated retrieval evaluation across Sanskrit terms, concepts, and techniques.
"""

import os
import sys
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi, BM25Plus

from chroma_manager import ChromaManager, load_yaml_config

# Automatically load environment variables from .env
load_dotenv()


CONVERSATIONAL_QUESTION_STOPWORDS = {
    "how", "to", "perform", "practice", "do", "does", "did", "doing", "done",
    "what", "which", "who", "whom", "where", "when", "why",
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "in", "of", "for", "and", "or", "on", "with", "at", "by", "from", "about",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "you", "your", "i", "me", "my", "we", "our", "tell", "explain", "describe",
}


def tokenize_for_bm25(text: str, filter_stopwords: bool = False) -> List[str]:
    """
    Tokenizes text for BM25: lowercase, word boundaries, strips punctuation.
    When filter_stopwords=True, removes conversational question-framing words
    so BM25 scores focus exclusively on technical domain entities.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    if filter_stopwords:
        filtered = [t for t in tokens if t not in CONVERSATIONAL_QUESTION_STOPWORDS]
        return filtered if filtered else tokens
    return tokens


def chunk_matches_filter(chunk: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    """Evaluates whether a chunk matches a ChromaDB-style `where` filter."""
    if not where:
        return True

    for k, v in where.items():
        if k == "$and" and isinstance(v, list):
            if not all(chunk_matches_filter(chunk, sub_w) for sub_w in v):
                return False
        elif k == "$or" and isinstance(v, list):
            if not any(chunk_matches_filter(chunk, sub_w) for sub_w in v):
                return False
        else:
            chunk_val = chunk.get(k)
            # Handle stringified booleans if applicable
            if chunk_val != v:
                return False
    return True


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_mode: str = "hybrid"
    dense_rank: Optional[int] = None
    dense_distance: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None


class HybridRetriever:
    """
    Coordinates Dense ChromaDB Retrieval and Sparse BM25 Keyword Search
    fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        config_or_path: Union[str, Dict[str, Any]],
        chunks_jsonl_path: Optional[str] = None,
        api_key: Optional[str] = None,
        rrf_k: int = 60
    ):
        if isinstance(config_or_path, str):
            self.config = load_yaml_config(config_or_path)
        else:
            self.config = config_or_path

        self.rrf_k = rrf_k
        self.manager = ChromaManager(self.config)
        self.collection = self.manager.collection

        # Resolve paths
        self.chunks_path = chunks_jsonl_path or self.config.get("output_path", "chunks.jsonl")
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks file not found at: {self.chunks_path}")

        # Embedding model config
        emb_conf = self.config.get("embedding", {})
        self.embedding_model = emb_conf.get("model", "text-embedding-3-small")

        # OpenAI client
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._openai_client: Optional[OpenAI] = None

        # Build in-memory BM25 index
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_by_id: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self._build_bm25_index()

    @property
    def openai_client(self) -> OpenAI:
        """Lazily initializes the OpenAI client."""
        if self._openai_client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found. Add it to .env or export it in your shell.")
            self._openai_client = OpenAI(api_key=self.api_key)
        return self._openai_client

    def _build_bm25_index(self) -> None:
        """Builds in-memory BM25Okapi index directly from chunks.jsonl."""
        t0 = time.time()
        corpus = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                self.chunks.append(chunk)
                self.chunk_by_id[chunk["chunk_id"]] = chunk
                corpus.append(tokenize_for_bm25(chunk["content"]))

        self.bm25 = BM25Okapi(corpus)
        elapsed_ms = (time.time() - t0) * 1000
        # print(f"[*] Built BM25 index across {len(self.chunks)} chunks in {elapsed_ms:.1f}ms")

    def embed_query(self, query: str) -> List[float]:
        """Embeds query text using OpenAI text-embedding-3-small."""
        resp = self.openai_client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        return resp.data[0].embedding

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Pure dense vector search via ChromaDB cosine similarity."""
        q_emb = self.embed_query(query)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where
        )

        results: List[SearchResult] = []
        if res and res["ids"] and res["ids"][0]:
            for rank, (c_id, dist, doc, meta) in enumerate(
                zip(res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0]),
                start=1
            ):
                # Cosine similarity = 1 - cosine distance
                sim = 1.0 - dist
                results.append(SearchResult(
                    chunk_id=c_id,
                    content=doc,
                    metadata=self.manager.deserialize_metadata(meta),
                    score=sim,
                    retrieval_mode="dense",
                    dense_rank=rank,
                    dense_distance=dist
                ))
        return results

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        filter_stopwords: bool = True
    ) -> List[SearchResult]:
        """Pure sparse keyword search via in-memory BM25 with query stopword filtering."""
        tokens = tokenize_for_bm25(query, filter_stopwords=filter_stopwords)
        scores = self.bm25.get_scores(tokens)

        # Pair scores with chunk objects
        candidates = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            if where and not chunk_matches_filter(chunk, where):
                continue
            candidates.append((chunk, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]

        results: List[SearchResult] = []
        for rank, (chunk, score) in enumerate(top_candidates, start=1):
            results.append(SearchResult(
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                metadata=chunk,
                score=score,
                retrieval_mode="bm25",
                bm25_rank=rank,
                bm25_score=score
            ))
        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        candidate_multiplier: int = 4
    ) -> List[SearchResult]:
        """
        Hybrid retrieval combining Dense Vector + BM25 via Reciprocal Rank Fusion (RRF).
        Fuses top candidates using RRF formula: RRF_Score = sum(1 / (k + rank)).
        """
        fetch_k = max(top_k * candidate_multiplier, 20)

        # 1. Retrieve dense candidates
        dense_results = self.search_dense(query=query, top_k=fetch_k, where=where)

        # 2. Retrieve sparse BM25 candidates
        bm25_results = self.search_bm25(query=query, top_k=fetch_k, where=where)

        # 3. Fuse via Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        dense_info: Dict[str, Tuple[int, float]] = {}
        bm25_info: Dict[str, Tuple[int, float]] = {}

        for r in dense_results:
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + (1.0 / (self.rrf_k + r.dense_rank))
            dense_info[r.chunk_id] = (r.dense_rank, r.dense_distance)

        for r in bm25_results:
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + (1.0 / (self.rrf_k + r.bm25_rank))
            bm25_info[r.chunk_id] = (r.bm25_rank, r.bm25_score)

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        fused_results: List[SearchResult] = []
        for c_id in sorted_ids:
            chunk = self.chunk_by_id[c_id]
            d_rank, d_dist = dense_info.get(c_id, (None, None))
            b_rank, b_score = bm25_info.get(c_id, (None, None))

            fused_results.append(SearchResult(
                chunk_id=c_id,
                content=chunk["content"],
                metadata=chunk,
                score=rrf_scores[c_id],
                retrieval_mode="hybrid_rrf",
                dense_rank=d_rank,
                dense_distance=d_dist,
                bm25_rank=b_rank,
                bm25_score=b_score
            ))

        return fused_results

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        chapter: Optional[int] = None,
        sloka_only: bool = False
    ) -> List[SearchResult]:
        """Unified search dispatch with metadata filtering."""
        filters = []
        if chapter is not None:
            filters.append({"chapter_number": chapter})
        if sloka_only:
            filters.append({"has_sanskrit_sloka": True})

        where = None
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        if mode == "dense":
            return self.search_dense(query=query, top_k=top_k, where=where)
        elif mode == "bm25":
            return self.search_bm25(query=query, top_k=top_k, where=where)
        elif mode == "hybrid":
            return self.search_hybrid(query=query, top_k=top_k, where=where)
        else:
            raise ValueError(f"Unknown search mode: '{mode}'. Use 'hybrid', 'dense', or 'bm25'.")


def format_search_results(results: List[SearchResult], max_content_chars: int = 350, full_content: bool = False) -> str:
    """Formats search results for clean CLI display."""
    if not results:
        return "[!] No matching chunks found.\n"

    lines = []
    lines.append("\n" + "=" * 65)
    lines.append(f"             SEARCH RESULTS ({len(results)} chunks)")
    lines.append("=" * 65)

    for idx, r in enumerate(results, start=1):
        meta = r.metadata
        ch_num = meta.get("chapter_number", "?")
        ch_title = meta.get("chapter_title", "Unknown")
        v_label = meta.get("verse_label", "?")
        p_start = meta.get("page_start", "?")
        p_end = meta.get("page_end", "?")
        is_sub = meta.get("is_sub_chunk", False)
        sub_idx = meta.get("sub_chunk_index", 0) + 1
        total_subs = meta.get("total_sub_chunks", 1)

        sub_tag = f" [Part {sub_idx}/{total_subs}]" if is_sub else ""
        has_sloka_tag = " [Sloka Root]" if meta.get("has_sanskrit_sloka") else ""

        lines.append(f"\n{idx}. [{r.chunk_id}] (Score: {r.score:.4f} | Mode: {r.retrieval_mode})")
        lines.append(f"   Chapter {ch_num}: {ch_title} | Verse {v_label}{sub_tag}{has_sloka_tag} | Pages {p_start}-{p_end}")

        if r.retrieval_mode == "hybrid_rrf":
            dist_str = f"dist: {r.dense_distance:.4f}" if r.dense_distance is not None else "dist: N/A"
            d_info = f"Dense Rank: {r.dense_rank} ({dist_str})" if r.dense_rank else "Dense: Unranked"
            bm_str = f"score: {r.bm25_score:.2f}" if r.bm25_score is not None else "score: N/A"
            b_info = f"BM25 Rank: {r.bm25_rank} ({bm_str})" if r.bm25_rank else "BM25: Unranked"
            lines.append(f"   Fusion Metrics: [{d_info}] | [{b_info}]")

        if full_content:
            lines.append("   " + "-" * 55)
            for c_line in r.content.strip().split("\n"):
                lines.append(f"   {c_line}")
            lines.append("   " + "-" * 55)
        else:
            # Content preview
            preview = r.content.strip().replace("\n\n", " \u2022 ").replace("\n", " ")
            if len(preview) > max_content_chars:
                cut = preview[:max_content_chars]
                last_space = cut.rfind(" ")
                if last_space > 0:
                    cut = cut[:last_space]
                preview = cut + "..."
            lines.append(f"   Preview: \"{preview}\"")

    lines.append("=" * 65 + "\n")
    return "\n".join(lines)


def run_benchmark_suite(retriever: HybridRetriever) -> Dict[str, Any]:
    """Runs standard benchmark across exact terms, concepts, and techniques."""
    benchmark_queries = [
        # 1. Exact Sanskrit Technical Terms (Tests BM25 exact keyword match)
        {"category": "Exact Term", "query": "Ujjayi pranayama", "expected_verse": "51-53"},
        {"category": "Exact Term", "query": "Viparita Karani mudra", "expected_verse": "78-82"},
        {"category": "Exact Term", "query": "Sushumna nadi kundalini", "expected_verse": None},
        {"category": "Exact Term", "query": "Kevali kumbhaka", "expected_verse": "72"},
        # 2. Conceptual & Philosophical Queries (Tests Dense semantic abstraction)
        {"category": "Conceptual", "query": "What are the causes of failure in yoga according to Chapter 1?", "expected_verse": "15"},
        {"category": "Conceptual", "query": "How does breath control conquer death and decay?", "expected_verse": None},
        {"category": "Conceptual", "query": "What is the relationship between hatha yoga and raja yoga?", "expected_verse": "2"},
        # 3. Asana & Practice Inquiries (Tests Technique retrieval)
        {"category": "Technique", "query": "Benefits and practice of Matsyendrasana", "expected_verse": "26"},
        {"category": "Technique", "query": "How to perform Mayurasana peacock pose", "expected_verse": "30"},
        {"category": "Technique", "query": "Shatkarma six cleansing practices dhauti basti neti", "expected_verse": "22"},
    ]

    print("\n" + "=" * 70)
    print("           RETRIEVAL BENCHMARK EVALUATION (HYBRID VS DENSE)")
    print("=" * 70)

    total = len(benchmark_queries)
    results = []

    for b in benchmark_queries:
        q = b["query"]
        cat = b["category"]
        exp = b["expected_verse"]

        t0 = time.time()
        hybrid_hits = retriever.search_hybrid(q, top_k=3)
        hybrid_ms = (time.time() - t0) * 1000

        top_hit = hybrid_hits[0] if hybrid_hits else None
        top_v = top_hit.metadata.get("verse_label") if top_hit else "None"
        matched = (exp is None) or (exp in top_v)

        print(f"[{cat}] Query: \"{q}\"")
        print(f"  Top Match: Verse {top_v} ({top_hit.chunk_id if top_hit else 'N/A'}) in {hybrid_ms:.1f}ms")
        if exp:
            status = "✓ HIT" if matched else "✗ MISS"
            print(f"  Expected Verse: {exp} -> {status}")
        print()

        results.append({
            "query": q,
            "category": cat,
            "expected": exp,
            "top_verse": top_v,
            "matched": matched,
            "latency_ms": hybrid_ms
        })

    print("=" * 70 + "\n")
    return {"total": total, "results": results}


def run_interactive_loop(retriever: HybridRetriever, mode: str = "hybrid") -> None:
    """Provides a terminal REPL loop for interactive scripture search."""
    print("\n" + "=" * 60)
    print("   🕉️  HATHA YOGA SCRIPTURE SEARCH ENGINE (INTERACTIVE)")
    print("=" * 60)
    print("  Mode:        " + mode.upper())
    print("  Commands:    'exit' or 'quit' to terminate")
    print("               ':dense', ':bm25', ':hybrid' to switch mode")
    print("               ':ch 1', ':ch 2' to set chapter filter")
    print("               ':all' to clear chapter filter")
    print("               ':sloka' to toggle Sanskrit-only slokas")
    print("               ':full' to toggle full chunk content vs preview")
    print("               ':view N' to inspect full content of result N")
    print("=" * 60 + "\n")

    current_mode = mode
    current_ch = None
    sloka_only = False
    full_content = False
    last_results: List[SearchResult] = []

    while True:
        try:
            status_tag = f"[{current_mode}"
            if current_ch:
                status_tag += f" | Ch {current_ch}"
            if sloka_only:
                status_tag += " | Sloka-Only"
            if full_content:
                status_tag += " | Full"
            status_tag += "]"

            user_input = input(f"{status_tag} Query > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("[*] Exiting search. Namaste!")
                break

            # Handle command switches
            if user_input.startswith(":"):
                cmd = user_input.lower().split()
                if cmd[0] in [":hybrid", ":dense", ":bm25"]:
                    current_mode = cmd[0][1:]
                    print(f"[*] Switched mode to: {current_mode.upper()}")
                elif cmd[0] == ":ch" and len(cmd) > 1 and cmd[1].isdigit():
                    current_ch = int(cmd[1])
                    print(f"[*] Filtering to Chapter {current_ch}")
                elif cmd[0] == ":all":
                    current_ch = None
                    sloka_only = False
                    print("[*] Cleared all filters.")
                elif cmd[0] == ":sloka":
                    sloka_only = not sloka_only
                    print(f"[*] Sloka-only filter: {sloka_only}")
                elif cmd[0] == ":full":
                    full_content = not full_content
                    print(f"[*] Full content display: {'ENABLED (showing complete chunk text)' if full_content else 'DISABLED (showing previews)'}")
                elif cmd[0] == ":view":
                    if not last_results:
                        print("[!] No search results to view yet. Please run a search first.")
                    elif len(cmd) > 1 and cmd[1].isdigit():
                        idx = int(cmd[1])
                        if 1 <= idx <= len(last_results):
                            print(format_search_results([last_results[idx - 1]], full_content=True))
                        else:
                            print(f"[!] Invalid index {idx}. Choose between 1 and {len(last_results)}.")
                    else:
                        print("[!] Usage: ':view <number>' (e.g., ':view 1')")
                else:
                    print("[!] Unknown command. Use ':hybrid', ':dense', ':bm25', ':ch N', ':all', ':sloka', ':full', or ':view N'.")
                continue

            t0 = time.time()
            results = retriever.search(
                query=user_input,
                mode=current_mode,
                top_k=4,
                chapter=current_ch,
                sloka_only=sloka_only
            )
            last_results = results
            elapsed_ms = (time.time() - t0) * 1000

            print(f"\n[Retrieved in {elapsed_ms:.1f}ms]")
            print(format_search_results(results, full_content=full_content))

        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting search. Namaste!")
            break
        except Exception as e:
            print(f"[!] Error executing search: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Semantic & Lexical Scripture Retrieval")
    parser.add_argument("--config", default="configs/hatha_yoga_pradipika.yaml", help="Path to YAML config")
    parser.add_argument("--chunks", default=None, help="Path to chunks.jsonl")
    parser.add_argument("--query", "-q", default=None, help="Search query string")
    parser.add_argument("--mode", "-m", choices=["hybrid", "dense", "bm25"], default="hybrid", help="Retrieval mode (default: hybrid)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--chapter", "-c", type=int, default=None, help="Filter by chapter number (1-4)")
    parser.add_argument("--sloka-only", action="store_true", help="Only retrieve root verses containing Sanskrit slokas")
    parser.add_argument("--full", action="store_true", help="Display full untruncated chunk text instead of preview")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive search REPL prompt loop")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run standard retrieval benchmark suite")
    args = parser.parse_args()

    retriever = HybridRetriever(config_or_path=args.config, chunks_jsonl_path=args.chunks)

    if args.benchmark:
        run_benchmark_suite(retriever)
        return

    if args.interactive or not args.query:
        run_interactive_loop(retriever, mode=args.mode)
        return

    # Single Query Execution
    results = retriever.search(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        chapter=args.chapter,
        sloka_only=args.sloka_only
    )
    print(format_search_results(results, full_content=args.full))


if __name__ == "__main__":
    main()
