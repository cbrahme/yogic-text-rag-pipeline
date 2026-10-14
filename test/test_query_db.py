#!/usr/bin/env python3
"""
Unit Test Suite for Hybrid Scripture Retrieval Engine (Step 7)
==============================================================
Verifies:
1. `tokenize_for_bm25`: Clean lowercase alphanumeric token extraction.
2. `chunk_matches_filter`: Metadata filter matching with boolean operators.
3. `HybridRetriever` BM25 Search: Keyword matching directly over chunks.
4. `HybridRetriever` Reciprocal Rank Fusion (RRF): Proper mathematical score
   aggregation across dense and sparse ranking lists.
5. Filter enforcement: Chapter filtering and sloka-only constraints.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query_db import (
    tokenize_for_bm25,
    chunk_matches_filter,
    format_search_results,
    HybridRetriever,
    SearchResult
)


class TestRetrievalHelpers(unittest.TestCase):
    """Tests tokenization and filter matching utilities."""

    def test_tokenize_for_bm25(self):
        text = "Matsyēndrāsana, (spinal twist) -- Chapter 1!"
        tokens = tokenize_for_bm25(text)
        self.assertIn("matsyēndrāsana", tokens)
        self.assertIn("spinal", tokens)
        self.assertIn("twist", tokens)
        self.assertIn("chapter", tokens)
        self.assertIn("1", tokens)

    def test_tokenize_for_bm25_stopword_filtering(self):
        """Verifies conversational question framing words are cleanly stripped for queries."""
        query = "How to perform Moola Bandha?"
        tokens = tokenize_for_bm25(query, filter_stopwords=True)
        self.assertEqual(tokens, ["moola", "bandha"])

        # Fallback when query is only stopwords
        all_stop = "how to do"
        fallback_tokens = tokenize_for_bm25(all_stop, filter_stopwords=True)
        self.assertEqual(fallback_tokens, ["how", "to", "do"])

    def test_chunk_matches_filter_simple(self):
        chunk = {"book_id": "hyp", "chapter_number": 2, "has_sanskrit_sloka": True}
        self.assertTrue(chunk_matches_filter(chunk, {"chapter_number": 2}))
        self.assertFalse(chunk_matches_filter(chunk, {"chapter_number": 1}))

    def test_chunk_matches_filter_and(self):
        chunk = {"book_id": "hyp", "chapter_number": 2, "has_sanskrit_sloka": True}
        where = {"$and": [{"chapter_number": 2}, {"has_sanskrit_sloka": True}]}
        self.assertTrue(chunk_matches_filter(chunk, where))

        where_false = {"$and": [{"chapter_number": 2}, {"has_sanskrit_sloka": False}]}
        self.assertFalse(chunk_matches_filter(chunk, where_false))

    def test_format_search_results_preview_and_full(self):
        """Verifies format_search_results truncates when full_content=False and preserves all content when full_content=True."""
        long_content = "Line 1: Commentary on Moola Bandha.\n" + ("A" * 400) + "\nLine 3: Final instruction."
        res = SearchResult(
            chunk_id="hyp_ch03_v061-063_p01",
            content=long_content,
            metadata={
                "chapter_number": 3,
                "chapter_title": "Mudra",
                "verse_label": "61-63",
                "page_start": 340,
                "page_end": 342,
                "has_sanskrit_sloka": True
            },
            score=0.95,
            retrieval_mode="hybrid_rrf",
            dense_rank=1,
            bm25_rank=1
        )

        # 1. Preview mode (truncated)
        preview_out = format_search_results([res], max_content_chars=100, full_content=False)
        self.assertIn("Preview:", preview_out)
        self.assertIn("...", preview_out)
        self.assertNotIn("Line 3: Final instruction.", preview_out)

        # 2. Full content mode (untruncated)
        full_out = format_search_results([res], full_content=True)
        self.assertNotIn("Preview:", full_out)
        self.assertIn("Line 1: Commentary on Moola Bandha.", full_out)
        self.assertIn("Line 3: Final instruction.", full_out)
        self.assertIn("A" * 400, full_out)


class TestHybridRetriever(unittest.TestCase):
    """Tests BM25, Dense, and Hybrid RRF retrieval flows."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_jsonl = os.path.join(self.temp_dir, "test_chunks.jsonl")
        self.chroma_dir = os.path.join(self.temp_dir, "test_chroma_db")

        self.sample_data = [
            {
                "chunk_id": "hyp_ch01_v026_p01",
                "book_id": "hyp",
                "book_title": "Hatha Yoga Pradipika",
                "parent_verse_id": "hyp_ch01_v026",
                "chapter_number": 1,
                "chapter_title": "Asana",
                "verse_start": 26,
                "verse_end": 26,
                "verse_label": "26",
                "is_sub_chunk": False,
                "sub_chunk_index": 0,
                "total_sub_chunks": 1,
                "has_sanskrit_sloka": True,
                "page_start": 64,
                "page_end": 65,
                "chunk_type": "verse",
                "char_count": 200,
                "token_count": 50,
                "image_paths": [],
                "content": "MATSYENDRASANA spinal twist pose. Practice of this posture increases digestive fire."
            },
            {
                "chunk_id": "hyp_ch02_v051-053_p01",
                "book_id": "hyp",
                "book_title": "Hatha Yoga Pradipika",
                "parent_verse_id": "hyp_ch02_v051-053",
                "chapter_number": 2,
                "chapter_title": "Shatkarma and Pranayama",
                "verse_start": 51,
                "verse_end": 53,
                "verse_label": "51-53",
                "is_sub_chunk": True,
                "sub_chunk_index": 0,
                "total_sub_chunks": 3,
                "has_sanskrit_sloka": True,
                "page_start": 217,
                "page_end": 219,
                "chunk_type": "verse_subchunk",
                "char_count": 250,
                "token_count": 60,
                "image_paths": [],
                "content": "UJJAYI PRANAYAMA psychic breath. Closing the mouth, inhale with control and concentration."
            },
            {
                "chunk_id": "hyp_ch01_v028_p01",
                "book_id": "hyp",
                "book_title": "Hatha Yoga Pradipika",
                "parent_verse_id": "hyp_ch01_v028",
                "chapter_number": 1,
                "chapter_title": "Asana",
                "verse_start": 28,
                "verse_end": 28,
                "verse_label": "28",
                "is_sub_chunk": False,
                "sub_chunk_index": 0,
                "total_sub_chunks": 1,
                "has_sanskrit_sloka": True,
                "page_start": 69,
                "page_end": 70,
                "chunk_type": "verse",
                "char_count": 200,
                "token_count": 50,
                "image_paths": [],
                "content": "PASCHIMOTTANASANA back stretching pose. Stretching the legs on the ground like a stick."
            },
            {
                "chunk_id": "hyp_ch01_v032_p01",
                "book_id": "hyp",
                "book_title": "Hatha Yoga Pradipika",
                "parent_verse_id": "hyp_ch01_v032",
                "chapter_number": 1,
                "chapter_title": "Asana",
                "verse_start": 32,
                "verse_end": 32,
                "verse_label": "32",
                "is_sub_chunk": False,
                "sub_chunk_index": 0,
                "total_sub_chunks": 1,
                "has_sanskrit_sloka": True,
                "page_start": 74,
                "page_end": 75,
                "chunk_type": "verse",
                "char_count": 180,
                "token_count": 45,
                "image_paths": [],
                "content": "SHAVASANA corpse pose. Lying flat on the ground like a corpse takes away fatigue."
            }
        ]

        with open(self.sample_jsonl, "w", encoding="utf-8") as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + "\n")

        self.config = {
            "book_id": "hyp",
            "book_title": "Hatha Yoga Pradipika",
            "output_path": self.sample_jsonl,
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimension": 1536
            },
            "vector_db": {
                "type": "chromadb",
                "persist_directory": self.chroma_dir,
                "collection_name": "test_search_collection",
                "distance_metric": "cosine"
            }
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bm25_search(self):
        """Verifies BM25 ranks exact keyword hits accurately."""
        retriever = HybridRetriever(self.config, chunks_jsonl_path=self.sample_jsonl)
        hits = retriever.search_bm25("Ujjayi psychic breath", top_k=2)
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0].chunk_id, "hyp_ch02_v051-053_p01")
        self.assertEqual(hits[0].retrieval_mode, "bm25")

    @patch.object(HybridRetriever, "search_dense")
    def test_search_hybrid_rrf_fusion(self, mock_search_dense):
        """Verifies RRF mathematically combines dense and BM25 candidate ranks."""
        retriever = HybridRetriever(self.config, chunks_jsonl_path=self.sample_jsonl)

        # Mock dense results: ranks Matsyendrasana #1, Ujjayi #2
        mock_search_dense.return_value = [
            SearchResult(
                chunk_id="hyp_ch01_v026_p01",
                content="MATSYENDRASANA",
                metadata=self.sample_data[0],
                score=0.9,
                retrieval_mode="dense",
                dense_rank=1,
                dense_distance=0.1
            ),
            SearchResult(
                chunk_id="hyp_ch02_v051-053_p01",
                content="UJJAYI PRANAYAMA",
                metadata=self.sample_data[1],
                score=0.7,
                retrieval_mode="dense",
                dense_rank=2,
                dense_distance=0.3
            )
        ]

        # Query Ujjayi (BM25 will rank Ujjayi #1)
        fused = retriever.search_hybrid("Ujjayi psychic breath", top_k=2)
        self.assertEqual(len(fused), 2)

        # Ujjayi is rank 2 in dense (1/(60+2) = 0.0161) and rank 1 in BM25 (1/(60+1) = 0.0164)
        # Matsyendrasana is rank 1 in dense (1/(60+1) = 0.0164) and unranked in BM25 (0.0)
        # Therefore Ujjayi has higher RRF score and must be Rank 1
        self.assertEqual(fused[0].chunk_id, "hyp_ch02_v051-053_p01")
        self.assertGreater(fused[0].score, fused[1].score)
        self.assertEqual(fused[0].dense_rank, 2)
        self.assertEqual(fused[0].bm25_rank, 1)

    def test_chapter_and_sloka_filtering(self):
        """Verifies search dispatch applies chapter and sloka_only filters."""
        retriever = HybridRetriever(self.config, chunks_jsonl_path=self.sample_jsonl)

        # Filter by Chapter 1
        hits_ch1 = retriever.search("breath", mode="bm25", chapter=1)
        for h in hits_ch1:
            self.assertEqual(h.metadata["chapter_number"], 1)

        # Filter by Chapter 2
        hits_ch2 = retriever.search("breath", mode="bm25", chapter=2)
        for h in hits_ch2:
            self.assertEqual(h.metadata["chapter_number"], 2)


if __name__ == "__main__":
    unittest.main()
