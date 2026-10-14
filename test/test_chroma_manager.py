#!/usr/bin/env python3
"""
Unit Tests for ChromaDB Manager & Collection Architecture (Step 5)
==================================================================
Tests:
1. Collection initialization & cosine distance metric.
2. Metadata sanitization (empty array prevention, None handling, content segregation).
3. Metadata deserialization (JSON string to Python list).
4. Direct insertion and metadata filtering (single & compound filters).
5. Targeted book-scoped purge (isolation between books).
"""

import os
import sys
import json
import unittest
import shutil
import tempfile

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chroma_manager import ChromaManager


class TestChromaManager(unittest.TestCase):

    def setUp(self):
        # Use an isolated temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            "vector_db": {
                "type": "chromadb",
                "persist_directory": self.test_dir,
                "collection_name": "test_collection",
                "distance_metric": "cosine",
                "mode": "embedded"
            }
        }
        self.manager = ChromaManager(self.config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization_and_metadata(self):
        """Test persistent collection creation and cosine space configuration."""
        col = self.manager.get_or_create_collection()
        self.assertEqual(col.name, "test_collection")
        self.assertEqual(col.metadata.get("hnsw:space"), "cosine")
        self.assertEqual(col.count(), 0)

    def test_sanitize_metadata_empty_list_and_none(self):
        """Test that image_paths: [] is serialized to JSON string and None is filtered."""
        raw_chunk = {
            "chunk_id": "hyp_ch01_v001_p01",
            "book_id": "hyp",
            "chapter_number": 1,
            "has_sanskrit_sloka": True,
            "parent_verse_id": None,  # None should be filtered out
            "content": "Salutations to Adinath...",  # Content should be omitted from metadata
            "image_paths": []  # Empty list should be converted to string '[]'
        }

        sanitized = ChromaManager.sanitize_metadata(raw_chunk)

        self.assertNotIn("content", sanitized)
        self.assertNotIn("parent_verse_id", sanitized)
        self.assertEqual(sanitized["image_paths"], "[]")
        self.assertEqual(sanitized["book_id"], "hyp")
        self.assertEqual(sanitized["chapter_number"], 1)
        self.assertTrue(sanitized["has_sanskrit_sloka"])

    def test_deserialize_metadata(self):
        """Test that image_paths JSON string is parsed back to list."""
        metadata = {
            "book_id": "hyp",
            "image_paths": "[\"images/hyp/fig1.png\"]"
        }
        deserialized = ChromaManager.deserialize_metadata(metadata)
        self.assertEqual(deserialized["image_paths"], ["images/hyp/fig1.png"])

        # Test empty string list
        empty_meta = {"image_paths": "[]"}
        deserialized_empty = ChromaManager.deserialize_metadata(empty_meta)
        self.assertEqual(deserialized_empty["image_paths"], [])

    def test_add_and_filter_records(self):
        """Test inserting records with sanitized metadata and filtering with where clauses."""
        col = self.manager.get_or_create_collection()

        chunk_1 = {
            "chunk_id": "hyp_ch01_v001_p01",
            "book_id": "hyp",
            "chapter_number": 1,
            "verse_start": 1,
            "has_sanskrit_sloka": True,
            "content": "Verse 1 content",
            "image_paths": []
        }
        chunk_2 = {
            "chunk_id": "hyp_ch02_v001_p01",
            "book_id": "hyp",
            "chapter_number": 2,
            "verse_start": 1,
            "has_sanskrit_sloka": False,
            "content": "Chapter 2 Pranayama",
            "image_paths": []
        }
        chunk_other_book = {
            "chunk_id": "loy_ch01_c001",
            "book_id": "light_on_yoga",
            "chapter_number": 1,
            "has_sanskrit_sloka": False,
            "content": "What is yoga narrative",
            "image_paths": []
        }

        # Mock 1536-dimensional embeddings for testing
        dummy_vec_1 = [0.1] * 1536
        dummy_vec_2 = [0.2] * 1536
        dummy_vec_3 = [0.3] * 1536

        col.add(
            ids=[chunk_1["chunk_id"], chunk_2["chunk_id"], chunk_other_book["chunk_id"]],
            documents=[chunk_1["content"], chunk_2["content"], chunk_other_book["content"]],
            metadatas=[
                ChromaManager.sanitize_metadata(chunk_1),
                ChromaManager.sanitize_metadata(chunk_2),
                ChromaManager.sanitize_metadata(chunk_other_book)
            ],
            embeddings=[dummy_vec_1, dummy_vec_2, dummy_vec_3]
        )

        self.assertEqual(col.count(), 3)

        # 1. Filter by book_id
        res_hyp = col.get(where={"book_id": "hyp"})
        self.assertEqual(len(res_hyp["ids"]), 2)

        # 2. Filter by chapter
        res_ch2 = col.get(where={"chapter_number": 2})
        self.assertEqual(len(res_ch2["ids"]), 1)
        self.assertEqual(res_ch2["ids"][0], "hyp_ch02_v001_p01")

        # 3. Filter by shloka
        res_shloka = col.get(where={"has_sanskrit_sloka": True})
        self.assertEqual(len(res_shloka["ids"]), 1)
        self.assertEqual(res_shloka["ids"][0], "hyp_ch01_v001_p01")

        # 4. Verify deserialization
        retrieved_meta = res_shloka["metadatas"][0]
        deserialized = ChromaManager.deserialize_metadata(retrieved_meta)
        self.assertEqual(deserialized["image_paths"], [])

    def test_targeted_book_scoped_purge(self):
        """Verify targeted purge deletes only chunks matching the book_id."""
        col = self.manager.get_or_create_collection()

        col.add(
            ids=["hyp_1", "hyp_2", "loy_1"],
            documents=["doc 1", "doc 2", "doc 3"],
            metadatas=[
                ChromaManager.sanitize_metadata({"book_id": "hyp", "image_paths": []}),
                ChromaManager.sanitize_metadata({"book_id": "hyp", "image_paths": []}),
                ChromaManager.sanitize_metadata({"book_id": "light_on_yoga", "image_paths": []})
            ],
            embeddings=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        )
        self.assertEqual(col.count(), 3)

        # Purge only 'hyp'
        self.manager.purge_book("hyp")

        # Check collection: hyp chunks gone, light_on_yoga chunk preserved
        remaining = col.get()
        self.assertEqual(len(remaining["ids"]), 1)
        self.assertEqual(remaining["ids"][0], "loy_1")
        self.assertEqual(remaining["metadatas"][0]["book_id"], "light_on_yoga")


if __name__ == "__main__":
    unittest.main()
