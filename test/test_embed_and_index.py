#!/usr/bin/env python3
"""
Unit & Integration Test Suite for Batch Ingestion & Vector Indexing (Step 6)
============================================================================
Verifies:
1. `stream_chunks`: Streaming line-by-line reading with O(1) memory.
2. `batch_iterator`: Chunk partitioning into correct batch sizes.
3. `BookIndexer.dry_run`: Full pass validation without calling OpenAI API.
4. `BookIndexer` with Mocked OpenAI: Verifies batch embedding, token tracking,
   scoped purge execution, and ChromaDB collection upsert contracts.
5. Error handling: Missing API key and malformed JSONL records.
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

from embed_and_index import stream_chunks, batch_iterator, BookIndexer
from chroma_manager import ChromaManager


class TestBatchIngestionComponents(unittest.TestCase):
    """Tests chunk streaming and batch iteration logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_jsonl = os.path.join(self.temp_dir, "test_chunks.jsonl")

        self.sample_data = [
            {"chunk_id": f"hyp_ch01_v{i:03d}_p01", "book_id": "hyp", "content": f"Verse {i} commentary", "token_count": 100}
            for i in range(1, 11)
        ]
        with open(self.sample_jsonl, "w", encoding="utf-8") as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stream_chunks(self):
        """Verifies stream_chunks yields all items accurately without loading entire file at once."""
        streamed = list(stream_chunks(self.sample_jsonl))
        self.assertEqual(len(streamed), 10)
        self.assertEqual(streamed[0]["chunk_id"], "hyp_ch01_v001_p01")
        self.assertEqual(streamed[9]["chunk_id"], "hyp_ch01_v010_p01")

    def test_stream_chunks_malformed_json_raises(self):
        """Verifies malformed JSONL lines raise ValueError with line reference."""
        bad_jsonl = os.path.join(self.temp_dir, "bad.jsonl")
        with open(bad_jsonl, "w", encoding="utf-8") as f:
            f.write('{"valid": true}\n')
            f.write('NOT_VALID_JSON\n')

        with self.assertRaises(ValueError):
            list(stream_chunks(bad_jsonl))

    def test_batch_iterator_even_split(self):
        """Verifies batch_iterator cleanly divides elements into exact batch sizes."""
        items = list(range(10))
        batches = list(batch_iterator(iter(items), batch_size=5))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], [0, 1, 2, 3, 4])
        self.assertEqual(batches[1], [5, 6, 7, 8, 9])

    def test_batch_iterator_with_remainder(self):
        """Verifies batch_iterator handles batches with remainders correctly."""
        items = list(range(11))
        batches = list(batch_iterator(iter(items), batch_size=4))
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], [0, 1, 2, 3])
        self.assertEqual(batches[1], [4, 5, 6, 7])
        self.assertEqual(batches[2], [8, 9, 10])


class TestBookIndexer(unittest.TestCase):
    """Tests the BookIndexer workflow, dry runs, mocking, and ChromaDB persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_dir = os.path.join(self.temp_dir, "test_chroma_db")
        self.sample_jsonl = os.path.join(self.temp_dir, "test_chunks.jsonl")

        self.sample_data = [
            {
                "chunk_id": f"hyp_ch01_v{i:03d}_p01",
                "book_id": "hyp",
                "book_title": "Hatha Yoga Pradipika",
                "parent_verse_id": f"hyp_ch01_v{i:03d}",
                "chapter_number": 1,
                "chapter_title": "Asana",
                "verse_start": i,
                "verse_end": i,
                "verse_label": str(i),
                "is_sub_chunk": False,
                "sub_chunk_index": 0,
                "total_sub_chunks": 1,
                "has_sanskrit_sloka": True,
                "page_start": i,
                "page_end": i,
                "chunk_type": "verse",
                "char_count": 150,
                "token_count": 40,
                "image_paths": [],
                "content": f"[Hatha Yoga Pradipika | Chapter 1: Asana | Verse {i}]\nContent of verse {i}."
            }
            for i in range(1, 6)
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
                "collection_name": "test_yoga_scriptures",
                "distance_metric": "cosine"
            }
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dry_run_execution(self):
        """Verifies dry run processes all chunks without requiring API key or writing to DB."""
        indexer = BookIndexer(
            config_or_path=self.config,
            batch_size=2,
            purge=True,
            dry_run=True
        )
        report = indexer.index(self.sample_jsonl)

        self.assertEqual(report["total_indexed"], 5)
        self.assertEqual(report["total_tokens"], 0)
        self.assertEqual(report["estimated_cost"], 0.0)
        self.assertTrue(report["dry_run"])

    def test_missing_api_key_raises(self):
        """Verifies live mode raises error when OPENAI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                BookIndexer(config_or_path=self.config, api_key=None, dry_run=False)
            self.assertIn("OpenAI API key not found", str(ctx.exception))

    @patch("embed_and_index.OpenAI")
    def test_mocked_live_indexing(self, mock_openai_cls):
        """Verifies live indexing calls OpenAI embedding and ChromaDB upsert accurately."""
        # Setup mock OpenAI embeddings response
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def fake_embeddings_create(input, model):
            mock_resp = MagicMock()
            mock_items = []
            for _ in input:
                item = MagicMock()
                item.embedding = [0.01] * 1536
                mock_items.append(item)
            mock_resp.data = mock_items
            mock_resp.usage.total_tokens = len(input) * 50
            return mock_resp

        mock_client.embeddings.create.side_effect = fake_embeddings_create

        indexer = BookIndexer(
            config_or_path=self.config,
            api_key="sk-mock-key-for-unit-test",
            batch_size=2,
            purge=True,
            dry_run=False
        )

        report = indexer.index(self.sample_jsonl)

        self.assertEqual(report["total_indexed"], 5)
        self.assertEqual(report["total_tokens"], 250)
        self.assertGreater(report["estimated_cost"], 0.0)
        self.assertFalse(report["dry_run"])

        # Verify ChromaDB collection has 5 items
        manager = ChromaManager(self.config)
        col = manager.collection
        self.assertEqual(col.count(), 5)

        # Verify retrieved metadata
        sample_meta = col.get(ids=["hyp_ch01_v001_p01"])["metadatas"][0]
        self.assertEqual(sample_meta["book_id"], "hyp")
        self.assertEqual(sample_meta["verse_start"], 1)
        self.assertEqual(sample_meta["image_paths"], "[]")

    @patch("embed_and_index.OpenAI")
    def test_targeted_book_purge_on_reindex(self, mock_openai_cls):
        """Verifies that purge=True removes existing book vectors before re-inserting."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def fake_embeddings_create(input, model):
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.01] * 1536) for _ in input]
            mock_resp.usage.total_tokens = 100
            return mock_resp

        mock_client.embeddings.create.side_effect = fake_embeddings_create

        indexer = BookIndexer(
            config_or_path=self.config,
            api_key="sk-mock-key",
            batch_size=5,
            purge=True,
            dry_run=False
        )

        # Index once
        indexer.index(self.sample_jsonl)
        manager = ChromaManager(self.config)
        self.assertEqual(manager.collection.count(), 5)

        # Re-index should purge first and not create duplicates
        indexer.index(self.sample_jsonl)
        self.assertEqual(manager.collection.count(), 5)


if __name__ == "__main__":
    unittest.main()
