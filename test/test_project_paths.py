#!/usr/bin/env python3
"""
Unit tests for working-directory-independent path handling:
1. Config files resolve from the current directory or the project root.
2. Relative data paths in configs resolve against the project root.
3. Read-only index access never creates an empty ChromaDB index.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from project_paths import (
    PROJECT_ROOT,
    DEFAULT_CONFIG_PATH,
    resolve_config_path,
    resolve_data_path,
    resolve_config_data_paths,
)
from chroma_manager import ChromaManager, IndexNotFoundError, load_yaml_config
from ingestion_pipeline import load_config as load_ingestion_config
from query_db import HybridRetriever


class _InOtherDirectory(unittest.TestCase):
    """Runs each test from an unrelated temporary working directory."""

    def setUp(self):
        self.orig_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestPathResolution(_InOtherDirectory):

    def test_default_config_resolves_from_any_directory(self):
        resolved = resolve_config_path("configs/hatha_yoga_pradipika.yaml")
        self.assertEqual(resolved, str(DEFAULT_CONFIG_PATH))
        self.assertTrue(os.path.exists(resolved))

    def test_config_relative_to_cwd_takes_precedence(self):
        os.makedirs("configs")
        local = os.path.join("configs", "hatha_yoga_pradipika.yaml")
        with open(local, "w") as f:
            f.write("book_id: local\n")
        self.assertEqual(resolve_config_path(local), os.path.realpath(local))

    def test_loaded_config_has_project_root_data_paths(self):
        for loader in (load_yaml_config, load_ingestion_config):
            config = loader("configs/hatha_yoga_pradipika.yaml")
            self.assertEqual(config["vector_db"]["persist_directory"], str(PROJECT_ROOT / "chroma_db"))
            self.assertEqual(config["output_path"], str(PROJECT_ROOT / "chunks.jsonl"))
            self.assertEqual(config["pdf_path"], str(PROJECT_ROOT / "docs" / "Hatha-Yoga-Pradipika-pages-2.pdf"))

    def test_absolute_paths_untouched_and_input_not_mutated(self):
        config = {"output_path": "/abs/chunks.jsonl", "vector_db": {"persist_directory": "rel_db"}}
        resolved = resolve_config_data_paths(config)
        self.assertEqual(resolved["output_path"], "/abs/chunks.jsonl")
        self.assertEqual(resolved["vector_db"]["persist_directory"], str(PROJECT_ROOT / "rel_db"))
        self.assertEqual(config["vector_db"]["persist_directory"], "rel_db")
        self.assertEqual(resolve_data_path("/x/y"), "/x/y")

    def test_missing_config_reports_where_it_looked(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_yaml_config("configs/does_not_exist.yaml")
        self.assertIn(str(PROJECT_ROOT), str(ctx.exception))


class TestReadOnlyIndexAccess(_InOtherDirectory):

    def _config(self, db_dir):
        return {
            "book_id": "hyp",
            "vector_db": {"persist_directory": db_dir, "collection_name": "test_col"},
        }

    def test_missing_index_raises_without_creating_directory(self):
        db_dir = os.path.join(self.temp_dir, "no_db_here")
        manager = ChromaManager(self._config(db_dir), create_if_missing=False)
        with self.assertRaises(IndexNotFoundError) as ctx:
            _ = manager.collection
        self.assertIn(db_dir, str(ctx.exception))
        self.assertIn("embed_and_index.py", str(ctx.exception))
        self.assertFalse(os.path.exists(db_dir))

    def test_missing_collection_raises_without_creating_it(self):
        db_dir = os.path.join(self.temp_dir, "db")
        ChromaManager(self._config(db_dir)).get_or_create_collection(name="other_col")
        reader = ChromaManager(self._config(db_dir), create_if_missing=False)
        with self.assertRaises(IndexNotFoundError):
            _ = reader.collection
        names = [c.name if hasattr(c, "name") else c for c in ChromaManager(self._config(db_dir)).client.list_collections()]
        self.assertNotIn("test_col", names)

    def test_retriever_with_jsonl_needs_no_index_for_bm25(self):
        db_dir = os.path.join(self.temp_dir, "no_db_here")
        jsonl = os.path.join(self.temp_dir, "chunks.jsonl")
        with open(jsonl, "w") as f:
            # BM25 needs several documents for non-zero IDF
            for cid, text in [("hyp_ch01_v001_p01", "Moola bandha perineum"),
                              ("hyp_ch02_v001_p01", "Ujjayi psychic breath"),
                              ("hyp_ch04_v001_p01", "Samadhi absorption")]:
                f.write(json.dumps({"chunk_id": cid, "chapter_number": 1, "content": text}) + "\n")
        retriever = HybridRetriever(self._config(db_dir), chunks_jsonl_path=jsonl)
        hits = retriever.search_bm25("moola bandha", top_k=1)
        self.assertEqual(hits[0].chunk_id, "hyp_ch01_v001_p01")
        self.assertFalse(os.path.exists(db_dir))

    def test_retriever_without_index_raises_clear_error(self):
        db_dir = os.path.join(self.temp_dir, "no_db_here")
        with self.assertRaises(IndexNotFoundError):
            HybridRetriever(self._config(db_dir))
        self.assertFalse(os.path.exists(db_dir))


if __name__ == "__main__":
    unittest.main()
