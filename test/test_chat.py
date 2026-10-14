#!/usr/bin/env python3
"""
Unit Test Suite for Conversational Chat Assistant (Step 9 & 10)
==============================================================
Verifies:
1. `ChatSession` initialization and default state.
2. Pronoun and contextual query enrichment (`_resolve_conversational_query`).
3. Multi-turn history accumulation and memory clearing.
4. Source attribution persistence across turns.
5. Mocked chat completion streaming and citation extraction.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query_db import SearchResult
from chat import ChatSession


class TestChatSession(unittest.TestCase):
    """Verifies ChatSession conversation management logic."""

    @patch("chat.OpenAI")
    @patch("chat.HybridRetriever")
    def setUp(self, mock_retriever_cls, mock_openai_cls):
        self.mock_retriever = MagicMock()
        mock_retriever_cls.return_value = self.mock_retriever

        self.mock_openai = MagicMock()
        mock_openai_cls.return_value = self.mock_openai

        self.session = ChatSession(
            config_path="configs/hatha_yoga_pradipika.yaml",
            api_key="sk-fake-test-key",
            model="gpt-4o-mini",
            temperature=0.2,
            top_k=3
        )

    def test_session_init(self):
        """Verifies session initializes with clean history and configured hyperparameters."""
        self.assertEqual(self.session.model, "gpt-4o-mini")
        self.assertEqual(self.session.temperature, 0.2)
        self.assertEqual(self.session.top_k, 3)
        self.assertEqual(len(self.session.history), 0)
        self.assertIsNone(self.session.chapter_filter)

    def test_resolve_conversational_query(self):
        """Verifies follow-up queries with pronouns resolve previous turn context."""
        # 1. No prior query
        self.assertEqual(
            self.session._resolve_conversational_query("What is it?"),
            "What is it?"
        )

        # 2. Prior query about Moola Bandha
        self.session.last_query = "How to perform the root lock?"
        resolved = self.session._resolve_conversational_query("What are its benefits?")
        self.assertIn("moola bandha", resolved)

        # 3. Independent long query should not be modified
        independent_q = "Explain the causes of failure in yoga according to Chapter 1"
        self.assertEqual(
            self.session._resolve_conversational_query(independent_q),
            independent_q
        )

    def test_clear_history(self):
        """Verifies memory clearing resets history, last sources, and citations."""
        self.session.history = [{"role": "user", "content": "hello"}]
        self.session.last_query = "hello"
        self.session.last_sources = [MagicMock()]
        self.session.last_citations = [{"raw": "[Chapter 1, Verse 1, pp. 1-2]"}]

        self.session.clear_history()
        self.assertEqual(len(self.session.history), 0)
        self.assertIsNone(self.session.last_query)
        self.assertEqual(len(self.session.last_sources), 0)
        self.assertEqual(len(self.session.last_citations), 0)

    def test_stream_answer_flow(self):
        """Verifies end-to-end streaming token delivery and history recording."""
        mock_chunk = SearchResult(
            chunk_id="hyp_ch01_v026_p01",
            content="Matsyendrasana pose.",
            metadata={"chapter_number": 1, "chapter_title": "Asana", "verse_start": 26, "verse_end": 26, "verse_label": "26", "page_start": 64, "page_end": 65},
            score=0.9
        )
        self.mock_retriever.search.return_value = [mock_chunk]

        # Mock OpenAI streaming chunk deltas
        token_deltas = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Matsyendrasana "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="increases digestive fire "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="[Chapter 1, Verse 26, pp. 64-65]."))]),
        ]
        self.session.client.chat.completions.create.return_value = iter(token_deltas)

        tokens = list(self.session.stream_answer("What is Matsyendrasana?"))
        full_text = "".join(tokens)

        # Check streamed text
        self.assertIn("Matsyendrasana increases digestive fire", full_text)
        self.assertIn("[Chapter 1, Verse 26, pp. 64-65].", full_text)

        # Check history recorded
        self.assertEqual(len(self.session.history), 2)
        self.assertEqual(self.session.history[0]["role"], "user")
        self.assertEqual(self.session.history[1]["role"], "assistant")

        # Check citations extracted
        self.assertEqual(len(self.session.last_citations), 1)
        self.assertEqual(self.session.last_citations[0]["chapter"], "1")
        self.assertEqual(self.session.last_citations[0]["verse"], "26")
        self.assertEqual(self.session.last_citations[0]["pages"], "64-65")


if __name__ == "__main__":
    unittest.main()
