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

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

import httpx
import openai

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query_db import SearchResult
import chat
from chat import ChatSession
from chroma_manager import IndexNotFoundError
from error_messages import EmptyResponseError


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

    def _delta(self, text):
        return MagicMock(choices=[MagicMock(delta=MagicMock(content=text))])

    def test_failure_mid_stream_leaves_history_unchanged(self):
        """A connection drop mid-answer raises and does not record a half turn."""
        self.mock_retriever.search.return_value = []

        def failing_stream():
            yield self._delta("Moola Bandha is ")
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

        self.session.client.chat.completions.create.return_value = failing_stream()
        with self.assertRaises(openai.APIConnectionError):
            list(self.session.stream_answer("How to perform Moola Bandha?"))
        self.assertEqual(self.session.history, [])

    def test_empty_response_raises(self):
        self.mock_retriever.search.return_value = []
        self.session.client.chat.completions.create.return_value = iter([self._delta("  ")])
        with self.assertRaises(EmptyResponseError):
            list(self.session.stream_answer("What is Mithahara?"))
        self.assertEqual(self.session.history, [])


class TestChatCLIErrors(unittest.TestCase):
    """Verifies the CLI exits cleanly with friendly messages instead of tracebacks."""

    def _run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["chat.py"] + argv), redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                chat.main()
        return ctx.exception.code, out.getvalue(), err.getvalue()

    @patch("chat.ChatSession")
    def test_single_query_openai_error_exits_with_message(self, mock_session_cls):
        session = MagicMock()
        session.stream_answer.side_effect = openai.AuthenticationError(
            "bad key", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None)
        mock_session_cls.return_value = session

        code, _, err = self._run_main(["--query", "How to perform Moola Bandha?"])
        self.assertEqual(code, 1)
        self.assertIn("API key was rejected", err)
        self.assertNotIn("Traceback", err)

    @patch("chat.ChatSession", side_effect=IndexNotFoundError("No ChromaDB index found at '/x/chroma_db'."))
    def test_startup_without_index_exits_with_message(self, _):
        code, _, err = self._run_main(["--query", "anything"])
        self.assertEqual(code, 1)
        self.assertIn("Could not start the assistant", err)
        self.assertIn("/x/chroma_db", err)

    @patch("chat.ChatSession", side_effect=ValueError("OPENAI_API_KEY not found."))
    def test_startup_without_api_key_exits_with_message(self, _):
        code, _, err = self._run_main([])
        self.assertEqual(code, 1)
        self.assertIn("OPENAI_API_KEY not found", err)

    @patch("builtins.input", side_effect=["How to perform Moola Bandha?", "exit"])
    def test_repl_survives_error_and_continues(self, _):
        session = MagicMock(model="gpt-4o-mini", top_k=4, chapter_filter=None, history=[])
        session.stream_answer.side_effect = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            chat.run_chat_loop(session)  # must return normally on "exit"
        self.assertIn("Could not reach OpenAI", err.getvalue())
        self.assertIn("you can ask it again", out.getvalue())
        self.assertIn("Om Shanti", out.getvalue())


if __name__ == "__main__":
    unittest.main()
