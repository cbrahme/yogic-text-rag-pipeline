#!/usr/bin/env python3
"""
Unit Test Suite for Prompt Architecture, Guardrails & Context Assembly (Step 8)
=============================================================================
Verifies:
1. `SYSTEM_PROMPT_TEMPLATE`: Mandatory citation contract and safety guardrail keywords.
2. `align_query_terminology`: Zero-annotation query-time Sanskrit mapping.
3. `reorder_chunks_chronologically`: Canonical scripture sorting and deduplication.
4. `format_context_for_prompt`: Delimited scripture excerpt rendering.
5. `build_rag_messages`: OpenAI chat completion message structure and history integration.
6. `extract_citations`: Robust regex extraction of scripture citations.
"""

import os
import sys
import unittest

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query_db import SearchResult
from prompt_engine import (
    SYSTEM_PROMPT_TEMPLATE,
    YOGA_SYNONYM_LEXICON,
    align_query_terminology,
    reorder_chunks_chronologically,
    format_context_for_prompt,
    build_rag_messages,
    extract_citations,
    _safe_int
)


class TestPromptArchitecture(unittest.TestCase):
    """Verifies system prompt contracts and safety guidelines."""

    def test_system_prompt_contracts(self):
        """Verifies core citation contracts and safety rules are present in the system prompt."""
        self.assertIn("MANDATORY CITATION CONTRACT", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("[Chapter X, Verse Y, pp. Z-W]", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("STRICT FACTUAL GROUNDING", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("GURU GUIDANCE & ADVANCED PRACTICE WARNINGS", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("Hatha Yoga Pradipika", SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("Swami Muktibodhananda", SYSTEM_PROMPT_TEMPLATE)


class TestDualLanguageAlignment(unittest.TestCase):
    """Verifies query-time terminology alignment across English and Sanskrit."""

    def test_align_single_term(self):
        terms = align_query_terminology("How do I practice the root lock properly?")
        self.assertEqual(terms, ["moola bandha"])

    def test_align_multiple_terms(self):
        query = "Explain the benefits of stomach cleansing and bellows breath."
        terms = align_query_terminology(query)
        self.assertIn("dhauti", terms)
        self.assertIn("bhastrika", terms)

    def test_align_asana_terms(self):
        query = "What is spinal twist and corpse pose?"
        terms = align_query_terminology(query)
        self.assertIn("matsyendrasana", terms)
        self.assertIn("shavasana", terms)

    def test_no_matches_returns_empty(self):
        terms = align_query_terminology("What is the meaning of life?")
        self.assertEqual(terms, [])


class TestContextAssembly(unittest.TestCase):
    """Verifies chronological reordering, deduplication, and excerpt formatting."""

    def setUp(self):
        # Create shuffled chunks spanning different chapters and sub-chunks
        self.chunk_ch3_p4 = SearchResult(
            chunk_id="hyp_ch03_v061-063_p04",
            content="Moola bandha in pranayama and kumbhaka.",
            metadata={"chapter_number": 3, "chapter_title": "Mudra", "verse_start": 61, "verse_end": 63, "verse_label": "61-63", "sub_chunk_index": 3, "total_sub_chunks": 4, "page_start": 315, "page_end": 318, "is_sub_chunk": True},
            score=0.9
        )
        self.chunk_ch1_v26 = SearchResult(
            chunk_id="hyp_ch01_v026_p01",
            content="Matsyendrasana technique.",
            metadata={"chapter_number": 1, "chapter_title": "Asana", "verse_start": 26, "verse_end": 26, "verse_label": "26", "sub_chunk_index": 0, "total_sub_chunks": 1, "page_start": 64, "page_end": 65, "is_sub_chunk": False},
            score=0.85
        )
        self.chunk_ch3_p1 = SearchResult(
            chunk_id="hyp_ch03_v061-063_p01",
            content="Moola bandha root sloka and translation.",
            metadata={"chapter_number": 3, "chapter_title": "Mudra", "verse_start": 61, "verse_end": 63, "verse_label": "61-63", "sub_chunk_index": 0, "total_sub_chunks": 4, "page_start": 315, "page_end": 318, "is_sub_chunk": True},
            score=0.88
        )
        self.chunk_ch2_v51 = SearchResult(
            chunk_id="hyp_ch02_v051-053_p01",
            content="Ujjayi pranayama root sloka.",
            metadata={"chapter_number": 2, "chapter_title": "Pranayama", "verse_start": 51, "verse_end": 53, "verse_label": "51-53", "sub_chunk_index": 0, "total_sub_chunks": 3, "page_start": 217, "page_end": 219, "is_sub_chunk": True},
            score=0.80
        )

    def test_reorder_chunks_chronologically(self):
        # Input order: [Ch3 Part 4, Ch1 V26, Ch3 Part 1, Ch2 V51]
        shuffled = [self.chunk_ch3_p4, self.chunk_ch1_v26, self.chunk_ch3_p1, self.chunk_ch2_v51]
        ordered = reorder_chunks_chronologically(shuffled)

        expected_ids = [
            "hyp_ch01_v026_p01",        # Chapter 1
            "hyp_ch02_v051-053_p01",    # Chapter 2
            "hyp_ch03_v061-063_p01",    # Chapter 3 Part 1
            "hyp_ch03_v061-063_p04",    # Chapter 3 Part 4
        ]
        self.assertEqual([c.chunk_id for c in ordered], expected_ids)

    def test_reorder_deduplication(self):
        # Duplicate of Ch1 V26
        duplicates = [self.chunk_ch1_v26, self.chunk_ch3_p1, self.chunk_ch1_v26]
        ordered = reorder_chunks_chronologically(duplicates)
        self.assertEqual(len(ordered), 2)
        self.assertEqual(ordered[0].chunk_id, "hyp_ch01_v026_p01")
        self.assertEqual(ordered[1].chunk_id, "hyp_ch03_v061-063_p01")

    def test_format_context_for_prompt(self):
        ordered = [self.chunk_ch1_v26, self.chunk_ch3_p1]
        context_str = format_context_for_prompt(ordered)

        self.assertIn("--- EXCERPT 1: Chapter 1 (Asana), Verse 26 | Pages 64-65 ---", context_str)
        self.assertIn("[Chunk ID: hyp_ch01_v026_p01]", context_str)
        self.assertIn("Matsyendrasana technique.", context_str)

        self.assertIn("--- EXCERPT 2: Chapter 3 (Mudra), Verse 61-63 (Part 1/4) | Pages 315-318 ---", context_str)
        self.assertIn("[Chunk ID: hyp_ch03_v061-063_p01]", context_str)
        self.assertIn("Moola bandha root sloka and translation.", context_str)

    def test_safe_int(self):
        """Verifies _safe_int fallback handling across None, strings, and bad inputs."""
        self.assertEqual(_safe_int(None, default=0), 0)
        self.assertEqual(_safe_int(None, default=42), 42)
        self.assertEqual(_safe_int(5), 5)
        self.assertEqual(_safe_int("10"), 10)
        self.assertEqual(_safe_int("invalid", default=99), 99)

    def test_reorder_narrative_chunks_with_none_verse_fields(self):
        """Verifies chunks from narrative/standard books with verse_start=None do not raise TypeError."""
        narrative_ch1 = SearchResult(
            chunk_id="loy_ch01_c001",
            content="Light on Yoga Chapter 1 text.",
            metadata={"chapter_number": 1, "chapter_title": "What is Yoga", "verse_start": None, "verse_end": None, "verse_label": None, "sub_chunk_index": 0, "total_sub_chunks": 2, "page_start": 1, "page_end": 10, "is_sub_chunk": True},
            score=0.9
        )
        narrative_ch2 = SearchResult(
            chunk_id="loy_ch02_c001",
            content="Light on Yoga Chapter 2 text.",
            metadata={"chapter_number": 2, "chapter_title": "Asana Section", "verse_start": None, "verse_end": None, "verse_label": None, "sub_chunk_index": 0, "total_sub_chunks": 1, "page_start": 11, "page_end": 20, "is_sub_chunk": False},
            score=0.8
        )
        shuffled = [narrative_ch2, narrative_ch1]
        ordered = reorder_chunks_chronologically(shuffled)
        self.assertEqual(len(ordered), 2)
        self.assertEqual(ordered[0].chunk_id, "loy_ch01_c001")
        self.assertEqual(ordered[1].chunk_id, "loy_ch02_c001")

    def test_format_context_for_narrative_chunks(self):
        """Verifies narrative chunks format cleanly without 'Verse None'."""
        narrative_chunk = SearchResult(
            chunk_id="loy_ch01_c001",
            content="What is Yoga content.",
            metadata={"chapter_number": 1, "chapter_title": "Introduction", "verse_start": None, "verse_end": None, "verse_label": None, "sub_chunk_index": 0, "total_sub_chunks": 1, "page_start": 1, "page_end": 5, "is_sub_chunk": False},
            score=0.9
        )
        formatted = format_context_for_prompt([narrative_chunk])
        self.assertIn("--- EXCERPT 1: Chapter 1 (Introduction) | Pages 1-5 ---", formatted)
        self.assertNotIn("Verse None", formatted)


class TestMessageBuilder(unittest.TestCase):
    """Verifies payload construction for OpenAI chat completions."""

    def test_build_rag_messages_structure(self):
        chunk = SearchResult(
            chunk_id="hyp_ch03_v061-063_p01",
            content="Moola bandha instruction.",
            metadata={"chapter_number": 3, "chapter_title": "Mudra", "verse_start": 61, "verse_end": 63, "verse_label": "61-63", "sub_chunk_index": 0, "total_sub_chunks": 4, "page_start": 315, "page_end": 318, "is_sub_chunk": True},
            score=0.9
        )
        query = "How to perform root lock?"
        messages = build_rag_messages(query=query, retrieved_chunks=[chunk])

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        user_msg = messages[1]["content"]
        self.assertIn("USER QUESTION:\nHow to perform root lock?", user_msg)
        self.assertIn("[Scriptural Terminology Note: The query mentions concepts corresponding to Sanskrit scripture terms: moola bandha]", user_msg)
        self.assertIn("RETRIEVED SCRIPTURAL CONTEXT (Hatha Yoga Pradipika):", user_msg)
        self.assertIn("--- EXCERPT 1: Chapter 3 (Mudra), Verse 61-63 (Part 1/4) | Pages 315-318 ---", user_msg)

    def test_build_rag_messages_with_chat_history(self):
        chunk = SearchResult(
            chunk_id="hyp_ch01_v026_p01",
            content="Matsyendrasana text.",
            metadata={"chapter_number": 1, "chapter_title": "Asana", "verse_start": 26, "verse_end": 26, "verse_label": "26", "sub_chunk_index": 0, "total_sub_chunks": 1, "page_start": 64, "page_end": 65, "is_sub_chunk": False},
            score=0.9
        )
        history = [
            {"role": "user", "content": "What is asana?"},
            {"role": "assistant", "content": "Asana is the first stage of Hatha Yoga [Chapter 1, Verse 17, pp. 48-49]."}
        ]
        messages = build_rag_messages(
            query="Tell me about Matsyendrasana.",
            retrieved_chunks=[chunk],
            chat_history=history
        )

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What is asana?")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("Tell me about Matsyendrasana.", messages[3]["content"])


class TestCitationExtractor(unittest.TestCase):
    """Verifies regex extraction and parsing of scripture citations."""

    def test_extract_single_and_multi_verse_citations(self):
        text = (
            "According to the scripture, Moola Bandha presses the perineum with the heel "
            "[Chapter 3, Verses 61-63, pp. 315-318]. Furthermore, failure in yoga occurs through overeating "
            "[Chapter 1, Verse 15, pp. 45-46], and kumbhaka control is detailed in [Chapter 2, Verse 5, p. 130]."
        )
        citations = extract_citations(text)
        self.assertEqual(len(citations), 3)

        # 1st citation
        self.assertEqual(citations[0]["chapter"], "3")
        self.assertEqual(citations[0]["verse"], "61-63")
        self.assertEqual(citations[0]["pages"], "315-318")

        # 2nd citation
        self.assertEqual(citations[1]["chapter"], "1")
        self.assertEqual(citations[1]["verse"], "15")
        self.assertEqual(citations[1]["pages"], "45-46")

        # 3rd citation (singular 'p.')
        self.assertEqual(citations[2]["chapter"], "2")
        self.assertEqual(citations[2]["verse"], "5")
        self.assertEqual(citations[2]["pages"], "130")

    def test_extract_no_citations(self):
        text = "This is a statement without any citations."
        citations = extract_citations(text)
        self.assertEqual(citations, [])


if __name__ == "__main__":
    unittest.main()
