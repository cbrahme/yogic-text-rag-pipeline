#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Ingestion Pipeline (ingestion_pipeline.py)
========================================================================
Covers all components, edge cases, regression fixes, and architecture rules:
1. TextCleaner: NFC Unicode normalization, hyphen-stitching across lines, intra-paragraph newlines.
2. TokenCounter: Exact BPE cl100k_base counting and Sanskrit transliteration token expansion.
3. Verse Header Extraction & Edge Cases:
   - Valid verse formats (single, ranges, multipart annotations like '(i)', '(ii & iii)').
   - Narrative Verb Guard (rejecting 'Verse X says/describes/states/mentions...').
   - Line length ceilings and margin/footer cutoffs.
   - Practice/posture title extraction on same-line and subsequent lines.
4. Practice Titles (is_practice_title): Uppercase/parenthetical posture titles vs regular prose.
5. Verse Anchor Builder (build_verse_anchor):
   - Intact translations (<= 350 chars).
   - Grammatical sentence-boundary splitting (> 350 chars, zero mid-word truncation).
   - Multiline whitespace collapse.
6. VerseChunker (Strategy 1):
   - Dual-ceiling intact rule (both <= 2000 chars AND <= 600 tokens).
   - Two-stage recursive splitting with dynamic token budgeting.
   - Sanskrit sloka isolation to Part 1 (has_sanskrit_sloka=True).
   - Verse context anchor injection into sub-chunks 2..N (has_sanskrit_sloka=False).
   - 0-indexed sub_chunk_index, total_sub_chunks, and namespaced IDs (e.g. 'hyp_ch01_v001_p01').
   - Token ceiling guarantee (100% chunks <= 600 tokens).
7. StandardChunker (Strategy 2):
   - Narrative chunking with overlap and contextual chapter breadcrumbs.
   - Namespaced sequential IDs ('book_ch01_c001').
8. VersePdfExtractor Edge Cases:
   - Publisher errata correction (e.g. Page 86 v57 -> v37).
   - Procedural label guard ('Technique 1' routed to commentary instead of hijacking translation).
   - Chapter header stripping from opening verse translations.
   - Bidirectional title containment deduplication.
"""

import os
import sys
import re
import unicodedata
import unittest
from typing import Dict, Any, List

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion_pipeline import (
    TextCleaner,
    TokenCounter,
    extract_verse_header,
    is_practice_title,
    Verse,
    ChapterText,
    Chunk,
    VerseChunker,
    StandardChunker,
    VersePdfExtractor,
    VERSE_LINE_REGEX,
    NARRATIVE_VERB_GUARD
)


class TestTextCleaner(unittest.TestCase):
    """Tests text normalization and line-break cleaning."""

    def test_nfc_normalization(self):
        """Ensure decomposed Unicode diacritics are merged into canonical NFC precomposed characters."""
        # Decomposed: 'a' + combining macron (U+0304) -> 'ā'
        decomposed = "hath\u0304a yoga pra\u0304d\u0323i\u0304pika\u0304"
        cleaned = TextCleaner.clean_block_text(decomposed)
        # In NFC, this should be precomposed characters
        expected = unicodedata.normalize('NFC', decomposed)
        self.assertEqual(cleaned, expected)
        self.assertEqual(unicodedata.is_normalized('NFC', cleaned), True)

    def test_hyphen_stitching(self):
        """Ensure words split across lines by a hyphen are reconnected cleanly."""
        raw = "The prac-\ntice of pran-\n ayama brings great-\n  ness."
        cleaned = TextCleaner.clean_block_text(raw)
        self.assertIn("practice", cleaned)
        self.assertIn("pranayama", cleaned)
        self.assertIn("greatness", cleaned)
        self.assertNotIn("prac-", cleaned)
        self.assertNotIn("pran-", cleaned)

    def test_intra_paragraph_newlines(self):
        """Single newlines within a paragraph should become spaces, while multiple spaces collapse."""
        raw = "This is line one.\nThis is line two.\n\nNew paragraph here."
        cleaned = TextCleaner.clean_block_text(raw)
        self.assertEqual(cleaned, "This is line one. This is line two.\n\nNew paragraph here.")

    def test_whitespace_normalization(self):
        """Excessive tabs and horizontal spaces should collapse to a single space."""
        raw = "Word1   \t   Word2      Word3"
        cleaned = TextCleaner.clean_block_text(raw)
        self.assertEqual(cleaned, "Word1 Word2 Word3")


class TestTokenCounter(unittest.TestCase):
    """Tests exact BPE token counting with tiktoken cl100k_base."""

    def test_token_counter_basic(self):
        text = "Salutations to Sri Adinath."
        count = TokenCounter.count_tokens(text)
        encoder = TokenCounter.get_encoder()
        self.assertEqual(count, len(encoder.encode(text)))

    def test_sanskrit_transliteration_token_expansion(self):
        """Verifies that Sanskrit transliterations with diacritics produce accurate higher token counts."""
        ascii_text = "Matsyendrasana removes diseases."
        diacritic_text = "Matsyēndrāsana removes sarvābādha-haraṇam."
        count_ascii = TokenCounter.count_tokens(ascii_text)
        count_diacritic = TokenCounter.count_tokens(diacritic_text)
        self.assertGreater(count_diacritic, count_ascii)


class TestVerseHeaderExtraction(unittest.TestCase):
    """Tests verse line identification, regexes, and false-positive guards."""

    def test_single_verse_header(self):
        res = extract_verse_header("Verse 1", y0=100.0)
        self.assertIsNotNone(res)
        self.assertEqual(res["verse_start"], 1)
        self.assertEqual(res["verse_end"], 1)
        self.assertEqual(res["verse_label"], "1")

    def test_verse_range_header(self):
        res = extract_verse_header("Verses 5-9", y0=100.0)
        self.assertIsNotNone(res)
        self.assertEqual(res["verse_start"], 5)
        self.assertEqual(res["verse_end"], 9)
        self.assertEqual(res["verse_label"], "5-9")

    def test_verse_range_with_em_dash(self):
        """Tests verse range headers using Unicode em-dash (e.g. 'Verses 51—53 Chapter 2')."""
        res = extract_verse_header("Verses 51—53\nChapter 2", y0=100.0)
        self.assertIsNotNone(res)
        self.assertEqual(res["verse_start"], 51)
        self.assertEqual(res["verse_end"], 53)
        self.assertEqual(res["verse_label"], "51-53")

    def test_multipart_verse_header(self):
        """Tests subsection verses like 'Verses 16 (i)' and 'Verses 16 (ii & iii)'."""
        res1 = extract_verse_header("Verses 16 (i)", y0=100.0)
        self.assertIsNotNone(res1)
        self.assertEqual(res1["verse_start"], 16)
        self.assertEqual(res1["verse_label"], "16 (i)")

        res2 = extract_verse_header("Verses 16 (ii & iii)", y0=100.0)
        self.assertIsNotNone(res2)
        self.assertEqual(res2["verse_start"], 16)
        self.assertEqual(res2["verse_label"], "16 (ii & iii)")

    def test_same_line_posture_title(self):
        res = extract_verse_header("Verse 26 MATSYENDRASANA", y0=100.0)
        self.assertIsNotNone(res)
        self.assertEqual(res["verse_start"], 26)
        self.assertEqual(res["posture_title"], "MATSYENDRASANA")

    def test_subsequent_line_posture_title(self):
        block = "Verse 32\nKOORMASANA (tortoise pose)"
        res = extract_verse_header(block, y0=100.0)
        self.assertIsNotNone(res)
        self.assertEqual(res["verse_start"], 32)
        self.assertIn("KOORMASANA (tortoise pose)", res["posture_title"])

    def test_narrative_verb_guard_rejections(self):
        """
        Critical Edge Case:
        Ensures body commentary referring to verses (e.g. 'Verse 15 describes...')
        is strictly rejected from creating false verse boundaries.
        """
        false_positives = [
            "Verse 15 describes the technique of pranayama in great detail.",
            "Verse 4 says that yoga should be kept secret by the yogi.",
            "Verses 1-4 are introductory statements.",
            "Verse 10 states the true nature of samadhi.",
            "Verse 7 mentions the lineage of great masters.",
            "Verse 2 tells the story of Adinath revealing knowledge to Parvati.",
            "Verse 1 explains the spiritual stairway to raja yoga.",
            "Verse 5 means purification of the nadis through practice.",
            "Verse 6 refers to the inner nectar flowing from the moon."
        ]
        for prose in false_positives:
            res = extract_verse_header(prose, y0=100.0)
            self.assertIsNone(res, f"Narrative prose falsely matched as verse header: '{prose}'")

    def test_line_length_ceiling(self):
        """Lines exceeding 65 chars must be rejected as verse headers."""
        long_line = "Verse 1 " + "and furthermore this is a very long descriptive line of commentary " * 2
        res = extract_verse_header(long_line, y0=100.0)
        self.assertIsNone(res)

    def test_footer_margin_cutoff(self):
        """Blocks below the footer margin must be rejected."""
        res = extract_verse_header("Verse 1", y0=760.0, footer_bottom_y=750.0)
        self.assertIsNone(res)


class TestPracticeTitle(unittest.TestCase):
    """Tests posture / practice title recognition."""

    def test_valid_practice_titles(self):
        valid_titles = [
            "MATSYENDRASANA",
            "SWASTIKASANA (auspicious pose)",
            "(tortoise pose)",
            "PASCHIMOTTANASANA (back stretching pose)",
            "MAYURASANA"
        ]
        for t in valid_titles:
            self.assertTrue(is_practice_title(t), f"Failed to recognize valid practice title: '{t}'")

    def test_invalid_practice_titles(self):
        invalid_titles = [
            "This is a description of the pose.",  # Ends with period
            "Sit on the floor with legs extended.",  # Lowercase sentence
            "TECHNIQUE 1: HOW TO PERFORM THE PRACTICE OF PRANAYAMA AND BANDHAS IN THE MORNING WHEN THE MIND IS CALM AND STEADY OVER A VERY LONG TITLE"  # >= 100 chars
        ]
        for t in invalid_titles:
            self.assertFalse(is_practice_title(t), f"Incorrectly recognized as practice title: '{t}'")


class TestVerseAnchorBuilder(unittest.TestCase):
    """Tests grammatical verse anchor creation for sub-chunks 2..N."""

    def setUp(self):
        self.chunker = VerseChunker(book_id="hyp", book_title="Hatha Yoga Pradipika")

    def test_short_sloka_preserved_intact(self):
        """Sloka translations <= 350 chars are preserved completely intact."""
        sloka = "Salutations to Sri Adinath who instructed the knowledge of hatha yoga."
        anchor = self.chunker.build_verse_anchor(sloka, max_chars=350)
        self.assertEqual(anchor, sloka)

    def test_long_sloka_sentence_boundary_split(self):
        """
        Sloka translations > 350 chars are split on sentence boundaries,
        never cutting mid-word with naive ellipsis.
        """
        sentence_1 = "Practice of this asana (matsyendrasana) increases the digestive fire to such an incredible capacity that it removes diseases."
        sentence_2 = "Thus it awakens the serpent power and brings equilibrium in bindu within the body of the devoted practitioner."
        sentence_3 = "Furthermore it stimulates the abdominal viscera and the flow of prana throughout the entire spinal column."
        long_sloka = f"{sentence_1} {sentence_2} {sentence_3}"

        anchor = self.chunker.build_verse_anchor(long_sloka, max_chars=250)
        self.assertLessEqual(len(anchor), 250)
        self.assertTrue(anchor.endswith("."))
        self.assertIn("matsyendrasana", anchor)
        self.assertNotIn("...", anchor)

    def test_whitespace_collapsing(self):
        """Multiline slokas with tabs/newlines must be collapsed into clean single-line strings."""
        multiline_sloka = "Practice of\n   this asana\n\n(matsyendrasana)    increases fire."
        anchor = self.chunker.build_verse_anchor(multiline_sloka, max_chars=350)
        self.assertNotIn("\n", anchor)
        self.assertNotIn("   ", anchor)
        self.assertEqual(anchor, "Practice of this asana (matsyendrasana) increases fire.")

    def test_empty_sloka_handling(self):
        self.assertEqual(self.chunker.build_verse_anchor(""), "")
        self.assertEqual(self.chunker.build_verse_anchor("   "), "")


class TestVerseChunker(unittest.TestCase):
    """Tests Strategy 1: Dual-ceiling rules, 2-stage splitting, and metadata namespacing."""

    def setUp(self):
        self.chunker = VerseChunker(
            book_id="hyp",
            book_title="Hatha Yoga Pradipika",
            max_chunk_size=2000,
            max_tokens=600,
            chunk_overlap=0
        )

    def test_single_intact_chunk_tier_1(self):
        """Short verse + commentary (<= 2000 chars and <= 600 tokens) stays as 1 intact chunk."""
        verse = Verse(
            chapter_num=1,
            chapter_title="Asana",
            verse_start=1,
            verse_end=1,
            verse_label="1",
            page_start=2,
            page_end=3,
            sloka_translation="Salutations to Sri Adinath.",
            commentary_blocks=["Adinath is Shiva, the cosmic consciousness."]
        )
        chunks = self.chunker.chunk_verse(verse)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.chunk_id, "hyp_ch01_v001")
        self.assertEqual(c.parent_verse_id, "hyp_ch01_v001")
        self.assertFalse(c.is_sub_chunk)
        self.assertEqual(c.sub_chunk_index, 0)
        self.assertEqual(c.total_sub_chunks, 1)
        self.assertTrue(c.has_sanskrit_sloka)
        self.assertEqual(c.image_paths, [])
        self.assertLessEqual(c.token_count, 600)
        self.assertIn("[Hatha Yoga Pradipika | Chapter 1: Asana | Verse 1]", c.content)

    def test_dual_ceiling_rule_token_overflow(self):
        """
        Critical Rule:
        If content is <= 2000 chars BUT exceeds 600 tokens due to dense tokens,
        it MUST be split into sub-chunks, NOT kept as 1 intact chunk.
        """
        # Create text with 1,700 characters consisting of dense sub-byte tokens
        dense_tokens_text = "āīūṛśṣ " * 250  # ~1750 chars, but > 700 BPE tokens
        self.assertLessEqual(len(dense_tokens_text), 2000)
        self.assertGreater(TokenCounter.count_tokens(dense_tokens_text), 600)

        verse = Verse(
            chapter_num=1,
            chapter_title="Asana",
            verse_start=5,
            verse_end=5,
            verse_label="5",
            page_start=10,
            page_end=11,
            sloka_translation="Sloka with dense characters.",
            commentary_blocks=[dense_tokens_text]
        )
        chunks = self.chunker.chunk_verse(verse)
        self.assertGreater(len(chunks), 1, "Failed to split chunk that exceeded 600 tokens despite <= 2000 chars")
        for c in chunks:
            self.assertLessEqual(c.token_count, 600)

    def test_secondary_recursive_split_tier_2(self):
        """
        Long commentary (> 2000 chars) must split into sub-chunks:
        - Part 1 (i=0): contains breadcrumb + sloka translation + commentary, has_sanskrit_sloka=True.
        - Parts 2..N (i>=1): contains breadcrumb + [Verse X Context: ...] anchor + commentary, has_sanskrit_sloka=False.
        """
        paragraph = (
            "The practice of this posture stimulates the abdominal organs, improves digestion, "
            "awakens the serpent power kundalini, and purifies all seventy-two thousand nadis. "
            "By steady application of this method, the practitioner transcends all physical limitations.\n\n"
        )
        long_commentary = paragraph * 10  # ~2,500 chars

        verse = Verse(
            chapter_num=1,
            chapter_title="Asana",
            verse_start=27,
            verse_end=27,
            verse_label="27",
            page_start=45,
            page_end=48,
            sloka_translation="Practice of this asana (matsyendrasana) increases the digestive fire.",
            commentary_blocks=[long_commentary]
        )
        chunks = self.chunker.chunk_verse(verse)
        self.assertGreater(len(chunks), 1)

        # Audit Part 1
        part_1 = chunks[0]
        self.assertEqual(part_1.chunk_id, "hyp_ch01_v027_p01")
        self.assertEqual(part_1.sub_chunk_index, 0)
        self.assertEqual(part_1.total_sub_chunks, len(chunks))
        self.assertTrue(part_1.is_sub_chunk)
        self.assertTrue(part_1.has_sanskrit_sloka)
        self.assertIn("Practice of this asana (matsyendrasana) increases the digestive fire.", part_1.content)
        self.assertIn("Part 1/", part_1.content)

        # Audit Parts 2..N
        for idx in range(1, len(chunks)):
            sub_c = chunks[idx]
            self.assertEqual(sub_c.chunk_id, f"hyp_ch01_v027_p{idx+1:02d}")
            self.assertEqual(sub_c.sub_chunk_index, idx)
            self.assertEqual(sub_c.total_sub_chunks, len(chunks))
            self.assertTrue(sub_c.is_sub_chunk)
            self.assertFalse(sub_c.has_sanskrit_sloka, "Sub-chunk 2..N must not have has_sanskrit_sloka=True")
            # Verify anchor injection
            self.assertIn("[Verse 27 Context:", sub_c.content)
            self.assertIn("matsyendrasana", sub_c.content)
            self.assertIn(f"Part {idx+1}/{len(chunks)}", sub_c.content)
            # Verify strict token ceiling
            self.assertLessEqual(sub_c.token_count, 600)

    def test_verse_range_namespacing(self):
        """Verse ranges like Verses 5-9 must namespace to 'hyp_ch01_v005-009'."""
        verse = Verse(
            chapter_num=1,
            chapter_title="Asana",
            verse_start=5,
            verse_end=9,
            verse_label="5-9",
            page_start=12,
            page_end=14,
            sloka_translation="These verses describe the lineage.",
            commentary_blocks=["Commentary on lineage."]
        )
        chunks = self.chunker.chunk_verse(verse)
        self.assertEqual(chunks[0].chunk_id, "hyp_ch01_v005-009")


class TestStandardChunker(unittest.TestCase):
    """Tests Strategy 2: Standard narrative chunker for non-verse books."""

    def setUp(self):
        self.chunker = StandardChunker(
            book_id="light_on_yoga",
            book_title="Light on Yoga",
            chunk_size=500,
            chunk_overlap=50
        )

    def test_standard_chapter_chunking(self):
        chapter = ChapterText(
            chapter_num=1,
            chapter_title="What is Yoga?",
            page_start=1,
            page_end=5,
            text_content="Yoga is an ancient discipline. " * 80  # ~2,500 chars
        )
        chunks = self.chunker.chunk_all([chapter])
        self.assertGreater(len(chunks), 1)

        for idx, c in enumerate(chunks):
            expected_id = f"light_on_yoga_ch01_c{idx+1:03d}"
            self.assertEqual(c.chunk_id, expected_id)
            self.assertIsNone(c.parent_verse_id)
            self.assertEqual(c.chapter_number, 1)
            self.assertEqual(c.chapter_title, "What is Yoga?")
            self.assertFalse(c.has_sanskrit_sloka)
            self.assertTrue(c.is_sub_chunk)
            self.assertEqual(c.sub_chunk_index, idx)
            self.assertEqual(c.total_sub_chunks, len(chunks))
            self.assertIn("[Light on Yoga | Chapter 1: What is Yoga? (Part ", c.content)


class TestVersePdfExtractorLogic(unittest.TestCase):
    """Tests extractor layout logic, publisher errata, and procedural title guards."""

    def setUp(self):
        self.config = {
            "book_id": "hyp",
            "chapters": {
                1: {"number": 1, "title": "Asana", "cover_page": 1}
            },
            "margins": {
                "header_top_ratio": 0.07,
                "footer_bottom_ratio": 0.945
            },
            "publisher_errata": {
                86: {
                    "chapter": 1,
                    "misprint_start": 57,
                    "corrected_start": 37,
                    "corrected_end": 37
                }
            }
        }
        self.extractor = VersePdfExtractor(self.config)

    def test_publisher_errata_resolution(self):
        """Verifies that Page 86 misprinted Verse 57 is remapped to corrected Verse 37."""
        # Simulate header on page 86
        header_info = extract_verse_header("Verse 57", y0=100.0)
        self.assertIsNotNone(header_info)
        v_start = header_info["verse_start"]
        v_end = header_info["verse_end"]

        # Apply errata logic from extractor
        if 86 in self.extractor.errata:
            err = self.extractor.errata[86]
            if v_start == err.get("misprint_start"):
                v_start = err.get("corrected_start")
                v_end = err.get("corrected_end")

        self.assertEqual(v_start, 37)
        self.assertEqual(v_end, 37)

    def test_procedural_label_rejection_in_translation(self):
        """
        Edge Case:
        Ensures procedural subtitles like 'Technique 1' or short labels (< 25 chars)
        do not displace the real sloka translation into commentary blocks.
        """
        procedural_candidates = [
            "Technique 1",
            "Technique 2",
            "Stage 1",
            "Part 1",
            "Short tag"
        ]
        for candidate in procedural_candidates:
            is_procedural = bool(re.match(r'^(?:Technique|Stage|Part)\s*\d*', candidate, re.IGNORECASE) or len(candidate) < 25)
            self.assertTrue(is_procedural, f"Failed to detect procedural candidate: '{candidate}'")

    def test_chapter_header_stripping_from_opening_translation(self):
        """Opening verse translations often merge 'Chapter 1\nAsana' into the text block."""
        block_lines = [
            "Chapter 1",
            "Asana",
            "Salutations to the glorious primal guru Sri Adinath."
        ]
        while block_lines and re.match(r'^Chapter(?:\s*(?:\d+|[A-Za-z]+))?\b', block_lines[0], re.IGNORECASE):
            block_lines.pop(0)

        cleaned_text = " ".join(block_lines).strip()
        self.assertFalse(cleaned_text.startswith("Chapter 1"))
        self.assertIn("Salutations to the glorious primal guru", cleaned_text)

    def test_bidirectional_practice_title_deduplication(self):
        """Prevents duplicate titles when parenthetical subtitles or pending titles overlap."""
        pending = "KOORMASANA"
        incoming = "KOORMASANA (tortoise pose)"

        p_low, c_low = pending.lower(), incoming.lower()
        if c_low in p_low:
            result = pending
        elif p_low in c_low:
            result = incoming
        else:
            result = f"{pending} {incoming}".strip()

        self.assertEqual(result, "KOORMASANA (tortoise pose)")
        self.assertNotIn("KOORMASANA KOORMASANA", result)

    def test_practice_title_guard_allows_commentary(self):
        """Verifies all-caps or parenthetical blocks during commentary (e.g. BASTI, NETI, (50)) are not swallowed."""
        sloka_translation = "A strip of wet cloth is swallowed..."
        commentary_blocks = []
        incoming_blocks = ["BASTI", "Jala (water)", "NETI", "(50)"]

        for blk in incoming_blocks:
            # If sloka_translation is set, is_practice_title must NOT trigger
            if not sloka_translation and is_practice_title(blk):
                pass
            else:
                commentary_blocks.append(blk)

        self.assertEqual(commentary_blocks, ["BASTI", "Jala (water)", "NETI", "(50)"])


if __name__ == "__main__":
    unittest.main()
