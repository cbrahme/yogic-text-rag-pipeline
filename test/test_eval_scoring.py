#!/usr/bin/env python3
"""Unit tests for the evaluation scoring helpers (no API calls, no index)."""

import os
import sys
import unittest

import yaml

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, EVAL_DIR)

from run_eval import (  # noqa: E402
    parse_verse_ref, first_hit_rank, cites_expected, is_decline, score_case, summarize,
    evidence_found, unsupported_from_claims,
)

M = lambda ch, a, b=None: {"chapter_number": ch, "verse_start": a, "verse_end": a if b is None else b}


class TestEvalScoring(unittest.TestCase):

    def test_parse_verse_ref(self):
        self.assertEqual(parse_verse_ref("3:61-63"), (3, 61, 63))
        self.assertEqual(parse_verse_ref("1:15"), (1, 15, 15))

    def test_first_hit_rank_uses_overlap_and_chapter(self):
        metas = [M(2, 61), M(3, 1), M(3, 62)]
        self.assertEqual(first_hit_rank(metas, ["3:61-63"]), 3)
        self.assertIsNone(first_hit_rank([M(2, 61), M(3, 64)], ["3:61-63"]))
        self.assertEqual(first_hit_rank([M(1, 14, 16)], ["1:15"]), 1)

    def test_cites_expected(self):
        cits = [{"chapter": "3", "verse": "61–62"}, {"chapter": "1", "verse": "99"}]
        self.assertTrue(cites_expected(cits, ["3:61-63"]))
        self.assertFalse(cites_expected(cits, ["2:7-9"]))

    def test_is_decline(self):
        self.assertTrue(is_decline("Based on the retrieved excerpts ... this specific information is not detailed in the text."))
        self.assertFalse(is_decline("Moola Bandha is performed by ..."))

    def test_score_answer_case(self):
        case = {"id": "x", "expected_verses": ["3:61-63"], "key_facts": ["a", "b", "c"], "guru_warning": True}
        judgement = {"facts": [{"fact": "a", "present": True}, {"fact": "b", "present": False}, {"fact": "c", "present": True}],
                     "unsupported_claims": ["made up"], "guru_warning_present": False}
        r = score_case(case, "answer [Chapter 3, Verses 61-63]", [M(1, 1), M(3, 61, 63)],
                       [{"chapter": "3", "verse": "61-63", "match": "exact"}], judgement)
        self.assertEqual((r["retrieval_hit"], r["hit_rank"], r["cites_expected"]), (True, 2, True))
        self.assertEqual((r["facts_present"], r["facts_total"], r["missing_facts"]), (2, 3, ["b"]))
        self.assertEqual(r["unsupported_claims"], ["made up"])
        self.assertFalse(r["guru_warning_ok"])
        self.assertTrue(r["decline_correct"])

    def test_score_decline_case(self):
        case = {"id": "off", "expect": "decline"}
        declined = score_case(case, "... is not detailed in the text.", [], [], None)
        answered = score_case(case, "Paris.", [], [], None)
        self.assertTrue(declined["decline_correct"])
        self.assertFalse(answered["decline_correct"])
        self.assertNotIn("retrieval_hit", declined)

    def test_summarize(self):
        a = {"id": "a", "expect": "answer", "declined": False, "decline_correct": True, "retrieval_hit": True,
             "cites_expected": True, "citation_counts": {"exact": 3, "approx": 1, "none": 0},
             "facts_present": 2, "facts_total": 3, "unsupported_claims": [], "guru_warning_ok": True}
        b = {"id": "b", "expect": "answer", "declined": True, "decline_correct": False, "retrieval_hit": False,
             "cites_expected": False, "citation_counts": {"exact": 0, "approx": 0, "none": 0},
             "facts_present": 0, "facts_total": 2, "unsupported_claims": ["x"]}
        c = {"id": "c", "expect": "decline", "declined": True, "decline_correct": True,
             "citation_counts": {"exact": 0, "approx": 0, "none": 0}}
        s = summarize([a, b, c, {"id": "d", "error": "boom"}])
        self.assertEqual(s["retrieval_hit_rate"], 50.0)
        self.assertEqual(s["key_fact_recall"], 40.0)
        self.assertEqual(s["citations_exact_pct"], 75.0)
        self.assertEqual(s["answers_with_unsupported_claims"], 1)
        self.assertEqual((s["guru_warnings"], s["correct_declines"], s["false_declines"], s["errors"]), ("1/1", "1/1", 1, 1))


class TestEvidenceCheck(unittest.TestCase):
    SOURCE = ("Sitting properly in padmasana, keeping neck and abdomen in alignment, exhale prana "
              "through the nose. (60) And again the air should be quickly inhaled up to the heart lotus.")

    def test_verbatim_quote_found_despite_verse_numbers_and_punctuation(self):
        self.assertTrue(evidence_found(
            "exhale prana through the nose. And again the air should be quickly inhaled", self.SOURCE))

    def test_small_copying_slip_tolerated(self):
        self.assertTrue(evidence_found(
            "keeping the neck and abdomen in alignment, exhale prana through the nose", self.SOURCE))

    def test_paraphrase_or_invented_quote_rejected(self):
        self.assertFalse(evidence_found("Bhastrika cures heart disease and insomnia quickly", self.SOURCE))
        self.assertFalse(evidence_found("", self.SOURCE))
        self.assertFalse(evidence_found("the nose", self.SOURCE))  # too short to be evidence

    def test_unsupported_from_claims(self):
        claims = [{"claim": "sit in padmasana", "evidence": "Sitting properly in padmasana, keeping neck and abdomen"},
                  {"claim": "cures fever", "evidence": ""},
                  {"claim": "invented", "evidence": "this sentence is not in the source at all"}]
        self.assertEqual(unsupported_from_claims(claims, self.SOURCE), ["cures fever", "invented"])


class TestQuestionFile(unittest.TestCase):
    """Keeps eval/questions.yaml well-formed (runs in CI without the index)."""

    def test_questions_are_valid(self):
        cases = yaml.safe_load(open(os.path.join(EVAL_DIR, "questions.yaml"), encoding="utf-8"))
        ids = [c["id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)), "duplicate ids")
        for c in cases:
            with self.subTest(c["id"]):
                self.assertTrue(c.get("question"))
                if c.get("expect", "answer") == "decline":
                    continue
                self.assertTrue(c.get("expected_verses"))
                self.assertTrue(c.get("key_facts"))
                for ref in c["expected_verses"]:
                    parse_verse_ref(ref)


if __name__ == "__main__":
    unittest.main()
