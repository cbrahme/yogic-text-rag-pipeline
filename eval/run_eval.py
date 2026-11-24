#!/usr/bin/env python3
"""
YogicScholar evaluation runner
==============================
Runs every case in eval/questions.yaml through the real chat pipeline
(hybrid retrieval + prompt + OpenAI) and scores:

1. Retrieval hit rate   - was an expected verse among the retrieved excerpts?
2. Citation accuracy    - share of citations marked exact / approx / none, and
                          whether the answer cites an expected verse
3. Key-fact recall      - an LLM judge checks each expected key fact is present
4. Unsupported claims   - the judge lists claims not supported by the excerpts
5. Guru warnings        - advanced practices must mention a guru's guidance
6. Declines             - off-topic questions declined, real questions answered

Needs OPENAI_API_KEY and a built index. A full run costs a few cents with
gpt-4o-mini. Not part of CI.

Usage (from anywhere):
    python eval/run_eval.py                     # full run, saves results JSON
    python eval/run_eval.py --only mitahara-what bhastrika-how
    python eval/run_eval.py --no-judge          # skip fact checking (cheaper)
    python eval/run_eval.py --compare eval/results/<previous>.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

DEFAULT_QUESTIONS = EVAL_DIR / "questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"

# Judge token usage for the run (answer generation streams, so its usage is not reported)
JUDGE_USAGE = {"input_tokens": 0, "output_tokens": 0}

# The system prompt tells the model to use this exact phrase when the text lacks the answer
DECLINE_MARKER = "not detailed in the text"


# ============================================================================
# Pure scoring helpers (unit-tested without any API calls)
# ============================================================================

def parse_verse_ref(ref: str) -> Tuple[int, int, int]:
    """'3:61-63' -> (3, 61, 63); '1:15' -> (1, 15, 15)."""
    chapter, verses = ref.split(":")
    parts = verses.split("-")
    start = int(parts[0])
    end = int(parts[-1])
    return int(chapter), min(start, end), max(start, end)


def _meta_span(meta: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    try:
        ch = int(meta.get("chapter_number"))
        start = int(meta.get("verse_start"))
        end = int(meta.get("verse_end") if meta.get("verse_end") is not None else start)
    except (TypeError, ValueError):
        return None
    return ch, start, end


def spans_overlap(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    return a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]


def first_hit_rank(source_metas: List[Dict[str, Any]], expected_refs: List[str]) -> Optional[int]:
    """1-based rank of the first retrieved excerpt overlapping an expected verse, else None."""
    expected = [parse_verse_ref(r) for r in expected_refs]
    for rank, meta in enumerate(source_metas, start=1):
        span = _meta_span(meta)
        if span and any(spans_overlap(span, e) for e in expected):
            return rank
    return None


def cites_expected(citations: List[Dict[str, Any]], expected_refs: List[str]) -> bool:
    """True if any citation in the answer overlaps an expected verse."""
    expected = [parse_verse_ref(r) for r in expected_refs]
    for c in citations:
        try:
            ref = f"{c['chapter']}:{c['verse']}".replace("–", "-").replace("—", "-").replace(" ", "")
            span = parse_verse_ref(ref)
        except (KeyError, ValueError):
            continue
        if any(spans_overlap(span, e) for e in expected):
            return True
    return False


def is_decline(answer: str) -> bool:
    return DECLINE_MARKER in (answer or "").lower()


def score_case(case: Dict[str, Any], answer: str, source_metas: List[Dict[str, Any]],
               citations: List[Dict[str, Any]], judgement: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Combines raw outputs into per-case scores."""
    expect_decline = case.get("expect", "answer") == "decline"
    declined = is_decline(answer)
    result: Dict[str, Any] = {
        "id": case["id"],
        "type": case.get("type", ""),
        "expect": "decline" if expect_decline else "answer",
        "declined": declined,
        "decline_correct": declined == expect_decline,
        "citation_counts": {
            m: sum(1 for c in citations if c.get("match") == m) for m in ("exact", "approx", "none")
        },
    }
    if not expect_decline:
        rank = first_hit_rank(source_metas, case.get("expected_verses", []))
        result["retrieval_hit"] = rank is not None
        result["hit_rank"] = rank
        result["cites_expected"] = cites_expected(citations, case.get("expected_verses", []))
    if judgement is not None and not expect_decline:
        facts = judgement.get("facts", [])
        present = sum(1 for f in facts if f.get("present"))
        result["facts_present"] = present
        result["facts_total"] = len(case.get("key_facts", []))
        result["missing_facts"] = [f.get("fact") for f in facts if not f.get("present")]
        result["unsupported_claims"] = judgement.get("unsupported_claims", [])
        if case.get("guru_warning"):
            result["guru_warning_ok"] = bool(judgement.get("guru_warning_present"))
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics over all scored cases (errors excluded)."""
    ok = [r for r in results if "error" not in r]
    answer_cases = [r for r in ok if r["expect"] == "answer"]
    decline_cases = [r for r in ok if r["expect"] == "decline"]
    judged = [r for r in answer_cases if "facts_total" in r]
    cit = {m: sum(r["citation_counts"][m] for r in ok) for m in ("exact", "approx", "none")}
    total_cit = sum(cit.values())

    def pct(n, d):
        return round(100.0 * n / d, 1) if d else None

    guru = [r for r in judged if "guru_warning_ok" in r]
    return {
        "cases": len(results),
        "errors": len(results) - len(ok),
        "retrieval_hit_rate": pct(sum(r["retrieval_hit"] for r in answer_cases), len(answer_cases)),
        "cites_expected_rate": pct(sum(r["cites_expected"] for r in answer_cases), len(answer_cases)),
        "citations_total": total_cit,
        "citations_exact_pct": pct(cit["exact"], total_cit),
        "citations_approx_pct": pct(cit["approx"], total_cit),
        "citations_none_pct": pct(cit["none"], total_cit),
        "key_fact_recall": pct(sum(r["facts_present"] for r in judged), sum(r["facts_total"] for r in judged)),
        "answers_with_unsupported_claims": sum(1 for r in judged if r["unsupported_claims"]) if judged else None,
        "guru_warnings": f"{sum(r['guru_warning_ok'] for r in guru)}/{len(guru)}" if guru else None,
        "correct_declines": f"{sum(r['decline_correct'] for r in decline_cases)}/{len(decline_cases)}",
        "false_declines": sum(1 for r in answer_cases if r["declined"]),
    }


# ============================================================================
# LLM judge
# ============================================================================

JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented assistant about the Hatha Yoga Pradipika.

QUESTION:
{question}

RETRIEVED EXCERPTS (the only material the assistant was allowed to use):
{excerpts}

ANSWER:
{answer}

KEY FACTS a correct answer should contain:
{facts}

Return JSON with exactly these fields:
- "facts": list of {{"fact": <key fact text>, "present": true/false}} - present means the answer states this fact or a clear paraphrase of it.
- "claims": list of {{"claim": <short paraphrase>, "evidence": <exact quote>}} - one entry per substantive factual claim in the answer (ignore greetings, offers of more help, and generic framing). "evidence" must be a short passage (5-25 words) copied VERBATIM from the RETRIEVED EXCERPTS above (never from the ANSWER) that supports the claim, or "" if no excerpt supports it.
- "guru_warning_present": true/false - whether the answer says the practice should be learned or done under the guidance of a guru or qualified teacher.
"""


def _normalize(text: str) -> str:
    """Lowercase, unify quotes, and drop punctuation and digits (the book has inline verse numbers like '(60)')."""
    text = (text or "").lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"[^a-z']+", " ", text).strip()


def evidence_found(evidence: str, source_text: str, shingle: int = 4, min_share: float = 0.6) -> bool:
    """
    True if a judge-provided quote really occurs in the retrieved text. Tolerates
    small copying slips: at least `min_share` of the quote's 4-word sequences
    must appear in the (normalized) source text.
    """
    words = _normalize(evidence).split()
    if len(words) < shingle:
        return False
    haystack = " " + _normalize(source_text) + " "
    grams = [" ".join(words[i:i + shingle]) for i in range(len(words) - shingle + 1)]
    found = sum(1 for g in grams if f" {g} " in haystack)
    return found / len(grams) >= min_share


def unsupported_from_claims(claims: List[Dict[str, Any]], source_text: str) -> List[str]:
    """Claims whose evidence is missing or cannot be found verbatim in the retrieved text."""
    return [c.get("claim", "") for c in claims if not evidence_found(c.get("evidence", ""), source_text)]


def judge_answer(client, model: str, case: Dict[str, Any], answer: str, sources) -> Dict[str, Any]:
    excerpts = "\n\n".join(
        f"[{i}] {s.content.strip()[:2500]}" for i, s in enumerate(sources, start=1)
    ) or "(none)"
    facts = "\n".join(f"- {f}" for f in case.get("key_facts", [])) or "(none)"
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=case["question"], excerpts=excerpts, answer=answer, facts=facts)}],
    )
    if getattr(resp, "usage", None):
        JUDGE_USAGE["input_tokens"] += resp.usage.prompt_tokens or 0
        JUDGE_USAGE["output_tokens"] += resp.usage.completion_tokens or 0
    judgement = json.loads(resp.choices[0].message.content)
    # Do not trust the judge's own verdict on support: verify its quotes against the real text
    source_text = " ".join(s.content for s in sources)
    judgement["claims"] = judgement.get("claims", [])
    judgement["unsupported_claims"] = unsupported_from_claims(judgement["claims"], source_text)
    return judgement


# ============================================================================
# Runner
# ============================================================================

def run_case(session, judge_client, judge_model: Optional[str], case: Dict[str, Any]) -> Dict[str, Any]:
    session.clear_history()
    for earlier in case.get("history", []):
        session.ask(earlier)
    t0 = time.time()
    answer, sources, citations = session.ask(case["question"])
    elapsed = round(time.time() - t0, 1)

    judgement = None
    if judge_model and case.get("expect", "answer") != "decline":
        judgement = judge_answer(judge_client, judge_model, case, answer, sources)

    result = score_case(case, answer, [s.metadata for s in sources], citations, judgement)
    result.update({
        "question": case["question"],
        "seconds": elapsed,
        "answer": answer,
        "retrieved": [s.chunk_id for s in sources],
        "citations": [{"raw": c["raw"], "match": c["match"]} for c in citations],
    })
    return result


def _mark(value) -> str:
    return "-" if value is None else ("yes" if value else "NO")


def print_report(results: List[Dict[str, Any]], summary: Dict[str, Any],
                 previous: Optional[Dict[str, Any]] = None) -> None:
    print("\n" + "=" * 96)
    print(f"{'case':32} {'hit@rank':>8} {'cites exp':>9} {'facts':>6} {'unsupp':>6} {'cit ✓/≈/⚠':>10} {'note':>12}")
    print("-" * 96)
    for r in results:
        if "error" in r:
            print(f"{r['id']:32} ERROR: {r['error']}")
            continue
        hit = "-" if r["expect"] == "decline" else (f"yes@{r['hit_rank']}" if r["retrieval_hit"] else "NO")
        facts = f"{r['facts_present']}/{r['facts_total']}" if "facts_total" in r else "-"
        unsup = str(len(r["unsupported_claims"])) if "unsupported_claims" in r else "-"
        cc = r["citation_counts"]
        note = ""
        if r["expect"] == "decline":
            note = "declined ✓" if r["decline_correct"] else "ANSWERED ✗"
        elif r["declined"]:
            note = "DECLINED ✗"
        elif "guru_warning_ok" in r:
            note = "guru ✓" if r["guru_warning_ok"] else "NO GURU ✗"
        print(f"{r['id']:32} {hit:>8} {_mark(r.get('cites_expected')):>9} {facts:>6} {unsup:>6} "
              f"{cc['exact']:>4}/{cc['approx']}/{cc['none']:<3} {note:>12}")

    print("=" * 96)
    labels = [
        ("retrieval_hit_rate", "Retrieval hit rate (%)"),
        ("cites_expected_rate", "Answers citing an expected verse (%)"),
        ("key_fact_recall", "Key-fact recall (%)"),
        ("answers_with_unsupported_claims", "Answers with unsupported claims"),
        ("citations_exact_pct", "Citations exact ✓ (%)"),
        ("citations_approx_pct", "Citations approx ≈ (%)"),
        ("citations_none_pct", "Citations no match ⚠ (%)"),
        ("guru_warnings", "Guru warnings given"),
        ("correct_declines", "Off-topic correctly declined"),
        ("false_declines", "Real questions wrongly declined"),
        ("errors", "Errors"),
    ]
    for key, label in labels:
        value = summary.get(key)
        line = f"  {label:40} {value if value is not None else '-'}"
        if previous and isinstance(value, (int, float)) and isinstance(previous.get(key), (int, float)):
            delta = round(value - previous[key], 1)
            line += f"   (prev {previous[key]}, {'+' if delta >= 0 else ''}{delta})"
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YogicScholar answers against eval/questions.yaml")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--config", default="configs/hatha_yoga_pradipika.yaml")
    parser.add_argument("--model", default="gpt-4o-mini", help="Answering model")
    parser.add_argument("--judge-model", default="gpt-4o", help="Model that checks key facts and claims (gpt-4o-mini is cheaper but much less reliable)")
    parser.add_argument("--no-judge", action="store_true", help="Skip key-fact / unsupported-claim checks")
    parser.add_argument("--only", nargs="+", metavar="ID", help="Run only these case ids")
    parser.add_argument("--compare", metavar="RESULTS_JSON", help="Show deltas against a previous run")
    parser.add_argument("--no-save", action="store_true", help="Do not write a results JSON file")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI
    from chat import ChatSession
    from error_messages import OPENAI_TIMEOUT_SECONDS, friendly_error_message

    load_dotenv(PROJECT_ROOT / ".env")

    cases = yaml.safe_load(open(args.questions, encoding="utf-8"))
    if args.only:
        unknown = set(args.only) - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"Unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["id"] in set(args.only)]

    try:
        session = ChatSession(config_path=args.config, model=args.model)
    except Exception as e:
        sys.exit(f"[!] Could not start: {e}")
    judge_model = None if args.no_judge else args.judge_model
    judge_client = OpenAI(timeout=OPENAI_TIMEOUT_SECONDS) if judge_model else None

    print(f"[*] Running {len(cases)} cases with {args.model}"
          f"{'' if args.no_judge else f' (judge: {args.judge_model})'} ...")
    results = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case['id']}", flush=True)
        try:
            results.append(run_case(session, judge_client, judge_model, case))
        except Exception as e:
            results.append({"id": case["id"], "error": friendly_error_message(e) + f" ({type(e).__name__})"})

    summary = summarize(results)
    previous = None
    if args.compare:
        previous = json.load(open(args.compare, encoding="utf-8")).get("summary")
    print_report(results, summary, previous)
    if judge_model:
        print(f"[*] Judge tokens: {JUDGE_USAGE['input_tokens']:,} in / {JUDGE_USAGE['output_tokens']:,} out")

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        payload = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "judge_model": judge_model,
            "judge_usage": dict(JUDGE_USAGE),
            "summary": summary,
            "results": results,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[*] Saved results to {out}")


if __name__ == "__main__":
    main()
