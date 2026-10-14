#!/usr/bin/env python3
"""
Prompt Architecture, Scriptural Guardrails & Context Assembly Engine (Step 8)
=============================================================================
Provides:
1. `SYSTEM_PROMPT_TEMPLATE`: Authoritative scholar persona, mandatory citation
   contracts (`[Chapter X, Verse Y, pp. Z-W]`), safety & non-hallucination guardrails,
   and guru guidance disclaimers.
2. `YOGA_SYNONYM_LEXICON` & `align_query_terminology`: Bridges English/vernacular
   terms to classical Sanskrit terms at query time with zero corpus annotation.
3. `reorder_chunks_chronologically`: Sorts retrieved chunks in canonical scripture
   order (chapter, verse_start, sub_chunk_index) so multi-part verses form coherent
   unbroken narratives.
4. `format_context_for_prompt`: Assembles delimited, readable scriptural excerpts
   with exact metadata headers for LLM context injection.
5. `build_rag_messages`: Creates the full OpenAI chat messages payload with system
   guardrails, multi-turn history, and grounded user query context.
6. `extract_citations`: Regex parser to validate citation adherence in LLM outputs.
"""

import os
import sys
import re
from typing import Any, Dict, List, Optional, Tuple

from query_db import SearchResult


# ============================================================================
# 1. System Prompt Template with Citation Contract & Safety Guardrails
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are an authoritative, revered, and faithful AI scholar of classical yoga scriptures, specializing in the Hatha Yoga Pradipika by Yogi Svatmarama (with commentary by Swami Muktibodhananda, Bihar School of Yoga).

Your primary mission is to provide accurate, respectful, and scripturally faithful answers based strictly and exclusively on the retrieved scriptural excerpts provided in the context.

### MANDATORY CITATION CONTRACT:
1. Every factual statement, technique instruction, philosophical premise, benefit, or contraindication MUST cite its scriptural source.
2. Citations must follow the exact format: `[Chapter X, Verse Y, pp. Z-W]` or `[Chapter X, Verses Y-Z, pp. Z-W]`.
   - Examples:
     * `[Chapter 3, Verses 61-63, pp. 315-318]`
     * `[Chapter 1, Verse 15, pp. 45-46]`
     * `[Chapter 2, Verses 51-53, pp. 217-219]`
3. If multiple excerpts support a claim, cite all relevant references.
4. Do not invent verse numbers or page ranges not present in the excerpt metadata.

### SCRIPTURAL SAFETY & NON-HALLUCINATION GUARDRAILS:
1. STRICT FACTUAL GROUNDING: Rely strictly on the provided scriptural excerpts. Never extrapolate, embellish, or invent modern variations, benefits, or medical claims not found in the text.
2. INSUFFICIENT CONTEXT: If the retrieved excerpts do not contain the answer, explicitly state:
   "Based on the retrieved excerpts from the Hatha Yoga Pradipika, this specific information is not detailed in the text."
   Then summarize whatever related teachings are available in the context without speculating.
3. ESOTERIC & PHYSIOLOGICAL CLAIMS: Classical scriptures contain traditional assertions (e.g., "conquers death", "destroys all diseases", "cures leprosy", "awakens kundalini"). Treat and explain these as traditional scriptural doctrine and esoteric yogic physiology, not as modern medical or therapeutic claims.
4. GURU GUIDANCE & ADVANCED PRACTICE WARNINGS:
   - For advanced practices (such as Khechari Mudra, Vajroli Mudra, Kevali Kumbhaka, Amaroli, or prolonged kumbhaka), explicitly reiterate the text's injunction that these must only be practiced under the direct guidance of an experienced guru or qualified teacher.
5. DUAL-LANGUAGE TERMINOLOGY ALIGNMENT:
   - Correlate English vernacular terms with their classical Sanskrit terms (e.g., "root lock" -> "Moola Bandha", "stomach cleansing" -> "Dhauti", "spinal twist" -> "Matsyendrasana").
   - Maintain the spiritual dignity and traditional precision of yogic terminology.

### INTENT-ADAPTIVE ANSWER SCOPE & CONVERSATIONAL FLOW:
Answer strictly what the user asked without dumping unrequested sections. Adapt your response scope to the user's specific intent:

1. DEFINITIONAL / CONCEPTUAL QUERIES (e.g., "What is X?", "Define X", "Explain the concept of X"):
   - Focus exclusively on defining and explaining what X is, its translation, its core spiritual/physiological concept, and scriptural context with exact verse citations.
   - DO NOT provide step-by-step technique instructions, benefits lists, or contraindications unless explicitly requested.
   - Conclude with a brief, polite prompt: "Let me know if you would like step-by-step practice instructions, benefits, or precautions from the text."

2. PROCEDURAL / PRACTICE QUERIES (e.g., "How to perform X?", "How to practice X?", "Steps for X"):
   - Provide the step-by-step technique faithfully extracted from the slokas and commentary with citations.
   - Include any vital physical alignment and essential guru warnings or precautions tied directly to practice execution.

3. BENEFITS / EFFECTS QUERIES (e.g., "What are the benefits of X?", "Why practice X?"):
   - Focus specifically on the physical, pranic, and spiritual benefits attributed by the text, citing all relevant verses.

4. COMPREHENSIVE QUERIES (e.g., "Tell me all about X", "Give a complete guide to X"):
   - Provide the full structured breakdown: Definition, Step-by-Step Technique, Benefits, and Guru Guidance/Precautions.

5. CONVERSATIONAL FOLLOW-UPS:
   - When the user asks a follow-up ("What are its benefits?", "Are there cautions?"), seamlessly address that specific facet with citations."""


# ============================================================================
# 2. Dual-Language Query Terminology Alignment Lexicon (Zero Corpus Annotation)
# ============================================================================

YOGA_SYNONYM_LEXICON: Dict[str, str] = {
    # Shatkarma / Cleansing practices
    "stomach wash": "dhauti",
    "stomach cleansing": "dhauti",
    "cleansing the stomach": "dhauti",
    "cloth cleansing": "vastra dhauti",
    "cloth wash": "vastra dhauti",
    "fire wash": "agnisar kriya",
    "fire cleansing": "agnisar kriya",
    "fire breath": "agnisar",
    "abdominal churning": "nauli",
    "churning of the abdomen": "nauli",
    "rectal cleansing": "basti",
    "colon cleansing": "basti",
    "nasal cleansing": "neti",
    "nasal wash": "jala neti",
    "nasal catheter": "sutra neti",
    "gazing": "trataka",
    "steady gazing": "trataka",
    "candle gazing": "trataka",
    "skull shining": "kapalabhati",
    "frontal brain cleansing": "kapalabhati",

    # Pranayama / Breath retention
    "bellows breath": "bhastrika",
    "ocean breath": "ujjayi",
    "victorious breath": "ujjayi",
    "psychic breath": "ujjayi",
    "cooling breath": "sheetali",
    "hissing breath": "sheetkari",
    "humming breath": "bhramari",
    "bee breath": "bhramari",
    "retention": "kumbhaka",
    "breath retention": "kumbhaka",
    "spontaneous retention": "kevali kumbhaka",
    "isolated retention": "kevali kumbhaka",
    "alternate nostril": "nadi shodhana",
    "channel purification": "nadi shodhana",
    "channel cleansing": "nadi shodhana",
    "cleansing of nadis": "nadi shodhana",

    # Bandhas & Mudras (Locks and seals)
    "root lock": "moola bandha",
    "perineum lock": "moola bandha",
    "perineum contraction": "moola bandha",
    "perineal lock": "moola bandha",
    "cervix lock": "moola bandha",
    "cervical lock": "moola bandha",
    "throat lock": "jalandhara bandha",
    "chin lock": "jalandhara bandha",
    "abdominal lock": "uddiyana bandha",
    "stomach lock": "uddiyana bandha",
    "great lock": "maha bandha",
    "great seal": "maha mudra",
    "great piercing": "maha vedha",
    "tongue lock": "khechari mudra",
    "space moving": "khechari mudra",
    "inverted pose": "viparita karani",
    "inverted practice": "viparita karani",
    "inverted psychic attitude": "viparita karani",
    "reverse seal": "viparita karani",
    "thunderbolt lock": "vajroli mudra",
    "serpent power": "kundalini",
    "coiled power": "kundalini",

    # Asanas / Postures
    "spinal twist": "matsyendrasana",
    "half spinal twist": "ardha matsyendrasana",
    "corpse pose": "shavasana",
    "peacock pose": "mayurasana",
    "bow pose": "dhanurasana",
    "cow face pose": "gomukhasana",
    "lotus pose": "padmasana",
    "accomplished pose": "siddhasana",
    "adept pose": "siddhasana",
    "auspicious pose": "swastikasana",
    "lion pose": "simhasana",
    "hero pose": "virasana",
    "back stretching pose": "paschimottanasana",
}


def align_query_terminology(query: str) -> List[str]:
    """
    Identifies vernacular or English yogic terms in the query and returns
    corresponding classical Sanskrit scriptural keywords.

    Runs strictly at query time; requires zero modifications or annotations
    to the underlying vector database or chunks.
    """
    if not query:
        return []

    lowered = query.lower()
    discovered_sanskrit: List[str] = []

    # Sort lexicon keys by descending length so multi-word matches take precedence
    sorted_phrases = sorted(YOGA_SYNONYM_LEXICON.keys(), key=len, reverse=True)

    for phrase in sorted_phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, lowered):
            sanskrit_term = YOGA_SYNONYM_LEXICON[phrase]
            if sanskrit_term not in discovered_sanskrit:
                discovered_sanskrit.append(sanskrit_term)

    return discovered_sanskrit


# ============================================================================
# 3. Context Re-ordering & Deduplication
# ============================================================================

def _safe_int(val: Any, default: int = 0) -> int:
    """Safely converts metadata values to integers with fallback default when None or missing."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def reorder_chunks_chronologically(chunks: List[SearchResult]) -> List[SearchResult]:
    """
    Sorts retrieved chunks in canonical scripture order:
    `(chapter_number, verse_start, verse_end, sub_chunk_index)`.

    Safely handles both verse-based chunks (Strategy 1) and narrative chapter
    chunks (Strategy 2) where verse fields may be None.
    """
    if not chunks:
        return []

    # Deduplicate by chunk_id while preserving the highest-scoring instance
    seen_ids = set()
    unique_chunks: List[SearchResult] = []
    for c in chunks:
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            unique_chunks.append(c)

    def _sort_key(chunk: SearchResult) -> Tuple[int, int, int, int]:
        meta = chunk.metadata
        ch_num = _safe_int(meta.get("chapter_number"), 0)
        v_start = _safe_int(meta.get("verse_start"), 0)
        v_end = _safe_int(meta.get("verse_end"), v_start)
        sub_idx = _safe_int(meta.get("sub_chunk_index"), 0)
        return (ch_num, v_start, v_end, sub_idx)

    return sorted(unique_chunks, key=_sort_key)


# ============================================================================
# 4. Delimited Context Formatting for Prompt Injection
# ============================================================================

def format_context_for_prompt(chunks: List[SearchResult]) -> str:
    """
    Formats retrieved and chronologically sorted chunks into clean,
    delimited context blocks with explicit scriptural citation headers.
    """
    if not chunks:
        return "[No scriptural excerpts available for this query.]"

    sections = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.metadata
        ch_num = meta.get("chapter_number") or "?"
        ch_title = meta.get("chapter_title") or "Unknown"
        v_label = meta.get("verse_label")
        p_start = meta.get("page_start") or "?"
        p_end = meta.get("page_end") or "?"
        is_sub = meta.get("is_sub_chunk", False)
        sub_idx = _safe_int(meta.get("sub_chunk_index"), 0) + 1
        total_subs = _safe_int(meta.get("total_sub_chunks"), 1)

        sub_tag = f" (Part {sub_idx}/{total_subs})" if is_sub else ""
        verse_str = f", Verse {v_label}" if v_label else ""

        header = f"--- EXCERPT {idx}: Chapter {ch_num} ({ch_title}){verse_str}{sub_tag} | Pages {p_start}-{p_end} ---"
        chunk_source = f"[Chunk ID: {c.chunk_id}]"
        body = c.content.strip()

        sections.append(f"{header}\n{chunk_source}\n{body}")

    return "\n\n".join(sections)


# ============================================================================
# 5. Full RAG Messages Assembly for OpenAI Chat API
# ============================================================================

def build_rag_messages(
    query: str,
    retrieved_chunks: List[SearchResult],
    chat_history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Assembles the complete OpenAI Chat Completion messages array:
    1. System message containing persona, citation contract, and safety guardrails.
    2. Prior multi-turn chat history (optional).
    3. User message containing current question and chronologically assembled excerpts.
    """
    # 1. Chronologically re-order chunks
    ordered_chunks = reorder_chunks_chronologically(retrieved_chunks)

    # 2. Format context string
    context_str = format_context_for_prompt(ordered_chunks)

    # 3. Check for English-Sanskrit terminology alignment
    aligned_terms = align_query_terminology(query)
    alignment_note = ""
    if aligned_terms:
        alignment_note = f"\n[Scriptural Terminology Note: The query mentions concepts corresponding to Sanskrit scripture terms: {', '.join(aligned_terms)}]\n"

    # 4. Construct messages payload
    messages: List[Dict[str, str]] = []

    sys_content = system_prompt or SYSTEM_PROMPT_TEMPLATE
    messages.append({"role": "system", "content": sys_content})

    # Append multi-turn history if provided
    if chat_history:
        for turn in chat_history:
            messages.append(turn)

    # User message with grounded context
    user_content = (
        f"USER QUESTION:\n"
        f"{query}\n"
        f"{alignment_note}\n"
        f"RETRIEVED SCRIPTURAL CONTEXT (Hatha Yoga Pradipika):\n"
        f"{context_str}\n\n"
        f"INSTRUCTION:\n"
        f"Answer strictly and specifically what the user asked (e.g. if asked 'What is X', define and explain X without unrequested technique or benefit lists). "
        f"Faithfully ground your answer in the retrieved scriptural excerpts above, adhering to the Mandatory Citation Contract using [Chapter X, Verse Y, pp. Z-W] format."
    )
    messages.append({"role": "user", "content": user_content})

    return messages


# ============================================================================
# 6. Citation Extractor & Adherence Validator
# ============================================================================

# Matches: [Chapter 3, Verse 61-63, pp. 315-318] or [Chapter 1, Verse 15, pp. 45]
CITATION_REGEX = re.compile(
    r'\[Chapter\s+(?P<chapter>\d+),\s+Verses?\s+(?P<verse>[\d\-]+),\s+pp?\.\s+(?P<pages>[\d\-]+)\]',
    re.IGNORECASE
)


def extract_citations(text: str) -> List[Dict[str, str]]:
    """
    Extracts structured citation metadata from LLM responses to verify
    adherence to the mandatory citation contract.
    """
    if not text:
        return []

    citations = []
    for match in CITATION_REGEX.finditer(text):
        citations.append({
            "raw": match.group(0),
            "chapter": match.group("chapter"),
            "verse": match.group("verse"),
            "pages": match.group("pages"),
        })
    return citations


# ============================================================================
# 7. CLI Test & Inspection Runner
# ============================================================================

def main():
    import argparse
    from dotenv import load_dotenv
    from openai import OpenAI
    from query_db import HybridRetriever

    load_dotenv()

    parser = argparse.ArgumentParser(description="Test Prompt Architecture & Scriptural Context Assembly")
    parser.add_argument("--query", "-q", default="How to perform the root lock?", help="User question to test")
    parser.add_argument("--config", default="configs/hatha_yoga_pradipika.yaml", help="Path to YAML config")
    parser.add_argument("--top-k", "-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--generate", "-g", action="store_true", help="Send assembled prompt to OpenAI gpt-4o-mini and stream response")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model (default: gpt-4o-mini)")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      🕉️  STEP 8: PROMPT ARCHITECTURE & CONTEXT ASSEMBLY TEST")
    print("=" * 65)
    print(f"  Query:       \"{args.query}\"")
    print(f"  Retrieval:   Top {args.top_k} Chunks via HybridRetriever")

    # 1. Dual-Language Terminology Alignment
    aligned = align_query_terminology(args.query)
    if aligned:
        print(f"  Aligned:     Sanskrit Keywords: {aligned}")
    else:
        print("  Aligned:     No English-to-Sanskrit mapping needed (direct term)")
    print("=" * 65 + "\n")

    # 2. Retrieve chunks
    retriever = HybridRetriever(config_or_path=args.config)
    retrieved = retriever.search(query=args.query, top_k=args.top_k)

    print(f"[*] Retrieved {len(retrieved)} chunks from vector DB + BM25.")
    print("    Raw Retrieval Rank Order:")
    for idx, c in enumerate(retrieved, start=1):
        print(f"      {idx}. [{c.chunk_id}] (Score: {c.score:.4f} | Mode: {c.retrieval_mode})")

    # 3. Chronological Re-ordering
    ordered = reorder_chunks_chronologically(retrieved)
    print("\n[*] Canonical Scripture Re-ordering (Chapter -> Verse -> Sub-chunk):")
    for idx, c in enumerate(ordered, start=1):
        meta = c.metadata
        print(f"      {idx}. [{c.chunk_id}] Ch {meta.get('chapter_number')} V {meta.get('verse_label')} Part {meta.get('sub_chunk_index', 0)+1}/{meta.get('total_sub_chunks', 1)}")

    # 4. Assemble Messages Payload
    messages = build_rag_messages(query=args.query, retrieved_chunks=retrieved)

    print("\n" + "=" * 65)
    print("             ASSEMBLED OPENAI PROMPT PAYLOAD")
    print("=" * 65)
    print(f"\n--- [SYSTEM MESSAGE ({len(messages[0]['content'])} chars)] ---")
    print(messages[0]["content"][:320] + "\n... [citation contracts & guardrails] ...\n")

    print(f"--- [USER MESSAGE CONTEXT PREVIEW ({len(messages[1]['content'])} chars)] ---")
    user_preview_lines = messages[1]["content"].split("\n")[:18]
    print("\n".join(user_preview_lines))
    print("... [full scripture context injected] ...\n")
    print("=" * 65)

    # 5. Optional Live LLM Generation Test
    if args.generate:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n[!] OPENAI_API_KEY not found in environment or .env. Cannot run live generation.")
            return

        print("\n" + "=" * 65)
        print(f"    🤖 LIVE GENERATION TEST ({args.model} | temp=0.2)")
        print("=" * 65 + "\n")

        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.2,
            stream=True
        )

        full_response = []
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            sys.stdout.write(token)
            sys.stdout.flush()
            full_response.append(token)
        print("\n\n" + "=" * 65)

        response_text = "".join(full_response)
        citations = extract_citations(response_text)
        print(f"[*] Verified Citations Detected in Response ({len(citations)} found):")
        for cit in citations:
            print(f"    ✓ {cit['raw']} -> Chapter {cit['chapter']}, Verse {cit['verse']}, pp. {cit['pages']}")
        print("=" * 65 + "\n")
    else:
        print("\n[Tip] Add '--generate' (or '-g') to stream a live answer from gpt-4o-mini and verify citations:")
        print(f"      ./venv/bin/python prompt_engine.py -q \"{args.query}\" --generate\n")


if __name__ == "__main__":
    main()
