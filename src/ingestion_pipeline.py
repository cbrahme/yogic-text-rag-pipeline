#!/usr/bin/env python3
"""
Multi-Strategy Config-Driven Ingestion & Chunking Pipeline (OpenAI Standardized)
================================================================================
This pipeline supports multi-book ingestion through declarative YAML configs:
1. Standardized Strictly on OpenAI:
   - Tokenization: exact BPE counting using tiktoken (cl100k_base), 100% aligned with text-embedding-3.
   - Vector Embedding target: OpenAI text-embedding-3-small (1536 dimensions).
2. Unicode Normalization:
   - Canonical NFC normalization to ensure Sanskrit IAST diacritics (ā, ī, ū, ṛ, ś, ṣ)
     are precomposed single code points, minimizing sub-byte BPE fragmentation.
3. Namespaced Chunk IDs:
   - All chunk IDs are prefixed with `book_id` (e.g., `hyp_ch01_v001_p01`).
4. Dual Chunking Strategies:
   - Strategy 1 ('verse'): 2-tier verse-centric hierarchical chunker with Sanskrit sloka isolation
     and contextual breadcrumbs for traditional scriptures (Hatha Yoga Pradipika, Yoga Sutras, etc.).
   - Strategy 2 ('standard'): Recursive semantic chunker for narrative/chapter-based yoga books
     (Light on Yoga, Anatomy of Hatha Yoga, etc.) with chapter contextual headers.
5. Multimodal Readiness:
   - Metadata includes `image_paths` placeholder list for upcoming diagram/illustration indexing.
"""

import os
import re
import json
import unicodedata
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import yaml
import fitz  # PyMuPDF
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


VERSE_LINE_REGEX = re.compile(
    r'^(?:Chapter\s*(?:\d+|[A-Za-z]+)\s+)?'
    r'Verses?\s+(\d+)'
    r'(?:\s*[-–—,]\s*(\d+))?'
    r'(?:\s*\(([^\)]+)\))?'
    r'(?:\s+Chapter\s*(?:\d+|[A-Za-z]+))?'
    r'(?:\s+([A-Za-z\s\(\)\'\-]+))?'
    r'\s*$',
    re.IGNORECASE
)

NARRATIVE_VERB_GUARD = re.compile(
    r'\b(says|states|describes|mentions|is|are|tells|explains|means|refers)\b',
    re.IGNORECASE
)


def extract_verse_header(block_text: str, y0: float, footer_bottom_y: float = 750.0) -> Optional[Dict[str, Any]]:
    """
    Identifies a genuine verse header anywhere on the page (top or mid-page):
    - Rejects running footers
    - Enforces structural header start
    - Prevents narrative prose matches via length and verb checks
    - Captures same-line and next-line posture/section titles
    """
    if y0 > footer_bottom_y:
        return None

    text = block_text.strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return None

    verse_line_idx = -1
    v_start, v_end, extra, same_line_title = None, None, "", ""

    # Check the first 1-3 lines of the block for the verse declaration
    for idx, line in enumerate(lines[:3]):
        if len(line) > 65 or NARRATIVE_VERB_GUARD.search(line):
            continue

        m = VERSE_LINE_REGEX.match(line)
        if m:
            verse_line_idx = idx
            v_start = int(m.group(1))
            v_end = int(m.group(2)) if m.group(2) else v_start
            extra = m.group(3) or ""
            same_line_title = m.group(4).strip() if m.group(4) else ""
            break

    if verse_line_idx == -1:
        return None

    # Check for a posture title on subsequent lines (e.g. KOORMASANA (tortoise pose))
    subsequent_lines = []
    for line in lines[verse_line_idx + 1:]:
        if re.match(r'^(?:Chapter(?:\s*(?:\d+|[A-Za-z]+))?|\d+|One|Two|Three|Four)$', line.strip(), re.IGNORECASE):
            continue
        if len(line) < 120 and not line.endswith('.'):
            subsequent_lines.append(line)

    next_line_title = " ".join(subsequent_lines).strip()

    if same_line_title and next_line_title:
        posture_title = f"{same_line_title} {next_line_title}".strip()
    else:
        posture_title = same_line_title or next_line_title

    v_label = f"{v_start}-{v_end}" if v_start != v_end else f"{v_start}"
    if extra:
        v_label += f" ({extra})"

    return {
        "verse_start": v_start,
        "verse_end": v_end,
        "verse_label": v_label,
        "posture_title": posture_title
    }


def is_practice_title(text: str) -> bool:
    clean = text.strip()
    return bool(
        re.match(r'^(?:[A-Z\s\'-]+(?:\s*\([^\)]+\))?|\([^\)]+\))$', clean)
        and len(clean) < 100
        and not clean.endswith('.')
    )


class TokenCounter:
    """Exact OpenAI BPE token counter using tiktoken (cl100k_base)."""
    _encoder = None

    @classmethod
    def get_encoder(cls):
        if cls._encoder is None:
            cls._encoder = tiktoken.get_encoding("cl100k_base")
        return cls._encoder

    @classmethod
    def count_tokens(cls, text: str) -> int:
        return len(cls.get_encoder().encode(text))


@dataclass
class Verse:
    chapter_num: int
    chapter_title: str
    verse_start: int
    verse_end: int
    verse_label: str
    page_start: int
    page_end: int
    sloka_translation: str
    commentary_blocks: List[str]

    @property
    def verse_id(self) -> str:
        if self.verse_start == self.verse_end:
            return f"ch{self.chapter_num:02d}_v{self.verse_start:03d}"
        return f"ch{self.chapter_num:02d}_v{self.verse_start:03d}-{self.verse_end:03d}"

    @property
    def full_commentary(self) -> str:
        return "\n\n".join(self.commentary_blocks).strip()


@dataclass
class ChapterText:
    chapter_num: int
    chapter_title: str
    page_start: int
    page_end: int
    text_content: str


@dataclass
class Chunk:
    chunk_id: str
    book_id: str
    book_title: str
    parent_verse_id: Optional[str]
    chapter_number: int
    chapter_title: str
    verse_start: Optional[int]
    verse_end: Optional[int]
    verse_label: Optional[str]
    is_sub_chunk: bool
    sub_chunk_index: int
    total_sub_chunks: int
    has_sanskrit_sloka: bool
    page_start: int
    page_end: int
    chunk_type: str
    char_count: int
    token_count: int
    content: str
    image_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextCleaner:
    @staticmethod
    def clean_block_text(text: str) -> str:
        # Canonical NFC Unicode normalization: merges decomposed diacritics
        # (e.g. 'a' + combining macron -> 'ā') to avoid tokenizer fragmentation
        text = unicodedata.normalize('NFC', text)
        # Re-join words broken across line breaks by hyphens
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Replace intra-paragraph newlines with space
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # Normalize excessive whitespace
        return re.sub(r'[ \t]+', ' ', text).strip()


class VersePdfExtractor:
    """Extracts verse units from the PDF using PyMuPDF and layout-aware filtering."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.book_id = config.get("book_id", "book")
        self.chapters = {int(k): v for k, v in config.get("chapters", {}).items()}
        self.margins = config.get("margins", {"header_top_y": 55, "footer_bottom_y": 750})
        self.errata = {int(k): v for k, v in config.get("publisher_errata", {}).items()}

    def extract(self, pdf_path: str) -> List[Verse]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        doc = fitz.open(pdf_path)
        verses: List[Verse] = []
        current_chapter = min(self.chapters.keys()) if self.chapters else 1
        current_verse: Optional[Dict[str, Any]] = None

        cover_page_map = {v["cover_page"]: k for k, v in self.chapters.items()}

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_h = page.rect.height

            # Dynamic resolution/trim-independent margin calculation
            top_val = self.margins.get("header_top_ratio", self.margins.get("header_top_y", 0.07))
            header_top_y = page_h * top_val if top_val <= 1.0 else top_val

            bot_val = self.margins.get("footer_bottom_ratio", self.margins.get("footer_bottom_y", 0.945))
            footer_bottom_y = page_h * bot_val if bot_val <= 1.0 else bot_val

            # Detect chapter cover pages
            if page_num in cover_page_map:
                current_chapter = cover_page_map[page_num]
                continue

            blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))

            for b in blocks:
                # 1. Header detection (works at top of page or mid-page)
                header_info = extract_verse_header(b[4], b[1], footer_bottom_y)
                if header_info:
                    v_start = header_info["verse_start"]
                    v_end = header_info["verse_end"]
                    v_label = header_info["verse_label"]

                    # Apply configured publisher errata
                    if page_num in self.errata:
                        err = self.errata[page_num]
                        if current_chapter == err.get("chapter") and v_start == err.get("misprint_start"):
                            v_start = err.get("corrected_start", v_start)
                            v_end = err.get("corrected_end", v_end)
                            v_label = f"{v_start}-{v_end}" if v_start != v_end else f"{v_start}"

                    is_same_verse = (
                        current_verse is not None
                        and current_verse["chapter_num"] == current_chapter
                        and current_verse["verse_start"] == v_start
                        and current_verse["verse_end"] == v_end
                    )

                    if is_same_verse:
                        current_verse["page_end"] = page_num
                    else:
                        if current_verse:
                            if current_verse.get("_pending_title") and not current_verse["sloka_translation"]:
                                current_verse["sloka_translation"] = current_verse["_pending_title"]
                            clean_v = {k: v for k, v in current_verse.items() if not k.startswith("_")}
                            verses.append(Verse(**clean_v))

                        ch_title = self.chapters.get(current_chapter, {}).get("title", f"Chapter {current_chapter}")
                        current_verse = {
                            "chapter_num": current_chapter,
                            "chapter_title": ch_title,
                            "verse_start": v_start,
                            "verse_end": v_end,
                            "verse_label": v_label,
                            "page_start": page_num,
                            "page_end": page_num,
                            "sloka_translation": "",
                            "commentary_blocks": [],
                            "_pending_title": header_info.get("posture_title", "")
                        }
                    continue

                # 2. Bottom footer detection: y0 > footer_bottom_y and pure digits (page number)
                if b[1] > footer_bottom_y and re.match(r'^\s*\d+\s*$', b[4]):
                    continue

                # 3. Body text block
                cleaned = TextCleaner.clean_block_text(b[4])
                if not cleaned or not current_verse:
                    continue

                # Strip standalone page running headers/footers that slipped through coordinates
                if re.match(r'^(?:Chapter(?:\s*(?:\d+|[A-Za-z]+))?|Hatha\s+Yoga\s+Pradipika)$', cleaned, re.IGNORECASE):
                    continue

                current_verse["page_end"] = page_num

                # Accumulate cross-block practice titles / parenthetical subtitles with bidirectional containment
                if not current_verse["sloka_translation"] and is_practice_title(cleaned):
                    pending = current_verse.get("_pending_title", "").strip()
                    if pending:
                        p_low, c_low = pending.lower(), cleaned.lower()
                        if c_low in p_low:
                            pass
                        elif p_low in c_low:
                            current_verse["_pending_title"] = cleaned
                        else:
                            current_verse["_pending_title"] = f"{pending} {cleaned}".strip()
                    else:
                        current_verse["_pending_title"] = cleaned
                    continue

                if not current_verse["sloka_translation"]:
                    # Strip leading "Chapter X" if it was grouped into the opening verse translation block
                    lines = [ln.strip() for ln in cleaned.split('\n') if ln.strip()]
                    while lines and re.match(r'^Chapter(?:\s*(?:\d+|[A-Za-z]+))?\b', lines[0], re.IGNORECASE):
                        lines.pop(0)
                    candidate = " ".join(lines).strip()

                    # If it's a procedural label like "Technique 1", push to commentary and keep searching
                    if re.match(r'^(?:Technique|Stage|Part)\s*\d*', candidate, re.IGNORECASE) or len(candidate) < 25:
                        current_verse["commentary_blocks"].append(cleaned)
                    else:
                        # Deduplicate before prepending _pending_title
                        title = current_verse.get("_pending_title", "").strip()
                        if title:
                            if not candidate.lower().startswith(title.lower()):
                                current_verse["sloka_translation"] = f"{title}\n\n{candidate}"
                            else:
                                current_verse["sloka_translation"] = candidate
                            current_verse["_pending_title"] = ""
                        else:
                            current_verse["sloka_translation"] = candidate
                else:
                    current_verse["commentary_blocks"].append(cleaned)

        if current_verse:
            if current_verse.get("_pending_title") and not current_verse["sloka_translation"]:
                current_verse["sloka_translation"] = current_verse["_pending_title"]
            clean_v = {k: v for k, v in current_verse.items() if not k.startswith("_")}
            verses.append(Verse(**clean_v))

        return verses


class StandardPdfExtractor:
    """Extracts chapter-level narrative text blocks for non-verse books."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chapters = {int(k): v for k, v in config.get("chapters", {}).items()}
        self.margins = config.get("margins", {"header_top_y": 50, "footer_bottom_y": 740})

    def extract(self, pdf_path: str) -> List[ChapterText]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        doc = fitz.open(pdf_path)
        cover_page_map = {v["cover_page"]: k for k, v in self.chapters.items()}
        sorted_chapter_nums = sorted(self.chapters.keys())

        chapter_texts: List[ChapterText] = []
        current_chapter_idx = None
        current_blocks: List[str] = []
        chapter_start_page = 1

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_h = page.rect.height

            # Dynamic resolution/trim-independent margin calculation
            top_val = self.margins.get("header_top_ratio", self.margins.get("header_top_y", 0.08))
            header_top_y = page_h * top_val if top_val <= 1.0 else top_val

            bot_val = self.margins.get("footer_bottom_ratio", self.margins.get("footer_bottom_y", 0.93))
            footer_bottom_y = page_h * bot_val if bot_val <= 1.0 else bot_val

            if page_num in cover_page_map:
                if current_chapter_idx is not None and current_blocks:
                    ch_title = self.chapters[current_chapter_idx]["title"]
                    chapter_texts.append(
                        ChapterText(
                            chapter_num=current_chapter_idx,
                            chapter_title=ch_title,
                            page_start=chapter_start_page,
                            page_end=page_num - 1,
                            text_content="\n\n".join(current_blocks)
                        )
                    )
                    current_blocks = []
                current_chapter_idx = cover_page_map[page_num]
                chapter_start_page = page_num
                continue

            if current_chapter_idx is None and sorted_chapter_nums:
                current_chapter_idx = sorted_chapter_nums[0]
                chapter_start_page = page_num

            blocks = page.get_text("blocks")
            for b in blocks:
                if b[1] < header_top_y and (b[3] - b[1]) < 40:
                    continue
                if b[1] > footer_bottom_y and re.match(r'^\s*\d+\s*$', b[4]):
                    continue
                cleaned = TextCleaner.clean_block_text(b[4])
                if cleaned:
                    current_blocks.append(cleaned)

        if current_chapter_idx is not None and current_blocks:
            ch_title = self.chapters[current_chapter_idx]["title"]
            chapter_texts.append(
                ChapterText(
                    chapter_num=current_chapter_idx,
                    chapter_title=ch_title,
                    page_start=chapter_start_page,
                    page_end=len(doc),
                    text_content="\n\n".join(current_blocks)
                )
            )

        return chapter_texts


class VerseChunker:
    """Strategy 1: 2-tier verse chunker with sloka isolation and contextual headers."""

    def __init__(self, book_id: str, book_title: str, max_chunk_size: int = 2000, max_tokens: int = 600, chunk_overlap: int = 0):
        self.book_id = book_id
        self.book_title = book_title
        self.max_chunk_size = max_chunk_size
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap

    def build_verse_anchor(self, sloka_text: str, max_chars: int = 350) -> str:
        """
        Generates a concise, grammatically complete verse anchor for sub-chunks 2..N.
        Prioritizes capturing the primary technical entity/practice name and grabs complete
        grammatical sentences (splitting on sentence terminators) rather than cutting off mid-phrase.
        """
        if not sloka_text:
            return ""
        clean = re.sub(r'\s+', ' ', sloka_text).strip()
        if len(clean) <= max_chars:
            return clean

        sentences = re.split(r'(?<=[.?!])\s+', clean)
        anchor = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            candidate = f"{anchor} {s}".strip() if anchor else s
            if not anchor:
                anchor = candidate
                if len(anchor) >= 120:
                    break
            elif len(candidate) <= max_chars:
                anchor = candidate
            else:
                break
        return anchor.strip()

    def _split_to_token_limit(self, text: str, token_limit: int, separators: List[str]) -> List[str]:
        """
        Recursively splits text until all partitions strictly satisfy the token ceiling,
        dynamically estimating character budget based on measured token density.
        """
        tokens = TokenCounter.count_tokens(text)
        if tokens <= token_limit or len(text) <= 50:
            return [text]

        ratio = len(text) / tokens
        target_size = max(50, int(token_limit * ratio * 0.85))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_size,
            chunk_overlap=0,
            separators=separators
        )
        splits = splitter.split_text(text)
        if len(splits) == 1:
            half = len(text) // 2
            splits = [text[:half], text[half:]]

        results: List[str] = []
        for s in splits:
            if TokenCounter.count_tokens(s) > token_limit and len(s) > 50:
                results.extend(self._split_to_token_limit(s, token_limit, separators))
            else:
                results.append(s)
        return results

    def _split_commentary_cleanly(self, commentary: str, first_size: int, reg_size: int, anchor_tokens: int = 0) -> List[str]:
        if not commentary:
            return []
        if len(commentary) <= first_size and TokenCounter.count_tokens(commentary) <= (self.max_tokens - 40):
            return [commentary]

        separators = ["\n\n", ". ", "? ", "! ", "\n", " ", ""]
        splitter_first = RecursiveCharacterTextSplitter(
            chunk_size=first_size,
            chunk_overlap=0,
            separators=separators
        )
        first_parts = splitter_first.split_text(commentary)
        part_0 = first_parts[0]

        remainder = commentary[len(part_0):].strip()
        remaining_parts: List[str] = []
        if remainder:
            splitter_reg = RecursiveCharacterTextSplitter(
                chunk_size=reg_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators
            )
            remaining_parts = splitter_reg.split_text(remainder)

        # Enforce token ceiling guard on all parts
        final_parts: List[str] = []
        for idx, p in enumerate([part_0] + remaining_parts):
            token_limit = (self.max_tokens - 35) if idx == 0 else (self.max_tokens - 35 - anchor_tokens)
            if TokenCounter.count_tokens(p) > token_limit:
                final_parts.extend(self._split_to_token_limit(p, token_limit, separators))
            else:
                final_parts.append(p)

        return final_parts

    def chunk_verse(self, verse: Verse) -> List[Chunk]:
        ch_num = verse.chapter_num
        ch_title = verse.chapter_title
        v_label = verse.verse_label
        raw_v_id = verse.verse_id
        prefixed_v_id = f"{self.book_id}_{raw_v_id}"
        sloka = verse.sloka_translation.strip()
        commentary = verse.full_commentary

        single_header = f"[{self.book_title} | Chapter {ch_num}: {ch_title} | Verse {v_label}]"
        full_body = f"{sloka}\n\n{commentary}".strip() if sloka else commentary
        single_content = f"{single_header}\n\n{full_body}".strip()
        single_tokens = TokenCounter.count_tokens(single_content)

        # Primary Chunking: fits intact in one chunk only if BOTH char and token ceilings pass
        if len(single_content) <= self.max_chunk_size and single_tokens <= self.max_tokens:
            return [
                Chunk(
                    chunk_id=prefixed_v_id,
                    book_id=self.book_id,
                    book_title=self.book_title,
                    parent_verse_id=prefixed_v_id,
                    chapter_number=ch_num,
                    chapter_title=ch_title,
                    verse_start=verse.verse_start,
                    verse_end=verse.verse_end,
                    verse_label=v_label,
                    is_sub_chunk=False,
                    sub_chunk_index=0,
                    total_sub_chunks=1,
                    has_sanskrit_sloka=True,
                    page_start=verse.page_start,
                    page_end=verse.page_end,
                    chunk_type="verse",
                    char_count=len(single_content),
                    token_count=single_tokens,
                    content=single_content,
                    image_paths=[]
                )
            ]

        # Secondary Recursive Chunking: exceeds size limit
        # Ensure character budget respects the token ceiling
        # (Using a conservative 2.9 chars/token for dense Sanskrit transliteration)
        verse_anchor = self.build_verse_anchor(sloka)
        anchor_str = f"[Verse {v_label} Context: {verse_anchor}]" if verse_anchor else ""
        anchor_tokens = TokenCounter.count_tokens(anchor_str) if anchor_str else 0

        effective_max = min(self.max_chunk_size, int(self.max_tokens * 2.9))
        header_budget = 90
        first_comm_budget = max(200, effective_max - header_budget - len(sloka) - 4)
        reg_comm_budget = max(200, effective_max - header_budget - len(anchor_str) - 4)

        comm_parts = self._split_commentary_cleanly(commentary, first_comm_budget, reg_comm_budget, anchor_tokens)
        total_parts = len(comm_parts) if comm_parts else 1

        chunks: List[Chunk] = []
        for i, c_text in enumerate(comm_parts):
            part_num = i + 1
            header = f"[{self.book_title} | Chapter {ch_num}: {ch_title} | Verse {v_label} (Part {part_num}/{total_parts})]"

            if i == 0:
                content = f"{header}\n\n{sloka}\n\n{c_text}".strip() if sloka else f"{header}\n\n{c_text}".strip()
                has_sloka = True
            else:
                if anchor_str:
                    content = f"{header}\n\n{anchor_str}\n\n{c_text}".strip()
                else:
                    content = f"{header}\n\n{c_text}".strip()
                has_sloka = False

            exact_tokens = TokenCounter.count_tokens(content)
            chunks.append(
                Chunk(
                    chunk_id=f"{prefixed_v_id}_p{part_num:02d}",
                    book_id=self.book_id,
                    book_title=self.book_title,
                    parent_verse_id=prefixed_v_id,
                    chapter_number=ch_num,
                    chapter_title=ch_title,
                    verse_start=verse.verse_start,
                    verse_end=verse.verse_end,
                    verse_label=v_label,
                    is_sub_chunk=True,
                    sub_chunk_index=i,
                    total_sub_chunks=total_parts,
                    has_sanskrit_sloka=has_sloka,
                    page_start=verse.page_start,
                    page_end=verse.page_end,
                    chunk_type="verse_subchunk",
                    char_count=len(content),
                    token_count=exact_tokens,
                    content=content,
                    image_paths=[]
                )
            )

        return chunks

    def chunk_all(self, verses: List[Verse]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for verse in verses:
            all_chunks.extend(self.chunk_verse(verse))
        return all_chunks


class StandardChunker:
    """Strategy 2: Recursive character chunker for narrative/chapter-based yoga books."""

    def __init__(self, book_id: str, book_title: str, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.book_id = book_id
        self.book_title = book_title
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_all(self, chapters: List[ChapterText]) -> List[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "]
        )

        all_chunks: List[Chunk] = []
        for ch in chapters:
            raw_splits = splitter.split_text(ch.text_content)
            total_parts = len(raw_splits)

            for idx, split_text in enumerate(raw_splits):
                part_num = idx + 1
                header = f"[{self.book_title} | Chapter {ch.chapter_num}: {ch.chapter_title} (Part {part_num}/{total_parts})]"
                content = f"{header}\n\n{split_text}".strip()
                exact_tokens = TokenCounter.count_tokens(content)

                chunk_id = f"{self.book_id}_ch{ch.chapter_num:02d}_c{part_num:03d}"
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        book_id=self.book_id,
                        book_title=self.book_title,
                        parent_verse_id=None,
                        chapter_number=ch.chapter_num,
                        chapter_title=ch.chapter_title,
                        verse_start=None,
                        verse_end=None,
                        verse_label=None,
                        is_sub_chunk=(total_parts > 1),
                        sub_chunk_index=idx,
                        total_sub_chunks=total_parts,
                        has_sanskrit_sloka=False,
                        page_start=ch.page_start,
                        page_end=ch.page_end,
                        chunk_type="standard_chapter_chunk",
                        char_count=len(content),
                        token_count=exact_tokens,
                        content=content,
                        image_paths=[]
                    )
                )

        return all_chunks


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "configs/hatha_yoga_pradipika.yaml") -> List[Chunk]:
    config = load_config(config_path)
    book_id = config.get("book_id", "hyp")
    book_title = config.get("book_title", "Yoga Scripture")
    pdf_path = config.get("pdf_path", "docs/Hatha-Yoga-Pradipika-pages-2.pdf")
    output_path = config.get("output_path", "chunks.jsonl")
    strategy = config.get("chunking_strategy", "verse").lower()

    emb_conf = config.get("embedding", {})
    provider = emb_conf.get("provider", "openai")
    model = emb_conf.get("model", "text-embedding-3-small")
    dimension = emb_conf.get("dimension", 1536)

    print(f"[*] Configuration Loaded: {config_path}")
    print(f"[*] Book: '{book_title}' (ID: {book_id})")
    print(f"[*] Strategy: '{strategy}'")
    print(f"[*] Embedding Target: {provider.upper()} ({model}, {dimension} dims)")
    print(f"[*] Tokenizer: tiktoken cl100k_base (100% aligned with {model})")
    print(f"[*] PDF Source: {pdf_path}")

    if strategy == "verse":
        v_conf = config.get("verse_strategy", {})
        max_size = v_conf.get("max_chunk_size", 2000)
        max_tokens = v_conf.get("max_tokens", 600)
        overlap = v_conf.get("chunk_overlap", 0)

        extractor = VersePdfExtractor(config)
        verses = extractor.extract(pdf_path)
        print(f"[+] Extracted {len(verses)} verses across chapters.")

        chunker = VerseChunker(
            book_id=book_id,
            book_title=book_title,
            max_chunk_size=max_size,
            max_tokens=max_tokens,
            chunk_overlap=overlap
        )
        chunks = chunker.chunk_all(verses)
    elif strategy == "standard":
        s_conf = config.get("standard_strategy", {})
        chunk_size = s_conf.get("chunk_size", 1500)
        chunk_overlap = s_conf.get("chunk_overlap", 150)

        extractor = StandardPdfExtractor(config)
        chapters = extractor.extract(pdf_path)
        print(f"[+] Extracted {len(chapters)} chapter text streams.")

        chunker = StandardChunker(book_id=book_id, book_title=book_title, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_all(chapters)
    else:
        raise ValueError(f"Unknown chunking_strategy: '{strategy}'. Must be 'verse' or 'standard'.")

    # Save output (JSONL streaming or standard formatted JSON)
    if output_path.endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        print(f"[+] Streamed {len(chunks)} chunks to JSON Lines format: {output_path}")
    else:
        output_data = [c.to_dict() for c in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"[+] Saved {len(chunks)} chunks to JSON array format: {output_path}")

    # Summary Metrics
    lengths = [c.char_count for c in chunks]
    tokens = [c.token_count for c in chunks]
    print("\n" + "=" * 58)
    print("           CHUNKING PIPELINE SUMMARY (OPENAI ALIGNED)")
    print("=" * 58)
    print(f"  Book ID:                    {book_id}")
    print(f"  Strategy Applied:           {strategy}")
    print(f"  Total Retrieval Chunks:     {len(chunks)}")
    print(f"  Sample Chunk ID:            {chunks[0].chunk_id if chunks else 'N/A'}")
    print(f"  Min Chunk Size:             {min(lengths)} chars | {min(tokens)} BPE tokens")
    print(f"  Max Chunk Size:             {max(lengths)} chars | {max(tokens)} BPE tokens")
    print(f"  Avg Chunk Size:             {sum(lengths)/len(lengths):.1f} chars | {sum(tokens)/len(tokens):.1f} BPE tokens")
    print(f"  Max Token Budget Used:      {max(tokens)} / 8,191 ({max(tokens)/8191*100:.1f}%)")
    print(f"  Image Placeholders Ready:   {all('image_paths' in c.to_dict() for c in chunks)}")
    print("=" * 58 + "\n")

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Strategy Config-Driven Ingestion Pipeline (OpenAI Standardized)")
    parser.add_argument("--config", default="configs/hatha_yoga_pradipika.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    run_pipeline(config_path=args.config)
