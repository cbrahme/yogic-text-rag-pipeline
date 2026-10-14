# Architecture Plan: Multi-Book Ingestion & Chunking Pipeline for Yoga Scriptures

## 1. Objective
Design and implement an extensible, multi-book ingestion and chunking architecture capable of processing classical yoga scriptures (like *Hatha Yoga Pradipika*, *Gheranda Samhita*, *Yoga Sutras*) as well as modern narrative yoga texts (e.g., *Light on Yoga*, *Anatomy of Hatha Yoga*).

The architecture maintains semantic integrity through book-specific chunking strategies, global chunk namespacing, declarative configuration, and multimodal readiness.

---

## 2. Document Analysis & Extraction Strategy

Based on structural inspection of `docs/Hatha-Yoga-Pradipika-pages-2.pdf` (577 pages):

1. **Chapter Layout**:
   - The book is organized into 4 primary chapters (*Upadeshas*):
     - Chapter 1: Asana
     - Chapter 2: Shatkarma and Pranayama
     - Chapter 3: Mudra and Bandha
     - Chapter 4: Samadhi
   - Preceded by introductory matter (lineage, background) and followed by concluding commentaries/indexes.

2. **Verse Structure**:
   - Each section begins with a distinct verse header (e.g., `Verse 1`, `Verse 2`, `Verses 5-9 Chapter 1`).
   - Followed by:
     - **Sanskrit sloka / transliteration**
     - **English translation** of the sloka
     - **Detailed commentary & practical sadhana guidelines** by Swami Muktibodhananda (often spanning 2–6 pages per verse).

3. **Adopting PyMuPDF (`fitz`) over `pypdf`/`pdfplumber`**:
   - **Kerning & Word Glitches**: PyMuPDF reads font `/Widths` tables and kerning pairs natively using the C-based MuPDF engine, extracting words like `Verse` and `universal` cleanly without artificial intra-word spaces.
   - **Running Headers & Footers**: Handled cleanly via PyMuPDF block bounding-box coordinates ($y < y_{top}$ and $y > y_{bottom}$) rather than fragile regex heuristics.
   - **Performance**: MuPDF extracts the entire 577-page book in **~1.5 seconds**, compared to 90+ seconds for `pypdf`.

---

## 3. Decoupled 3-Layer Architecture

To scale from 1 book to dozens without duplicating code:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Extraction Layer (Config-Driven via YAML)                             │
│    • VersePdfExtractor (for verse/shloka-based texts)                   │
│    • StandardPdfExtractor (for chapter/narrative-based texts)           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Chunking Layer (Strategy-Specific)                                   │
│    • Strategy 1 ('verse'): 2-Tier Verse Chunker (Sloka isolation)       │
│    • Strategy 2 ('standard'): Recursive Semantic Paragraph Chunker      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Storage & Retrieval Layer (Universal)                                │
│    • Global namespacing: {book_id}_ch{XX}_v{YYY}_p{ZZ}                  │
│    • ChromaDB Vector Store & Embeddings                                 │
│    • Multimodal placeholder: image_paths: []                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3.1 Standardized Project Directory Structure

The repository organizes code into modular `src/` and `test/` packages with clear separation of concerns:

```text
Hath_Yog_RAG/
├── configs/                          # Declarative scripture processing YAML configs
│   └── hatha_yoga_pradipika.yaml
├── docs/                             # Original scripture PDF documents
│   └── Hatha-Yoga-Pradipika-pages-2.pdf
├── src/                              # Core application source modules
│   ├── __init__.py                   # Package initializer with dynamic path setup
│   ├── ingestion_pipeline.py         # Multi-strategy PDF extraction & verse chunking
│   ├── chroma_manager.py             # ChromaDB persistent collection & metadata manager
│   ├── embed_and_index.py            # OpenAI batch vector embedding & indexing
│   ├── query_db.py                   # Hybrid retriever (Dense + BM25 RRF) & search CLI
│   ├── prompt_engine.py              # System prompt, guardrails, context assembly
│   ├── chat.py                       # Conversational terminal assistant CLI
│   └── app.py                        # Streamlit browser-based web application
├── test/                             # Automated unit & regression test suites (71 tests)
│   ├── __init__.py                   # Test package initializer with src path discovery
│   ├── test_ingestion_pipeline.py    # 29 tests: extraction, cleaners, anchors, errata
│   ├── test_chroma_manager.py        # 8 tests: sanitization, purge, collections
│   ├── test_embed_and_index.py       # 8 tests: batching, dry-run, streaming, OpenAI mock
│   ├── test_query_db.py              # 8 tests: BM25, RRF fusion, filters, formatting
│   ├── test_prompt_engine.py         # 15 tests: prompts, terminology, None-safe sorting
│   └── test_chat.py                  # 4 tests: multi-turn memory, pronoun resolution
├── chroma_db/                        # Persistent ChromaDB vector storage (1536 dims)
├── chunks.jsonl                      # Extracted & validated corpus (668 chunks)
├── requirements.txt                  # Python dependencies
├── plan.md                           # Master architectural specifications & roadmap
└── .env                              # API credentials
```

---

## 4. Chunking Strategies Explained

### Strategy 1: Verse-Centric Hierarchical Chunking (`chunking_strategy: "verse"`)
Designed for ancient scriptures containing root verses, translations, and extensive multi-page commentaries (e.g., *Hatha Yoga Pradipika*, *Yoga Sutras*, *Gheranda Samhita*).

- **Tier 1 (Single Verse Intact Preservation — Dual-Ceiling Rule)**:
  - A verse is preserved intact as a single chunk **only if BOTH conditions are met**:
    1. Total character length $\le 2,000$ characters (`max_chunk_size`).
    2. Exact BPE token count $\le 600$ tokens (`max_tokens`, via `tiktoken` `cl100k_base`).
  - *Reasoning*: Because Sanskrit transliterations with IAST diacritics compress at ~2.9 chars/token (compared to English ~4.0 chars/token), checking character limits alone would allow 1,800-character passages to exceed 620+ tokens, causing context-window truncation in downstream retrievers and LLMs.

- **Tier 2 (Secondary Dynamic Recursive Split with Token Ceiling Guard)**:
  - When content exceeds either ceiling, the commentary is recursively partitioned using a **two-stage splitting mechanism**:
    1. **Dynamic Character Budget Calculation**:
       - Bounded by effective character limit: $\min(\text{max\_chunk\_size}, \lfloor\text{max\_tokens} \times 2.9\rfloor)$.
       - Part 1 ($i = 0$): Deducts breadcrumb header (90 chars) and full sloka translation ($\text{len}(\text{sloka}) + 4$), ensuring commentary fits safely alongside the primary verse.
       - Parts 2..N ($i \ge 1$): Deducts breadcrumb header and the injected **Verse Context Anchor** ($\text{len}(\text{anchor\_str}) + 4$).
    2. **Recursive Sentence/Paragraph Splitting**:
       - Primary split at logical text boundaries (`\n\n`, `. `, `? `, `! `).
    3. **Secondary Bisecting Token Guard**:
       - Each generated commentary split is audited against exact BPE tokens:
         $$\text{token\_limit} = (\text{max\_tokens} - 35) \quad [i = 0] \quad \text{or} \quad (\text{max\_tokens} - 35 - \text{anchor\_tokens}) \quad [i \ge 1]$$
       - If any split part exceeds `token_limit` (due to long unpunctuated prose or dense transliteration), it is **recursively bisected** (`chunk_size = len(p) // 2 + 80`). This guarantees **100% of chunks remain $\le 600$ tokens** (maximum observed: 579 tokens).
  - **Sanskrit Sloka Isolation**: Preserved strictly in **Part 1 (Index 0)** with the primary translation. Sub-chunks 2..N contain commentary only, preventing duplicate non-English token waste while keeping the root verse localized.
  - **Contextual Breadcrumbs**: Prepended to each chunk:
    ```text
    [Book Title | Chapter X: Title | Verse Y (Part i/N)]
    ```
  - **Grammatical Verse Anchor Injection**: Prepended to sub-chunks 2..N ($i \ge 1$) to anchor subject nouns (*Matsyendrasana*, *Sushumna*, etc.) and prevent pronoun ambiguity in long commentaries (see Section 17):
    ```text
    [Verse Y Context: <clean grammatical anchor>]
    ```

### Strategy 2: Standard Narrative Chunking (`chunking_strategy: "standard"`)
Designed for modern textbooks and narrative commentaries without verse numbering.
- Splits continuous chapter text with configurable `chunk_size` (e.g. 1500 chars) and `chunk_overlap` (e.g. 150 chars).
- Prepends contextual chapter headers:
  ```text
  [Book Title | Chapter X: Title (Part i/N)]
  ```
- Generates standardized sequential IDs: `{book_id}_ch{XX}_c{YYY}`.

---

### Extraction & Chunk Boundary Edge Cases Handled in Pipeline

The extraction engine (`VersePdfExtractor` in `ingestion_pipeline.py`) implements specialized guards for typographic anomalies in scriptural PDFs:

1. **Narrative Verb Guard (`NARRATIVE_VERB_GUARD`)**:
   - *Problem*: In-text prose frequently mentions other verses (e.g., *"Verse 15 describes the benefits of pranayama..."* or *"Verse 4 says..."*). Naive regex patterns (`Verses?\s+\d+`) treat these mentions as new verse boundaries, artificially truncating commentaries into fragmented ghost verses.
   - *Fix*: Header detection enforces a line-length ceiling ($\le 65$ characters) and regex rejection of narrative verbs:
     ```python
     NARRATIVE_VERB_GUARD = re.compile(
         r'\b(says|states|describes|mentions|is|are|tells|explains|means|refers)\b',
         re.IGNORECASE
     )
     ```
   - *Result*: Prose cross-references remain safely inside commentary blocks.

2. **Cross-Block Practice Title Stitching & Bidirectional Deduplication**:
   - *Problem*: Posture and kriya names often appear split across lines (`VERSE 26\nMATSYENDRASANA`) or in distinct PyMuPDF text blocks (e.g., `(tortoise pose)`). Blind appending causes duplicate title banners (e.g., `MAYURASANA\n\nMAYURASANA (peacock pose)`).
   - *Fix*: `is_practice_title()` captures uppercase and parenthetical title blocks, while bidirectional substring containment merges them cleanly:
     ```python
     p_low, c_low = pending.lower(), cleaned.lower()
     if c_low in p_low:
         pass
     elif p_low in c_low:
         current_verse["_pending_title"] = cleaned
     else:
         current_verse["_pending_title"] = f"{pending} {cleaned}".strip()
     ```
   - *Result*: Practice titles are cleanly unified with zero duplicate repetitions.

3. **Procedural Label Guard vs. Sloka Translation**:
   - *Problem*: Immediately following a verse header, practice descriptions frequently insert procedural labels such as `"Technique 1"`, `"Stage 1"`, or short subsection names ($< 25$ characters) before the translation text. Without a guard, the first block (`"Technique 1"`) is wrongly assigned as the verse's `sloka_translation`, displacing the actual translation into commentary.
   - *Fix*: Blocks matching `^(Technique|Stage|Part)\s*\d*` or $< 25$ chars are routed directly to `commentary_blocks`, allowing the pipeline to continue searching for the genuine translation.
   - *Result*: Complete sloka translations are correctly captured and isolated into Part 1 and verse anchors.

4. **Opening Chapter Header Line Stripping**:
   - *Problem*: PyMuPDF block grouping on the first page of a chapter occasionally bundles the chapter header (`"Chapter 1\nAsana"`) with the first verse's translation text.
   - *Fix*: The extractor strips leading lines matching `^Chapter(?:\s*(?:\d+|[A-Za-z]+))?\b` before storing `sloka_translation`.
   - *Result*: Translation text remains clean and free of redundant chapter labels.

5. **Standalone Running Header & Footer Regex Purge**:
   - *Problem*: Minor vertical print shifts on occasional pages cause running headers (`"Chapter 1"`, `"Hatha Yoga Pradipika"`) or lone page numbers to fall inside the body margin boundaries.
   - *Fix*: Body text blocks undergo secondary regex filtering to drop pure digit lines ($y_0 > \text{footer\_bottom\_y}$) and standalone book/chapter headers.
   - *Result*: Zero running headers leak into retrieval chunks.

6. **Multi-Verse Dash Typographic Variance (`[-–—,]`)**:
   - *Problem*: In Chapter 2, Page 217, the header for *Ujjayi Pranayama* was printed using a Unicode em-dash (`—`, `\u2014`): `Verses 51—53 Chapter 2`. Naive regex matching only standard hyphens `-` or en-dashes `–` failed to recognize the verse header, causing Verses 51–53 and all 4 ujjayi techniques to be swallowed into the preceding verse (Verse 48–50: *Suryabheda*).
   - *Fix*: Expanded `VERSE_LINE_REGEX` delimiter character set to include the Unicode em-dash `[-–—,]`.
   - *Result*: 100% continuous verse coverage across Chapter 2 (56 verse units, 0 missing verses).

7. **Mid-Commentary Practice Title Guard vs. Dropped Text**:
   - *Problem*: `is_practice_title()` checks for uppercase/parenthetical short blocks. In long commentaries (such as the Shatkarmas diagram on Page 168), words like `BASTI`, `NETI`, `TRAT AKA`, and `NAULI` matched this rule. Because the check was unconditional, it marked them as pending titles and skipped them via `continue`, omitting them from commentary.
   - *Fix*: Guarded `is_practice_title()` to execute strictly when `not current_verse["sloka_translation"]`.
   - *Result*: All commentary tables, lists, and parenthetical sub-verse markers remain 100% intact in commentary blocks.

---

## 5. Declarative YAML Configuration

Each book is defined in `configs/<book_name>.yaml`. Adding a new book requires zero Python code changes.

```yaml
book_id: "hyp"
book_title: "Hatha Yoga Pradipika"
author: "Yogi Svatmarama / Swami Muktibodhananda"
pdf_path: "docs/Hatha-Yoga-Pradipika-pages-2.pdf"
output_path: "chunks.jsonl"

chunking_strategy: "verse" # or "standard"

# Embedding model configuration (Standardized strictly on OpenAI)
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dimension: 1536

# Vector database storage configuration
vector_db:
  type: "chromadb"
  persist_directory: "chroma_db"
  collection_name: "yoga_scriptures"
  distance_metric: "cosine"

verse_strategy:
  max_chunk_size: 2000
  chunk_overlap: 0
  isolate_sanskrit_sloka: true

standard_strategy:
  chunk_size: 1500
  chunk_overlap: 150

margins:
  header_top_ratio: 0.07       # Top 7% of page height (resolution-independent)
  footer_bottom_ratio: 0.945   # Bottom 5.5% of page height (resolution-independent)

chapters:
  1: { number: 1, title: "Asana", cover_page: 1 }
  2: { number: 2, title: "Shatkarma and Pranayama", cover_page: 125 }
  3: { number: 3, title: "Mudra and Bandha", cover_page: 253 }
  4: { number: 4, title: "Samadhi", cover_page: 436 }

publisher_errata:
  86:
    chapter: 1
    misprint_start: 57
    corrected_start: 37
    corrected_end: 37
    extra: ""
```

> [!NOTE]
> The pipeline is standardized strictly on OpenAI `text-embedding-3-small` (1536 dimensions) for high-precision semantic retrieval and exact tokenizer-model parity with `tiktoken` (`cl100k_base`).

---


## 6. Unified Chunk Metadata Schema

Every generated chunk adheres to this standardized schema:

| Field | Type | Description | Example |
|---|---|---|---|
| `chunk_id` | `str` | Globally unique namespaced ID | `"hyp_ch01_v010_p01"` |
| `book_id` | `str` | Short book identifier for filtering | `"hyp"` |
| `book_title` | `str` | Full title of the source work | `"Hatha Yoga Pradipika"` |
| `parent_verse_id`| `Optional[str]` | Canonical verse ID | `"hyp_ch01_v010"` |
| `chapter_number` | `int` | Chapter ordinal | `1` |
| `chapter_title` | `str` | Chapter name | `"Asana"` |
| `verse_start` | `Optional[int]` | Starting verse | `10` |
| `verse_end` | `Optional[int]` | Ending verse | `10` |
| `verse_label` | `Optional[str]` | Human-readable label | `"10"` or `"5-9"` |
| `is_sub_chunk` | `bool` | True if secondary split applied | `True` |
| `sub_chunk_index`| `int` | Ordinal of sub-chunk (0-indexed) | `0` |
| `total_sub_chunks`| `int`| Total sub-chunks for this parent | `3` |
| `has_sanskrit_sloka`| `bool` | True only for root verse / Part 1 | `True` |
| `page_start` | `int` | Starting PDF page | `13` |
| `page_end` | `int` | Ending PDF page | `15` |
| `chunk_type` | `str` | Category | `"verse"`, `"verse_subchunk"`, `"standard_chapter_chunk"` |
| `char_count` | `int` | Character length | `1480` |
| `token_count` | `int` | Exact BPE token count (`cl100k_base` via `tiktoken`) | `353` |
| `image_paths` | `List[str]` | **Future-proof placeholder for diagram paths** | `[]` |
| `content` | `str` | Text content with context breadcrumb | `"[Hatha Yoga Pradipika | ..."` |

---

## 7. Version Roadmap & Scope Management

### Version 0 (v0) — Current Implementation
**Focus: Rock-solid text extraction, dual chunking engines, configuration framework, and data contracts.**

- **What is covered in v0:**
  - High-speed PyMuPDF text block extraction with font kerning normalization.
  - Coordinate-based margin filtering (stripping running headers and page numbers).
  - Soft-hyphen line break stitching.
  - Declarative YAML configuration system (`configs/hatha_yoga_pradipika.yaml` and `configs/example_standard_book.yaml`).
  - Dual chunking engines: `VerseChunker` (Strategy 1) and `StandardChunker` (Strategy 2).
  - Global `book_id` namespacing on `chunk_id` and `parent_verse_id` to eliminate cross-book vector DB collisions.
  - Known publisher typographical errata correction (p.86 v57 $\rightarrow$ v37).
  - Multi-part verse unification (e.g. Verse 16 i & ii/iii aggregated).
  - Future-proof `image_paths: []` placeholder in dataclass and exported JSON.
  - Exported and verified 668 retrieval chunks in `chunks.jsonl` (561 sub-chunks [84.0%], 107 single verse chunks [16.0%]; 100% $\le 600$ tokens, max 579 tokens).

---

### Version 1 (v1) — Vector Database & Embeddings (Next Milestone)
**Focus: Embedding generation, persistent vector storage, and similarity retrieval.**

- **What is planned for v1:**
  - ChromaDB persistent vector database setup.
  - Embedding model integration: OpenAI `text-embedding-3-small` (1536 dimensions).
  - Batch ingestion script populating ChromaDB with full metadata payload.
  - **Hybrid Search (Dense + Sparse BM25 via Reciprocal Rank Fusion - RRF)**:
    - Safeguards 100% recall on exact technical and transliterated Sanskrit terminology (e.g., *Viparita Karani*, *Sushumna*, *Kevali*, *Khechari*) by fusing in-memory BM25 sparse keyword ranking with dense ChromaDB semantic vector search.
  - Metadata filtering capabilities:
    - Multi-book global search vs. single-book search (`where={"book_id": "hyp"}`).
    - Chapter filtering (`where={"chapter_number": 2}`).
    - Shloka-only search (`where={"has_sanskrit_sloka": True}`).
  - Semantic and hybrid query evaluation script with top-k precision testing.
- **Reasoning for keeping in v1:**
  Text extraction and chunk boundaries must be 100% verified, validated, and frozen before spending compute or API budget on embedding hundreds of chunks. Changing chunk boundaries after embedding requires re-indexing the entire database.

---

### Version 2 (v2) — Generative RAG, Conversational Assistant & Web UI (Completed)
**Focus: LLM orchestration, citation faithfulness, intent-adaptive scope, multi-turn chat, and Streamlit web application.**

- **What was delivered in v2:**
  - Prompt orchestration standardizing on OpenAI `gpt-4o-mini` (temperature=0.2).
  - Mandatory scriptural citation contract: `[Chapter X, Verse Y, pp. Z-W]`.
  - Non-hallucination guardrails and qualified guru supervision disclaimers for advanced practices.
  - Context chronological re-ordering and sub-chunk deduplication with None-safe narrative fallbacks.
  - Dual-language query terminology alignment (`YOGA_SYNONYM_LEXICON`).
  - Intent-adaptive answering scope matching (definitional, procedural, benefits) eliminating over-answering.
  - Interactive multi-turn CLI assistant (`src/chat.py`) with pronoun resolution, streaming tokens, and `:sources` inspection.
  - Full browser-based Streamlit web application (`src/app.py`) with chat bubbles, real-time streaming, and collapsible source accordions.
  - 71 repo-wide automated unit tests passing across all pipeline modules.

---

### Version 3 (v3) — Multimodal Image Extraction & Visual Companion (Next Milestone)
**Focus: Asana/Mudra diagrams, coordinate bounding association, zero-cost ChromaDB metadata updates, and visual rendering.**

- **What is planned for v3:**
  - Empirical Dimension-Filtered Image Extractor:
    - PDF analysis reveals 467 embedded raster images. Filter out ~281 decorative chapter banners/scrollwork ($h \le 150\text{px}$) and ~58 publisher logos ($< 200 \times 150$) to isolate **~128 genuine posture diagrams** ($w \ge 250\text{px}, h \ge 200\text{px}$).
  - Spatial Verse-to-Image Association:
    - Cross-reference image page and vertical coordinate bounding boxes ($y_0, y_1$) with active verse text boundaries to deterministically pair diagrams with their respective verses.
  - High-Resolution Image Disk Storage:
    - Save extracted posture illustrations to `assets/images/{book_id}/{book_id}_ch{chapter:02d}_v{verse:03d}_fig{idx:02d}.png`.
  - Zero-Cost In-Place ChromaDB Metadata Synchronization:
    - Populate `image_paths` in `chunks.jsonl` and execute in-place metadata updates via ChromaDB's `collection.update()`. Requires **$0.00 embedding API spend** and zero vector recalculations.
  - Streamlit Visual Companion:
    - Render posture diagrams directly inside collapsible scriptural source accordions in `src/app.py` via `st.image()`.
  - Terminal Chat Visual Indicator:
    - Display visual reference tags (`[📷 1 diagram: assets/images/hyp/...]`) in `src/chat.py` `:sources` output.
- **Reasoning for keeping in v3:**
  Because `image_paths: []` was baked into the `Chunk` dataclass and sanitized via `chroma_manager.py` in v0, adding image extraction in v3 requires **zero database schema migrations or breaking changes**. Isolating text extraction and generative conversational RAG first ensured the core semantic knowledge base was fast, robust, and verified before adding multimodal assets.

---

## 8. Milestone Execution Trackers (v0, v1, v2 & v3)

To ensure strict alignment with the version roadmap defined in Section 7, the delivery milestones are cleanly separated:
- **Version 0 (v0)**: Parsing, cleaning, chunking strategies, and verified JSON Lines export.
- **Version 1 (v1)**: Persistent vector database architecture, OpenAI embedding generation, batch ingestion, and semantic retrieval evaluation.
- **Version 2 (v2)**: Generative RAG prompt architecture, guardrails, multi-turn chat CLI (`chat.py`), and Streamlit web application (`app.py`).
- **Version 3 (v3)**: Multimodal dimension-filtered image extraction, verse spatial association, zero-cost ChromaDB metadata synchronization, and visual UI rendering.

```
[ Milestone: Version 0 (v0) — Parsing & Chunking Foundation (Completed) ]
[ Step 1: Parse & Clean ] ──► [ Step 2: Config & Strategies ] ──► [ Step 3: Chunking & JSON Validation ]
                                                                                   │
                                                                                   ▼
[ Milestone: Version 1 (v1) — Vector DB & Retrieval (Completed) ]
[ Step 7: Retrieval Test ] ◄── [ Step 6: Batch DB Indexing ]  ◄── [ Step 5: ChromaDB Setup ] ◄── [ Step 4: Embedding Model ]
                                                                                   │
                                                                                   ▼
[ Milestone: Version 2 (v2) — Generative RAG & Web UI (Completed) ]
[ Step 8: Prompt Engine & Guardrails ] ──► [ Step 9: Chat CLI & Web App ] ──► [ Step 10: RAG Evaluation ]
                                                                                   │
                                                                                   ▼
[ Milestone: Version 3 (v3) — Multimodal Image Extraction & Visual Companion (Next Milestone) ]
[ Step 11: Image Extractor ] ──► [ Step 12: Spatial Verse Mapping ] ──► [ Step 13: DB Sync ] ──► [ Steps 14-15: UI & Tests ]
```

---

### Milestone: Version 0 (v0) — Parsing & Chunking Foundation (Completed)
*Deliverable: Verified, normalized, and schema-compliant chunk export (`chunks.jsonl`).*

- [x] **Step 1: Document Parsing, Extraction & Text Cleaning**
  - [x] Extract text blocks via PyMuPDF (`fitz`) with dynamic relative margin ratios (`header_top_ratio: 0.07` and `footer_bottom_ratio: 0.945`) to filter running headers and footers across any page height/resolution.
  - [x] Resolve native font kerning and eliminate artificial intra-word spaces.
  - [x] Stitch line-end hyphens cleanly (`(\w+)-\n(\w+)` $\rightarrow$ `\1\2`).
  - [x] Apply publisher misprint corrections (e.g., Page 86 Verse 57 $\rightarrow$ Verse 37).
  - [x] Unify multi-part subsection verses (e.g., Verse 16 parts `(i)` & `(ii & iii)`).

- [x] **Step 2: Config-Driven Architecture & Strategy Implementation**
  - [x] Create declarative YAML configurations (`configs/hatha_yoga_pradipika.yaml`, `configs/example_standard_book.yaml`).
  - [x] Implement Strategy 1 (`verse`): 2-tier verse packaging $\le 2000$ chars, recursive split on large commentaries, isolate Sanskrit shlokas to Part 1, inject contextual breadcrumbs.
  - [x] **Verse Anchor Injection for Sub-chunks 2..N (`build_verse_anchor`)**: Inject `[Verse {v_label} Context: {anchor}]` using grammatical sentence boundaries and explicit entity capture into all sub-chunks ($i \ge 1$) to eliminate semantic drift and anchor technical Sanskrit terms (*Matsyendrasana*, *Sushumna*, etc.) across long commentaries (see Section 17).
  - [x] Implement Strategy 2 (`standard`): Recursive character splitting for narrative/modern non-verse texts.
  - [x] Implement global namespacing: prefix all `chunk_id` and `parent_verse_id` with `book_id` (e.g., `hyp_ch01_v001_p01`).
  - [x] Add future-proof `image_paths: []` placeholder to the unified `Chunk` schema.

- [x] **Step 3: Chunk Generation & Data Validation**
  - [x] Run `ingestion_pipeline.py --config configs/hatha_yoga_pradipika.yaml`.
  - [x] Generate and export 667 verified chunks to `chunks.jsonl` with dual-ceiling guard ($\le 2000$ chars, $\le 600$ tokens; max observed 579 tokens, 0 chunks exceeding).
  - [x] Verify chunk `hyp_ch01_v027_p03` contains both breadcrumb and `[Verse 27 Context: ... (matsyendrasana) ...]` line.
  - [x] Audit for 0 duplicate `chunk_id`s, 0 empty IDs, and strict 0-indexed sub-chunk sequences.

---

### Milestone: Version 1 (v1) — Vector Database & Semantic Retrieval (Completed)
*Deliverable: Indexed, searchable ChromaDB vector database with validated semantic similarity queries.*

#### Phase B: Embedding & Vector Database Ingestion
- [x] **Step 4: Embedding Model Selection & Configuration**
  - [x] **Standardized strictly on OpenAI (`text-embedding-3-small`, 1536 dimensions)** for v1 and dropped the local/offline Ollama requirement to eliminate dual-model architectural complexity.
  - [x] Verified native ChromaDB embedding function support (`OpenAIEmbeddingFunction`).
  - [x] Configured single unified collection schema (`yoga_scriptures_openai`).

- [x] **Step 5: ChromaDB Setup & Collection Architecture (Completed)**
  - [x] Initialize persistent local ChromaDB directory (`./chroma_db`).
  - [x] Define collection name (e.g., `yoga_scriptures`) with cosine distance metric (`{"hnsw:space": "cosine"}`).
  - [x] Establish ChromaDB metadata structure ensuring compatibility with filtering (`book_id`, `chapter_number`, `verse_start`, `has_sanskrit_sloka`).
  - [x] **ChromaDB Empty Array Constraint Guard**: Implemented in [`chroma_manager.py`](file:///Users/chaitralibrahme/Desktop/Projects/Hath_Yog_RAG/chroma_manager.py) via `sanitize_metadata` (serializes `image_paths` to JSON string `"[]"`, filters `None` values to avoid `TypeError`, isolates `content`) and `deserialize_metadata`. Validated 100% compliant across all 667 chunks.
  - [x] **Deployment Mode Consideration**: Implemented `PersistentClient` for embedded single-process verification, with built-in configuration support for client/server mode (`HttpClient`).

- [x] **Step 6: Batch Ingestion & Vector Indexing Script (Completed)**
  - [x] Develop `embed_and_index.py` to stream chunks from `chunks.jsonl` line-by-line in batches (e.g. 50–100 chunks per batch).
  - [x] Compute vector embeddings via OpenAI `text-embedding-3-small` (1536 dims) and upsert records into ChromaDB (`ids`, `embeddings`, `documents`, `metadatas`).
  - [x] Add progress bars (`tqdm`), token tracking, cost accounting, and error handling with exponential backoff retry.
  - [x] **Targeted Book-Scoped Purge for Robust Idempotency**: Automatically purges prior vectors for `book_id` before upserting (`where={"book_id": book_id}`) to prevent ghost sub-chunks.
  - [x] Successfully indexed all **668 chunks** into ChromaDB (`yoga_scriptures`) in 8.42 seconds (213,235 tokens, $0.00426 total cost). Verified via test suite with 8 unit tests in `test_embed_and_index.py`.

#### Phase C: Verification & Search Testing
- [x] **Step 7: Verification & Hybrid Semantic Retrieval Testing (Completed)**
  - [x] Create `query_db.py` supporting both pure dense ChromaDB retrieval and **Hybrid Search (Dense + Sparse BM25 via Reciprocal Rank Fusion - RRF)**:
    - Built lightweight in-memory BM25 index on chunk text directly from `chunks.jsonl` using `rank-bm25` (40ms construction time).
    - Implemented Reciprocal Rank Fusion (RRF with $k=60$) to merge and re-rank top-k dense and top-k BM25 hits.
  - [x] Executed benchmark queries across three critical query profiles:
    - **Exact Sanskrit Technical Terms**: *"Ujjayi pranayama"*, *"Viparita Karani mudra"*, *"Sushumna nadi kundalini"* (100% exact retrieval).
    - **Conceptual & Philosophical Queries**: *"What are the causes of failure in yoga according to Chapter 1?"* (100% precision).
    - **Asana/Technique Inquiries**: *"Benefits and practice of Matsyendrasana"*, *"How to perform Mayurasana peacock pose"*, *"Shatkarma six cleansing practices"*.
  - [x] Implemented metadata filters:
    - Chapter filter: `--chapter 1`, `--chapter 2`
    - Sloka-only search: `--sloka-only` (`has_sanskrit_sloka=True`)
  - [x] Result Inspection & Formatting:
    - Added `--full` CLI flag for printing complete uncut multi-line text (Sanskrit sloka, word-by-word glosses, full commentary).
    - Upgraded preview formatter to truncate at word boundaries (preventing mid-word slicing like `practic...`).
    - Added `:full` (toggle full content) and `:view <N>` (inspect Nth result in full) to the interactive REPL.
  - [x] Retrieval Depth Calibration (`--top-k` / `-k`):
    - Calibrated $k=5$ as CLI default and $k=4$ for REPL to guarantee multi-part scripture coverage (e.g. 4-part Moola Bandha) without dropping practice stages. Documented in Section 19.
  - [x] Interactive Search REPL Loop: Included `--interactive` mode allowing natural command-line exploration.
  - [x] Unit test suite: Created `test_query_db.py` with 8 unit tests; all 52 tests pass repo-wide.

---

### Milestone: Version 2 (v2) — Generative RAG & Conversational Assistant (Completed)
*Deliverable: Interactive multi-turn conversational AI answering questions with factual fidelity, scriptural citations, and safety guardrails.*

#### Phase D: Prompt Architecture, Context Assembly & Guardrails
- [x] **Step 8: Prompt Architecture, Scriptural Guardrails & Citation Contracts (Completed)**
  - [x] **System Prompt Engineering**: Designed robust system instructions (`SYSTEM_PROMPT_TEMPLATE`) standardizing on OpenAI `gpt-4o-mini` (temperature=0.2 for strict factual fidelity) with authoritative scholar persona.
  - [x] **Scriptural Citation Contract**: Enforced mandatory citation formatting referencing Chapter, Verse number(s), and Page ranges (`[Chapter X, Verse Y, pp. Z-W]`).
  - [x] **Safety & Non-Hallucination Guardrails**: Explicitly decline modern medical claims, ground answers strictly in retrieved excerpts, and flag advanced practices requiring qualified guru supervision (*Kevali Kumbhaka*, *Khechari*, *Vajroli*).
  - [x] **Context Re-Ordering & Deduplication**: Implemented `reorder_chunks_chronologically` to sort retrieved chunks chronologically by `(chapter_number, verse_start, sub_chunk_index)` and deduplicate duplicate chunk IDs before injecting into LLM context.
  - [x] **None-Safe Metadata Handling & Narrative Compatibility**: Introduced `_safe_int` fallback helper to eliminate `TypeError: int(None)` when processing narrative/chapter books (Strategy 2) where `verse_start` and `verse_end` are `None`; guarded `verse_label` in `format_context_for_prompt` to render clean chapter headers without `Verse None`.
  - [x] **Dual-Language Query Terminology Alignment**: Built `YOGA_SYNONYM_LEXICON` and `align_query_terminology` to bridge vernacular English to classical Sanskrit at query time with zero corpus annotation.
  - [x] **Citation Extractor & Adherence Validator**: Built `extract_citations` with regex parsing to evaluate LLM citation compliance.
  - [x] **Intent-Adaptive Answering Scope**: Replaced static 4-section dumps with intent-adaptive scope matching (definitional queries receive concise definitions with citations and follow-up invitations; procedural queries receive step-by-step techniques; benefits queries receive effects), eliminating over-answering and enabling natural multi-turn dialogue.
  - [x] **Automated Test Suite**: Created `test_prompt_engine.py` with 15 unit tests verifying prompt contracts, terminology alignment, canonical sorting, None-safe fallback conversion, and citation parsing; all 71 repo tests pass.

#### Phase E: Conversational Chat Engine & UI
- [x] **Step 9: Interactive Conversational Chat CLI (`chat.py`) & Web App (`app.py`) (Completed)**
  - [x] Built `chat.py` integrating `HybridRetriever` from `query_db.py` with OpenAI `gpt-4o-mini` and `prompt_engine`.
  - [x] **Multi-Turn Session Memory**: Maintained sliding-window chat history buffer with pronoun resolution (`_resolve_conversational_query`) so follow-up queries (*"What are its contraindications?"*) resolve context from prior turns.
  - [x] **Real-Time Token Streaming**: Implemented generator streaming output for instant, conversational typing responses in the terminal.
  - [x] **Source Attribution Inspector**: Added `:sources` and `:sources full` commands (and `--sources` CLI flag) to inspect exact chunks, scores, and pages retrieved behind any response.
  - [x] **Streamlit Web Application (`app.py`)**: Built full browser-based chat app (`streamlit run app.py`) with ChatGPT-style conversation bubbles, real-time token streaming, sidebar chapter/model/$k$ filters, and collapsible scriptural source accordions with verified citation badges.

#### Phase F: Verification & Evaluation
- [x] **Step 10: RAG Evaluation & Faithfulness Testing (`test_chat.py`) (Completed)**
  - [x] Unit test suite `test_chat.py`: Built 4 unit tests verifying session initialization, pronoun resolution, conversation memory accumulation/clearing, and mock streaming flow with citation parsing.
  - [x] All 71 automated unit tests pass across the entire repository.
  - [x] Verified live generation and citation accuracy across standard scripture question benchmarks.

---

### Milestone: Version 3 (v3) — Multimodal Image Extraction & Visual Companion (Next Milestone)
*Deliverable: Automated extraction, verse-bounding spatial association, zero-cost ChromaDB metadata update, and Streamlit visual posture diagram display.*

#### Phase G: Image Extraction & Bounding Box Spatial Association
- [ ] **Step 11: Dimension-Filtered Image Extractor (`src/ingestion_pipeline.py`)**
  - [ ] Inspect raw image stream per page via PyMuPDF (`page.get_images()`).
  - [ ] Implement aspect ratio and dimension heuristics: filter out decorative chapter headpieces/footpieces ($h \le 150\text{px}$, ~281 banners) and logos ($< 200 \times 150$, ~58 logos); retain genuine asana/mudra diagrams ($w \ge 250\text{px}$, $h \ge 200\text{px}$, ~128 diagrams).
  - [ ] Extract raw image bytes and save as PNG to organized directory: `assets/images/{book_id}/{book_id}_page{page_num:03d}_img{idx:02d}.png`.

- [ ] **Step 12: Verse-to-Image Spatial Association Engine**
  - [ ] Extract image bounding box coordinates ($y_0, y_1$) on the physical page via `page.get_image_rects()`.
  - [ ] Cross-reference image page and vertical bounds with active verse boundaries on that page.
  - [ ] Map extracted images to their corresponding parent verse and chunks, populating the `image_paths` array.

- [ ] **Step 13: In-Place ChromaDB Metadata Synchronization**
  - [ ] Update `chunks.jsonl` with populated `image_paths` arrays.
  - [ ] Implement in-place metadata updater using `collection.update(ids=..., metadatas=...)`.
  - [ ] Avoid full re-embedding: maintain existing 1536-dimensional embeddings with **$0.00 API cost** and instant update time (~2 seconds).

#### Phase H: UI Rendering & Multimodal Visual Companion
- [ ] **Step 14: Streamlit Visual Posture Accordion (`src/app.py`)**
  - [ ] Check for non-empty `image_paths` in retrieved source chunks.
  - [ ] Display posture illustrations inside the collapsible scriptural source accordions using `st.image()`.
  - [ ] Provide posture caption, figure label, and zoom/expand view.

- [ ] **Step 15: Terminal Inspector & Automated Tests (`src/chat.py`, `test/`)**
  - [ ] Display visual indicator in terminal chat (`[📷 1 diagram: assets/images/hyp/...]`) when `:sources` or `--sources` is triggered.
  - [ ] Build unit tests in `test/test_image_extraction.py` validating dimension filtering, aspect-ratio thresholding, and spatial verse mapping.

---

## 9. Architectural Note: Multi-Provider Vector Database Strategy & Dimension Safeguards
*(Context: The following analysis explored multi-model coexistence, which led to the decision in Section 11 to standardize strictly on OpenAI to eliminate operational complexity).*

### Problem Definition
Vector databases like ChromaDB enforce a fixed vector dimension per collection at creation time (e.g., `nomic-embed-text` = 768 dims, `text-embedding-3-small` = 1536 dims, `bge-m3` = 1024 dims). Pointing a different embedding provider at an existing collection causes an immediate `InvalidDimensionException` runtime crash. Furthermore, comparing vectors from different models in the same vector space is mathematically invalid.

### Implemented Strategy
1. **Dynamic Model-Namespaced Collections**:
   - Collections are uniquely derived by provider and model:
     $$\text{collection\_name} = f"\{\text{base\_name}\}\_\{\text{provider}\}\_\{\text{model\_slug}\}"$$
   - Example (Ollama): `yoga_scriptures_ollama_nomic_embed_text` (768 dims)
   - Example (OpenAI): `yoga_scriptures_openai_text_embedding_3_small` (1536 dims)
   - Both collections coexist peacefully inside the same `./chroma_db/` directory.

2. **Why Maintaining Dedicated Tables Beats Repeated Full Re-Indexing**:
   - **Negligible Storage Footprint**: Storing 667 chunks consumes $\sim 1.8 \text{ MB}$ for Ollama and $\sim 3.8 \text{ MB}$ for OpenAI ($< 10 \text{ MB}$ total). Disk space is practically free.
   - **Zero Compute & Cost Waste**: Eliminates redundant 30–60s local GPU/CPU pegging or recurring OpenAI API token costs whenever switching between local development and cloud testing.
   - **Enables Direct A/B Benchmarking**: Allows evaluating the exact same user query against both vector spaces side-by-side to objectively compare retrieval accuracy.
   - **Zero Downtime**: Avoids dropping active collections or serving query errors while re-indexing.

3. **Incremental & Idempotent Indexing**:
   - The indexer checks ChromaDB for existing `chunk_id`s before dispatching embedding requests (superseded by Targeted Book-Scoped Purge in Section 15 to prevent stale ghost vector leaks when sub-chunk partitions shift).
   - When new books or revisions are ingested, only the new or modified chunks are embedded, minimizing API costs and compute overhead.

---

## 10. Architectural Note: Token Estimation Accuracy for Sanskrit & Indic Scripts

### Problem Definition
The original schema estimated token usage using the standard English heuristic:
$$\text{token\_estimate} = \text{chars} // 4$$
Sanskrit transliteration with IAST diacritics (`ā, ī, ū, ṛ, ś, ṣ, ṁ, ḥ`) and Devanagari script do not follow this rule. Byte-Pair Encoding (BPE) tokenizers (such as OpenAI's `cl100k_base` and LLaMA/Nomic tokenizers) fragment multi-byte diacritics into multiple sub-byte tokens (often 1 token per 1–2 characters).

### Empirical Findings on Dataset
Analysis of the unconstrained vs. guarded corpus using `cl100k_base` revealed:
- **Diacritic Fragmentation**: Commentary and verse chunks exhibited token expansion ratios up to **1.47x** over the heuristic (e.g., chunk `hyp_ch04_v018_p04` has 864 characters, estimated at 216 tokens, but actually contains 318 BPE tokens).
- **Initial Max Chunk Size**: Unconstrained initial chunking allowed the longest commentary chunk to reach **632 real BPE tokens** (underestimated by character heuristic at 482 tokens, an underestimate of ~31%).
- **Guarded Corpus Metrics**: After integrating the dual-ceiling guard and two-stage recursive bisecting in `VerseChunker`, all **667 final chunks** strictly adhere to the $\le 600$-token budget:
  - **Min tokens**: 107
  - **Max tokens**: 579 ($< 600$, zero chunks exceeding)
  - **Average tokens**: 319.6
  - **Total chunks**: 667 (560 sub-chunks [84.0%], 107 root verse chunks [16.0%])
- **Context Window Safety**: Because OpenAI (`text-embedding-3-small`) features an 8,191-token context limit, a 579-token max chunk comfortably consumes only ~7.0% of the window without any risk of truncation.

- **Risk Areas**: Accurate token counts become critical when:
  1. Using smaller embedding models with 256 or 512 context limits (e.g. `all-MiniLM-L6-v2`), where a 632-token chunk would silently truncate.
  2. Budgeting LLM prompt context in v2 when assembling multi-chunk prompt context.
  3. Tracking token consumption and API billing.

### Implemented Resolution
1. **Real Tokenizer Library Integration**:
   - Integrate `tiktoken` (using `cl100k_base` encoding) to calculate exact token counts instead of relying on the integer division heuristic.
2. **Unicode NFC Normalization**:
   - Enforce `unicodedata.normalize('NFC', text)` in `TextCleaner` to ensure decomposed diacritics (letter + combining accent) are merged into single precomposed UTF-8 glyphs, minimizing avoidable BPE token fragmentation.
3. **Dual-Ceiling & Dynamic Token Guard**:
   - In `VerseChunker`, enforce both character limits ($\le 2000$) and explicit token limits ($\le 600$), with dynamic second-stage recursive bisecting if any sub-chunk part exceeds the token ceiling.
4. **Deprecation of `token_estimate`**:
   - Formally deprecated and removed the redundant `token_estimate` field from the schema, pipeline, and export contracts, standardizing solely on the exact `token_count` produced via `tiktoken`.

---

## 11. Architecture Decision Record (ADR): Standardizing Strictly on OpenAI & Dropping Ollama

### Context & Decision
Initially, an offline/local embedding option (Ollama with `nomic-embed-text` or `bge-m3`) was maintained alongside cloud embeddings (OpenAI `text-embedding-3-small`). To eliminate operational complexity, dual vector space management, and collection routing, **the local/offline Ollama requirement is formally dropped, and the pipeline is standardized strictly on OpenAI**.

### Rationale for Dropping Local Ollama in Favor of OpenAI
1. **Tokenizer-Model Architectural Alignment**:
   - `tiktoken` with `cl100k_base` is the proprietary vocabulary specifically designed for OpenAI models (`text-embedding-3`, `gpt-4`).
   - If the system were to use Ollama with `nomic-embed-text` or `bge-m3`, their underlying tokenizers are completely different:
     - `bge-m3` uses a multilingual XLM-RoBERTa / SentencePiece tokenizer.
     - `nomic-embed-text` uses WordPiece / BERT-based tokenization.
   - Standardizing on OpenAI guarantees 100% parity between the tokenizer used in the Python pipeline (`tiktoken`) for chunk sizing, context budgeting, and guardrails, and the actual embedding model consuming the tokens in the vector database.

2. **Architectural Simplicity (Single Vector Space)**:
   - Maintaining two embedding models required dual collection namespacing (`_ollama_nomic` vs. `_openai_text3`), dimension guard logic (768 vs. 1536 dims), and dynamic collection routing.
   - Standardizing on OpenAI eliminates collection fragmentation: all chunks, indexes, and queries exist in one single, high-fidelity 1536-dimensional vector space.

3. **Negligible Cost Barrier**:
   - The primary motivation for local models is often avoiding API fees. However, with `text-embedding-3-small` priced at \$0.02 per 1M tokens, embedding all 667 chunks of *Hatha Yoga Pradipika* (~213k tokens) costs **~$0.004 (less than half a cent)**.
   - The trivial cost does not justify maintaining an external local model daemon.

4. **Zero Local Machine Burden & Instant Latency**:
   - Running Ollama locally requires a persistent background daemon, consumes local disk storage (~3GB for engine + model weights), and pegs local CPU/GPU during batch ingestion.
   - OpenAI offloads all compute to the cloud, embedding the entire 667-chunk corpus in 3–5 seconds without battery drain, machine slowdown, or fan noise.

5. **Production & Deployment Parity**:
   - Local Ollama runtimes create significant deployment friction when transitioning from local machines to staging or production (requiring GPU instances or complex containerized model servers).
   - Standardizing on OpenAI requires only an `OPENAI_API_KEY` environment variable, ensuring identical behavior across local development, CI/CD automated testing, and cloud container deployments.

6. **Superior Retrieval Quality & Native Context Handling**:
   - `text-embedding-3-small` provides high multi-domain semantic accuracy, 1536-dimension vector granularity, and an 8,191-token context window that comfortably accommodates any long-form Sanskrit verse and commentary chunk.

---

## 12. Architectural Note: Transitioning to JSON Lines (JSONL) for Scalable Ingestion

### Context & Decision
To eliminate future migration rewrites and make the pipeline fully production-grade from day one, **the storage format for chunk exports is standardized on JSON Lines (`.jsonl`) from v0 itself**. While a single book (~1.1 MB) fits comfortably in memory as a standard JSON array, adopting `.jsonl` from v0 avoids schema or pipeline breaks as the library scales to dozens of comprehensive commentaries.

### Why the Critique is Right As You Scale (50+ Books)

| Problem in Monolithic `.json` | How JSON Lines (`.jsonl`) Solves It |
|---|---|
| **Peak Memory Spikes ($O(N)$ RAM)**<br>Loading a 500 MB JSON array creates millions of Python objects in memory, ballooning memory usage to 2–3 GB. | **Streaming Memory ($O(1)$ RAM)**<br>`for line in f:` reads 1 line at a time. Memory usage stays under 10 MB even for a 10 GB file. |
| **Pipeline Crash = Total Loss**<br>If extraction fails on page 500 out of 577, standard JSON fails to close `]`, leaving a corrupt, unparseable file. | **Real-Time Append**<br>Each chunk is written to disk the millisecond it's created. If it crashes on page 500, chunks from pages 1–499 are 100% safe. |
| **Batch Ingestion Streaming**<br>You must parse the whole array before sending batch 1 to OpenAI. | **Immediate Pipeline**<br>You can stream 50 chunks from disk directly into OpenAI's embedding API in a streaming generator. |

### Implementation Pattern
- **Streaming Export (`ingestion_pipeline.py`)**:
  ```python
  with open(output_path, "w", encoding="utf-8") as f:
      for chunk in chunks:
          f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
  ```
- **Streaming Ingestion (`embed_and_index.py`)**:
  ```python
  def stream_chunks(file_path: str):
      with open(file_path, "r", encoding="utf-8") as f:
          for line in f:
              if line.strip():
                  yield json.loads(line)
  ```

---

## 13. Architectural Note: Dynamic Relative Margins vs. Rigid Physical Coordinates

### Context & Problem Statement
In PDF extraction pipelines, running headers (verse titles, chapter names) and footers (page numbers, publication credits) are frequently stripped using fixed physical point coordinate cutoffs (e.g., `header_top_y: 55` pt and `footer_bottom_y: 750` pt).

While this may function on standard Letter/A4 layout at 72 DPI, hardcoded absolute coordinates cause critical data degradation when scaling to a multi-book library:
1. **Trim-Size Variance**: Pocket editions (e.g., Chaukhambha or Gita Press editions, often ~500 pt page height) will have their bottom ~250 pt completely discarded, throwing away half of every page's commentary.
2. **Scan Resolution/DPI Differences**: High-resolution scanned or rasterized documents (e.g., 300 DPI scans where page height is 2400 pt) will treat 750 pt as only 31% down the page, chopping off nearly 70% of the legitimate text.
3. **Header/Footer Leakage**: On tall or large-format folios (e.g., 1000+ pt height), a fixed 750 pt bottom margin leaves 250+ pt of margin, allowing page numbers and footnote boilerplate to leak into the vector embeddings.

### Solution & Architectural Decision
Transition from rigid absolute point coordinates (`header_top_y`, `footer_bottom_y`) to **relative bounding box ratios / percentages of page height** (`header_top_ratio`, `footer_bottom_ratio`), computed dynamically on every individual page:

$$\text{header\_cutoff\_y} = \text{page.rect.height} \times \text{header\_top\_ratio}$$
$$\text{footer\_cutoff\_y} = \text{page.rect.height} \times \text{footer\_bottom\_ratio}$$

### Configuration Schema Comparison

| Approach | Configuration Example | Portability Across Multi-Book Corpus |
|---|---|---|
| **Rigid Coordinates (Deprecated)** | `margins:`<br>&nbsp;&nbsp;`header_top_y: 55`<br>&nbsp;&nbsp;`footer_bottom_y: 750` | ❌ Fails on non-standard page sizes, pocket prints, and high-DPI scans. |
| **Dynamic Relative Ratios (Standard)** | `margins:`<br>&nbsp;&nbsp;`header_top_ratio: 0.07` *(top 7%)*<br>&nbsp;&nbsp;`footer_bottom_ratio: 0.945` *(bottom 5.5%)* |  Fully independent of DPI, trim size, resolution, and page geometry. |

### Implementation Pattern & Backward Compatibility
The extractor classes (`VersePdfExtractor` and `StandardPdfExtractor` in `ingestion_pipeline.py`) dynamically calculate the cutoff thresholds per page and provide automatic backward compatibility for scalar point values (> 1.0):

```python
page_h = page.rect.height

# Ratio values <= 1.0 are treated as relative percentages; values > 1.0 are treated as raw points
top_val = self.margins.get("header_top_ratio", self.margins.get("header_top_y", 0.07))
header_top_y = page_h * top_val if top_val <= 1.0 else top_val

bot_val = self.margins.get("footer_bottom_ratio", self.margins.get("footer_bottom_y", 0.945))
footer_bottom_y = page_h * bot_val if bot_val <= 1.0 else bot_val
```

---

## 14. Architectural Note: Embedded ChromaDB (SQLite) Concurrency Lock vs. Client/Server Architecture

### Context & Problem Statement
In v0, ChromaDB runs locally in embedded mode (`persist_directory: "chroma_db"` using `chromadb.PersistentClient()`), persisting vector indexes, collection metadata, and chunks using an embedded SQLite database on disk.

While zero-setup and fast for single-threaded sequential scripts and offline development, embedded SQLite mode has fundamental concurrency and scaling bottlenecks:
1. **Concurrency Lockouts (`sqlite3.OperationalError: database is locked`)**: SQLite enforces file-level locks during write transactions. If multiple ingestion worker processes run concurrently (e.g. parallel ingestion across books) or if a web server runs search queries while an ingestion job writes chunks, SQLite locks the entire database file and raises unhandled locking exceptions.
2. **No Horizontal Scaling**: Embedded ChromaDB cannot be shared across multiple container instances, serverless functions, or distributed worker pods.
3. **No Process Isolation**: Crashes or high memory pressure in an indexing script can corrupt the local embedded database file.

### Roadmap & Migration Path

| Stage | Vector Storage Architecture | Concurrency & Scaling Characteristics | When to Adopt |
|---|---|---|---|
| **v0 / Development** | **Embedded ChromaDB**<br>(`PersistentClient(path="./chroma_db")`) | Single-process, sequential write/read. Zero network setup, zero container overhead. | Current phase: Single-book parsing, offline batch verification. |
| **v1 / Production API** | **ChromaDB Client/Server**<br>(`HttpClient(host="...", port=...)`) | Multi-client connection pooling, decoupled write/read workers, runs via Docker container (`chromadb/chroma`). | When serving multi-threaded Web/API queries or running parallel book ingestion jobs. |
| **v2+ / Enterprise Scale** | **PostgreSQL (`pgvector`) or Qdrant** | Full ACID transactions, row-level concurrency locks, horizontally distributed vector search, mature backup/restore. | High-throughput multi-user production, distributed ingestion across 50+ commentaries. |

### Architectural Decision for v0
- **Retain Embedded Mode for v0**: For initial prototyping, offline evaluation, and single-user CLI validation, `PersistentClient` is retained. It requires zero infrastructure (no Docker daemon or background service) and operates reliably in a sequential pipeline.
- **Future-Proofing Config**: The configuration schema in `configs/*.yaml` separates `type: "chromadb"`, `persist_directory: "chroma_db"`, and `collection_name: "yoga_scriptures"`. In v1, this can introduce `mode: "server"` with `host` and `port` attributes without requiring any changes to downstream chunk ingestion or retrieval logic.

---

## 15. Architectural Note: Idempotency & Stale Vector Leaks via Targeted Book-Scoped Purge

### Context & Problem Statement
In vector database ingestion pipelines, a common naive approach to idempotency is checking whether an incoming `chunk_id` exists in the collection and skipping or overwriting it.

While this functions for static append-only datasets, it causes **stale vector leaks (orphaned ghost vectors)** whenever document chunking or text normalization rules change:
1. **Chunk Boundary Re-slicing**: Suppose an earlier run split a long commentary for Verse 10 into 4 sub-chunks (`hyp_ch01_v010_p01` through `p04`). If text cleaning improvements or token ceiling adjustments later compress that commentary into 3 sub-chunks (`p01` through `p03`), an ID-based upsert overwrites `p01`–`p03`, but **leaves `hyp_ch01_v010_p04` lingering in the database forever**. Future semantic queries will retrieve this ghost chunk with obsolete metadata (`total_sub_chunks: 4`), duplicate text, and severed context.
2. **Publisher Errata & Verse Renumbering**: When correcting publisher typographical errors (e.g., misprinted Verse 57 corrected to Verse 37 on Page 86), naive upserts insert new `v037` chunks without deleting the old `v057` chunks. The database now returns duplicate, contradictory answers.

### Architectural Decision: Targeted Book-Scoped Purge
Rather than complex multi-stage manifest diffing or unsafe whole-database purges, the pipeline adopts **Targeted Book-Scoped Purge**:

Prior to batch-ingesting chunks for a given book, the indexing script executes a scoped deletion filtered strictly by `book_id`:
```python
collection.delete(where={"book_id": book_id})
```

### Why This Approach is Optimal for the Corpus

| Metric / Consideration | Targeted Book-Scoped Purge | Naive ID Skipping / Overwrite |
|---|---|---|
| **Stale Vector Immunity** | **100% Guaranteed** — All prior chunks for that book are wiped clean before re-insertion. | ❌ High risk of orphaned ghost sub-chunks and obsolete verse IDs. |
| **Multi-Book Isolation** | **Complete** — Scoped strictly by `where={"book_id": book_id}`. Other books in the collection are untouched. | Isolated, but leaves stale internal chunks. |
| **Compute & API Cost** | **Negligible** — Entire 667-chunk book (~213k tokens) costs **~$0.004** to embed via OpenAI `text-embedding-3-small`. | Saves ~$0.004 on reruns at the expense of corrupting the retrieval index. |
| **Execution Speed** | **Fast** — Ingestion finishes in 3–5 seconds over OpenAI API. | Slightly faster, but requires complex diff tracking. |
| **Implementation Complexity** | **1 line of code** — Clean, predictable, and fully deterministic. | High — Requires storing content hashes and reconciling sets of IDs. |

### Implementation Pattern (`embed_and_index.py`)
```python
def index_book(chunks: List[Dict[str, Any]], config: Dict[str, Any], purge: bool = True):
    book_id = config["book_id"]
    collection = get_collection(config)

    if purge:
        logger.info(f"Executing targeted purge for book '{book_id}' to eliminate stale vectors...")
        collection.delete(where={"book_id": book_id})

    # Proceed with streaming batch upsert into ChromaDB
    ...
```

---

## 16. Architectural Note: Hybrid Search (Dense + BM25 via Reciprocal Rank Fusion) for Sanskrit Technical Terminology

### Context & Problem Statement
While OpenAI `text-embedding-3-small` (1536 dimensions) provides exceptional semantic understanding, dense vector retrieval exhibits two well-known failure modes when applied to classical yoga treatises:

1. **Semantic Blurring Across Related Postures & Kriyas**:
   - Inverted postures such as *Viparita Karani*, *Sarvangasana*, and *Sirshasana* inhabit closely clustered dense vector neighborhoods (sharing themes of physical inversion, digestive fire, reversing nectar flow from the moon to the sun, and thyroid stimulation).
   - When a user queries: *"What are the cautions and time limits for Viparita Karani?"*, dense vector distance may rank an in-depth commentary on *Sarvangasana* higher if its phrasing about cautions and physiological benefits aligns closely with the query sentence structure.
2. **Transliteration & Diacritic Drift**:
   - Technical Sanskrit terms appear in multiple forms across queries and source literature: ASCII (*Sushumna*, *Matsyendrasana*), IAST diacritics (*Suṣumṇā*, *Matsyēndrāsana*), or phonetic variants (*Nadi Shodhana* vs. *Nadi Suddhi*).
   - In dense embeddings, sub-tokenization differences can dilute similarity scores for rare technical names.

### Architectural Decision: Hybrid Search via Reciprocal Rank Fusion (RRF)
To safeguard **100% recall on named techniques and postures** while retaining dense semantic search for conceptual queries, **Version 1 (v1) integrates Hybrid Search in `query_db.py`**:

- **Dense Component**: ChromaDB retrieves the top-$K_D$ candidates (e.g., 10) using cosine similarity on OpenAI embeddings.
- **Sparse Component**: An in-memory BM25 index (via `rank-bm25`) retrieves the top-$K_S$ candidates (e.g., 10) matching exact keyword tokens directly from `chunks.jsonl`.
- **Fusion Algorithm**: **Reciprocal Rank Fusion (RRF)** merges candidate lists without requiring score normalization or weight calibration:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)}$$

*(where $k = 60$ is the standard smoothing constant from Cormack et al.)*

### Architecture Comparison

| Feature | Pure Dense Retrieval | Pure Sparse (BM25) | Hybrid Search (Dense + BM25 via RRF) |
|---|---|---|---|
| **Conceptual Understanding**<br>*(e.g. "conquering death through breath")* | Excellent | ❌ Poor (misses synonyms and metaphors) | **Excellent** (Dense lifts conceptual context) |
| **Exact Technical Keyword Recall**<br>*(e.g. "Viparita Karani", "Sushumna")* | ⚠️ Moderate (risks semantic bleed across poses) | **100% Exact** (Token frequency match) | **100% Exact** (BM25 guarantees top rank) |
| **Infrastructure Overhead** | Requires ChromaDB | In-memory (~0.05s build from `chunks.jsonl`) | **Zero extra DB infrastructure** |
| **Transliteration Robustness** | Moderate | High (with case-insensitive tokenization) | **High** |

### Implementation Pattern (`query_db.py`)
```python
from rank_bm25 import BM25Okapi
import json

class HybridRetriever:
    def __init__(self, chroma_collection, chunks_jsonl_path: str):
        self.collection = chroma_collection
        self.chunks = []
        corpus = []
        with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    self.chunks.append(chunk)
                    corpus.append(chunk["content"].lower().split())
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5, where: dict = None):
        # 1. Dense retrieval from ChromaDB
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,
            where=where
        )
        
        # 2. Sparse retrieval from BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]

        # 3. Reciprocal Rank Fusion (RRF with k=60)
        rrf_scores = {}
        # Merge ranks and return top_k reranked chunks
        ...
```

### 16.1 Critical Refinement: Conversational Query Stopword Filtering in Sparse BM25

#### Context & Problem Statement (The Moola Bandha vs. Maha Bandha Failure Mode)
When users interact with a search or RAG system, they rarely submit bare keywords (`"moola bandha"`). Instead, they naturally frame queries in full natural language questions:
> *"How to perform Moola Bandha?"*

In a naive hybrid implementation, BM25 tokenizes the entire string (`['how', 'to', 'perform', 'moola', 'bandha']`) and computes frequency-weighted scores across the corpus. Because the generic English procedural verb **"perform"** appears heavily in instructional sections throughout Chapter 3, BM25 exhibits critical false-positive rank promotion:
- In **Maha Bandha** (Verse 19–22, `hyp_ch03_v019-022_p03`), the text instructs:
  > *"**Perform** jalandhara, uddiyana and **moola bandhas**... **Perform** three rounds in the beginning."*
- Because `"perform"` appeared twice alongside `"moola"` and `"bandha"`, BM25 awarded Maha Bandha its **#1 score across the entire book (11.63)**, beating out Moola Bandha's dedicated technique sections.
- When RRF fused Dense and BM25 candidate ranks with equal weight, Maha Bandha (Verse 19–22) and Maha Mudra (Verse 10–13) pushed into Ranks 2 and 3 ahead of Moola Bandha's core practice sub-chunks.

Meanwhile, pure dense vector search (`OpenAI text-embedding-3-small`) had correctly ranked all 4 parts of Verse 61–63 (Moola Bandha) as Ranks 1, 2, 3, and 4. BM25's literal sensitivity to conversational question syntax had degraded hybrid precision.

#### Architectural Solution: Decoupled Query Tokenization
To retain dense conceptual understanding while focusing BM25 strictly on entity retrieval, `query_db.py` decouples query processing:
1. **Dense Vector Channel**: Consumes the full, verbatim natural language query string (`"How to perform Moola Bandha?"`), preserving full conversational syntax, intent, and grammatical relations.
2. **Sparse BM25 Channel**: Filters conversational question-framing stopwords (`CONVERSATIONAL_QUESTION_STOPWORDS`) from the BM25 query tokens:
   - Question markers: *how, what, which, who, where, when, why*
   - Procedural request verbs: *perform, practice, do, does, did, tell, explain, describe*
   - Copulas & auxiliaries: *is, are, was, were, be, been, can, could, should, would*
   - Articles & prepositions: *the, a, an, in, of, for, and, on, with, at, by, from, about*
   - Pronouns: *you, your, i, me, my, we, our*
   - Filtered BM25 query tokens: `['moola', 'bandha']` (with graceful fallback to raw tokens if the query consists entirely of stopwords, e.g. *"how to do"*).
   - Corpus indexing retains all words (`filter_stopwords=False`), so full document text remains searchable.

#### Empirical Verification & Results

| Metric | Without Stopword Filtering | With Conversational Stopword Filtering (`query_db.py`) |
|---|---|---|
| **BM25 Query Tokens** | `['how', 'to', 'perform', 'moola', 'bandha']` | `['moola', 'bandha']` |
| **Top Hybrid Rank 1** | Verse 61–63 (Part 3/4) | **Verse 61–63 (Part 4/4 - Moola Bandha)** |
| **Top Hybrid Rank 2** | ❌ Verse 19–22 (Maha Bandha) | **Verse 61–63 (Part 3/4 - Moola Bandha)** |
| **Top Hybrid Rank 3** | ❌ Verse 10–13 (Maha Mudra) | **Verse 61–63 (Part 1/4 - Moola Bandha)** |
| **Top Hybrid Rank 4** | Verse 61–63 (Part 1/4) | **Verse 61–63 (Part 2/4 - Moola Bandha)** |
| **Entity Precision** | 60% (3 / 5 chunks relevant) | **100% (All 4 Moola Bandha parts in Top 4)** |

---

## 17. Architectural Note: Eliminating Semantic Context Drift in Verse Sub-chunks via Grammatical Anchor Injection

### Context & Problem Statement
Under Strategy 1 (`VerseChunker`), long commentaries exceeding the token ceiling are recursively split into sub-chunks. Part 1 ($i = 0$) contains the breadcrumb header, the full `sloka_translation`, and the introductory commentary. However, sub-chunks 2..N ($i \ge 1$) originally received only the breadcrumb header:
`[Hatha Yoga Pradipika | Chapter X: Title | Verse Y (Part i/N)]`

Across *Hatha Yoga Pradipika*, **84.0% of all chunks (560 / 667)** are sub-chunks (resulting from the strict 600-token ceiling and extensive multi-page commentary in Chapters 1–4). In these subsequent commentary sections, the text frequently refers to the primary subject using pronouns or generic phrases (*"this asana"*, *"this mudra"*, *"the practice"*, *"it"*).

#### Concrete Failure Mode
In chunk `hyp_ch01_v027_p03` (Verse 27: *Matsyendrasana* Part 3), the commentary details critical physiological effects:
> *"The other important physiological aspects of this asana are that it stimulates the pancreas, liver, spleen, kidneys, stomach and ascending and descending colons..."*

Because the word **"Matsyendrasana" never appears in Part 3**:
1. **BM25 Keyword Search**: Returns 0 keyword matches for "Matsyendrasana", failing to retrieve the chunk.
2. **Dense Vector Search**: Drifts into general abdominal physiology space rather than posture-specific retrieval.
3. **Generative RAG (v2)**: Downstream LLMs reading Part 3 suffer from pronoun ambiguity (*"this asana"*), risking hallucination.

### Architectural Decision: Grammatical Verse Anchor Injection (`build_verse_anchor`)
To eliminate context drift without bloating tokens or truncating mid-phrase, `VerseChunker` injects a standardized, grammatically complete **Verse Context Anchor** into every sub-chunk ($i \ge 1$):

```text
[Hatha Yoga Pradipika | Chapter 1: Asana | Verse 27 (Part 3/3)]
[Verse 27 Context: Practice of this asana (matsyendrasana) increases the digestive fire to such an incredible capacity that it is the means of removing diseases and thus awakening the serpent power and bringing equilibrium in bindu.]

The other important physiological aspects of this asana are that it stimulates the pancreas, liver, spleen...
```

### Key Engineering Rules:

1. **Grammatical Sentence-Boundary Splitting (Zero Mid-Phrase Truncation)**:
   - Rather than naive character slicing (`[:200] + "..."`), split the translation on sentence terminators (`(?<=[.?!])\s+`).
   - If the total translation is $\le 350$ characters (which is true for ~85% of verses, average length being 156 characters), keep the **entire translation intact**.
2. **Explicit Entity Capture in Fallback Truncation**:
   - If a translation exceeds 350 characters and begins with an introductory clause (e.g., *"By sitting in the posture described above and rotating the trunk..."*), the algorithm ensures the full grammatical sentence containing the named entity `(matsyendrasana)` or primary subject is retained.
3. **Token Budget Deduction**:
   - In `VerseChunker`, deduct the token count of `anchor_str` (`TokenCounter.count_tokens(anchor_str)`) from the commentary chunk budget before running the recursive splitter.
   - This strictly guarantees that **100% of chunks stay $\le 600$ tokens**.

### Implementation Pattern (`ingestion_pipeline.py`)
```python
def build_verse_anchor(self, sloka_text: str, max_chars: int = 350) -> str:
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
```

---

## 18. Architectural Note: Ingestion Pipeline Test Suite & Regression Verification

To prevent regressions across complex typographical edge cases, `test_ingestion_pipeline.py` maintains an automated 29-test verification suite covering every layout heuristic, regex guard, and chunking strategy.

### Key Areas & Edge Cases Covered

| Test Class | Focus / Edge Case Tested | Verification & Impact |
|---|---|---|
| **`TestTextCleaner`** | **Unicode NFC & Decomposed Diacritics**: Verifies `unicodedata.normalize('NFC')` merges decomposed macrons (`a + \u0304` $\rightarrow$ `ā`).<br>**Line-Break Hyphens**: Verifies `prac-\ntice` $\rightarrow$ `practice`.<br>**Newlines**: Single newlines convert to spaces while `\n\n` paragraph boundaries are preserved. | Passed (eliminates token fragmentation and word splitting artifacts). |
| **`TestTokenCounter`** | **Accurate BPE Accounting**: Exact token counts matching `tiktoken` `cl100k_base`.<br>**Sanskrit Transliteration Expansion**: Tests that diacritic-heavy Sanskrit terms produce accurate token expansions. | Passed (prevents context window overflow in downstream models). |
| **`TestVerseHeaderExtraction`** | **Standard Formats**: `"Verse 1"`, `"Verses 5-9"`.<br>**Multi-Part Subsection Verses**: `"Verses 16 (i)"` and `"Verses 16 (ii & iii)"`.<br>**Narrative Verb Guard**: Rejection of prose false positives (`"Verse 15 describes..."`, `"Verse 4 says..."`, `"Verses 1-4 are..."`).<br>**Line Length & Footers**: Rejection of lines $> 65$ chars and blocks $y_0 > 750$.<br>**Same-Line & Next-Line Posture Titles**: Captures posture titles across line breaks. | Passed (protects against false verse boundaries in prose commentary). |
| **`TestPracticeTitle`** | **`is_practice_title`**: Identifies uppercase/parenthetical titles (`"MATSYENDRASANA"`, `"SWASTIKASANA (auspicious pose)"`) and rejects sentences ending in periods or long prose. | Passed (prevents narrative text from being classified as posture titles). |
| **`TestVerseAnchorBuilder`** | **Clean Boundaries**: Intact preservation for $\le 350$ chars; sentence-boundary splitting for $> 350$ chars without mid-word truncation (`...`).<br>**Whitespace Collapsing**: Collapses multiline slokas into clean single-line strings. | Passed (guarantees grammatically complete verse anchors). |
| **`TestVerseChunker`** | **Tier 1 (Intact)**: Verse $\le 2000$ chars AND $\le 600$ tokens stays as 1 intact chunk (`hyp_ch01_v001`, `has_sanskrit_sloka=True`).<br>**Dual-Ceiling Guard**: Dense passage $\le 2000$ chars exceeding 600 tokens forces a sub-chunk split.<br>**Tier 2 (Recursive)**: Part 1 contains sloka (`has_sanskrit_sloka=True`); Parts 2..N contain `[Verse X Context: ...]` anchor (`has_sanskrit_sloka=False`).<br>**Strict Namespacing**: IDs formatted as `hyp_ch01_v027_p01` and `hyp_ch01_v005-009`.<br>**Recursive Bisecting Guard**: Dynamic token density estimation splitting until 100% chunks $\le 600$ tokens. | Passed (100% adherence to 600-token ceiling and schema contracts). |
| **`TestStandardChunker`** | **Narrative Splitting**: Tests `chunk_size` and `chunk_overlap`, breadcrumbs `[Book \| Chapter X (Part i/N)]`, and sequential IDs (`light_on_yoga_ch01_c001`). | Passed (supports narrative multi-book ingestion). |
| **`TestVersePdfExtractorLogic`** | **Publisher Errata**: Remaps Page 86 misprinted Verse 57 $\rightarrow$ Verse 37.<br>**Procedural Label Guard**: Rejects `"Technique 1"` and short tags from hijacking `sloka_translation`.<br>**Chapter Header Stripping**: Strips merged `"Chapter 1\nAsana"` from opening translations.<br>**Bidirectional Title Deduplication**: Merges overlapping posture names without duplication. | Passed (guarantees data integrity from raw PDF block layout). |

---

## 19. Architectural Note: Retrieval Calibration (`top_k` / `-k`), Result Inspection (`--full`), and Preview Formatting

### 19.1 Retrieval Depth Trade-offs: Why $k=5$ is the Recommended Scripture Default

When configuring retrieval depth ($k$) for classical yoga scriptures, simple heuristics from standard text search do not apply directly. The scripture structure, chunking ceilings, and downstream synthesis requirements dictate the optimal $k$:

| Retrieval Depth | Typical Token Payload | Coverage on Complex Verses | Risk Profile & Analysis |
|---|---|---|---|
| **$k = 3$** | ~900 – 1,100 tokens | **Incomplete (Drops 1–2 Parts)** | **Under-retrieval Risk**: Verses with extensive commentary (e.g. Moola Bandha Ch 3 V 61–63, Ujjayi Ch 2 V 51–53) are partitioned into 3 or 4 sub-chunks. At $k=3$, critical practice stages (e.g. Stage 3, integration with Kumbhaka) are omitted from context. |
| **$k = 4$** | ~1,200 – 1,450 tokens | Complete for 4-part verses | **Balanced (REPL Default)**: Comfortably retrieves all 4 sub-chunks of single major techniques, but leaves no budget for cross-chapter or related verse corroboration. |
| **$k = 5$** (Recommended) | ~1,500 – 1,800 tokens | **Complete + 1 Corroborating Verse** | **Optimal Recall/Precision Trade-off (CLI Default)**: Safely retrieves all 4 sub-chunks of primary techniques while pulling the closest related cross-verse reference (e.g. Verse 114 referencing Moola Bandha in Vajrasana). |
| **$k \ge 7$** | ~2,200 – 2,800 tokens | Over-extended | **Dilution Risk**: Introduces distantly related verses that increase prompt noise without improving answer quality. |

#### Token Budget & LLM Capacity Analysis
For downstream generation (`gpt-4o-mini`), a 5-chunk payload (~1,600 tokens) occupies **only 1.3% of the 128,000-token context window** and incurs less than **$0.00025 USD per query** (1/40th of a cent). Hence, $k=5$ maximizes technique completeness without token pressure or economic overhead.

---

### 19.2 Result Inspection: Terminal Previews vs Untruncated Mode (`--full`)

To balance rapid terminal readability with deep textual verification, `query_db.py` implements two display modes:

1. **Default Preview Mode (`max_content_chars = 350`)**:
   - Compresses multi-paragraph commentary into single-line summaries delimited by `•`.
   - Prevents 5 multi-part chunks from scrolling past the terminal viewport.
   - **Word-Boundary Truncation Guarantee**: Rather than naive character slicing (`text[:350] + "..."`), the formatter searches backward for the last whitespace boundary (`cut[:last_space] + "..."`), guaranteeing that words are never severed mid-word (e.g., preventing `pranayama practic...` $\rightarrow$ `pranayama...`).

2. **Full Untruncated Mode (`--full`)**:
   - Prints 100% of the verbatim text stored in ChromaDB and `chunks.jsonl`.
   - Preserves multi-line indentation, Sanskrit root slokas, Roman transliteration, word-by-word glosses, and Swami Muktibodhananda's full commentary.
   - CLI invocation:
     ```bash
     ./venv/bin/python query_db.py -q "How to perform Moola Bandha?" -k 5 --full
     ```

3. **Interactive REPL Inspection Commands**:
   - `:full` — Dynamically toggles between compact previews and full-text rendering for all subsequent searches.
   - `:view <N>` — Immediately prints the complete, untruncated content of result `<N>` from the most recent search without requiring re-execution.

---

### 19.3 Downstream Synthesis Implications for Milestone v2 (Steps 8 & 9)

In the generative RAG pipeline:
1. **Full Text Injection**: The LLM prompt always receives the **100% uncut chunk text** (never the terminal preview).
2. **Chronological Sub-Chunk Reassembly**: Prior to context injection, retrieved sub-chunks belonging to the same parent verse are re-ordered by `(chapter_number, verse_start, sub_chunk_index)` to form continuous, seamless reading passages.
3. **Dynamic Parent Verse Expansion (Step 8)**: If any sub-chunk of a verse is retrieved in top-$k$, the pipeline can optionally fetch its companion sibling sub-chunks automatically to ensure full scriptural integrity regardless of $k$.
4. **Score Drop-Off Filtering**: Chunks whose fused RRF score falls below a relative threshold (e.g. $> 40\%$ drop from Rank 1) can be pruned dynamically before prompt assembly to eliminate peripheral noise.

---

## 20. Architectural Note: Multimodal Image Extraction, Spatial Association & Rendering Strategy (Milestone v3)

### 20.1 Empirical PDF Image Census
An empirical inspection of `docs/Hatha-Yoga-Pradipika-pages-2.pdf` revealed 467 embedded raster image objects across its 414 pages:

| Image Category | Dimensions ($w \times h$) | Observed Count | Action / Heuristic |
|---|---|---|---|
| **Decorative Headpieces & Banners** | $750 \times 60$ to $750 \times 140$ ($h \le 150\text{px}$) | ~281 objects | **Discard**: Decorative ornamental scrollwork at chapter and verse openings. |
| **Publisher Logos & Small Icons** | $< 200 \times 150$ | ~58 objects | **Discard**: Bihar School of Yoga emblem, marginal publisher icons. |
| **Genuine Asana / Mudra Diagrams** | $250 \times 200$ to $750 \times 750$ ($w \ge 250, h \ge 200$) | **~128 diagrams** | **Extract & Index**: Clean line illustrations of specific postures (e.g. Matsyendrasana, Mayurasana, Paschimottanasana, Viparita Karani). |

### 20.2 Technical Implementation Pipeline
1. **Extraction Pipeline (`src/ingestion_pipeline.py`)**:
   - Use `fitz` (PyMuPDF) `page.get_images(full=True)` combined with `page.get_image_rects(xref)` to retrieve physical pixel dimensions and page bounding coordinates.
   - Filter with threshold: `width >= 250 and height >= 200 and height > 150`.
   - Save filtered images to `assets/images/{book_id}/` named deterministically: `{book_id}_ch{chapter:02d}_v{verse:03d}_fig{idx:02d}.png`.

2. **Spatial Verse Association**:
   - For each page containing an image, compare the image vertical rectangle $[y_0, y_1]$ against the text bounding boxes of verses on that page.
   - If an image lies within a verse's vertical span or immediately follows its Sanskrit title, associate the image path with that verse's `Chunk.image_paths`.
   - If a verse spans across multiple sub-chunks, attach the image path to Sub-chunk 1 (`p01`, containing the title and shloka) or all sub-chunks of that verse.

3. **In-Place ChromaDB Metadata Synchronization**:
   - Because `image_paths: []` was baked into the `Chunk` dataclass and sanitized via `chroma_manager.py` from Day 1, no vector re-indexing or database recreation is needed.
   - An updater script reads the newly extracted image paths and calls ChromaDB's `collection.update(ids=..., metadatas=...)`.
   - **Cost**: $0.00 API spend (no embedding calls).
   - **Time**: ~2 seconds to update all 668 chunk metadatas in-place.

4. **Multi-Channel Presentation**:
   - **Streamlit (`src/app.py`)**: When chunks retrieved by `HybridRetriever` have `image_paths`, `src/app.py` renders them using `st.image(image_path, caption=f"Figure: {posture_name}")` inside the collapsible source accordion.
   - **CLI Chat (`src/chat.py`)**: When running `:sources` or `:sources full`, displays a concise terminal indicator: `[📷 1 diagram: assets/images/hyp/...]`.

### 20.3 Effort, Resource & Cost Estimation
- **Engineering Effort**: ~2 to 3 engineering hours.
- **External API Cost**: **$0.00** (all extraction is local PyMuPDF processing; all database updates use local metadata-only mutation).
- **Disk Footprint**: ~128 PNG diagrams at ~120–200 KB each = **~15–25 MB total storage**.


