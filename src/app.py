#!/usr/bin/env python3
"""
Streamlit Web Application: Hatha Yoga Scripture AI Scholar (Step 9 Web UI)
==========================================================================
Provides:
1. Browser-based conversational chat interface with streaming responses.
2. Multi-turn conversation memory with pronoun resolution.
3. Collapsible source attribution accordion showing exact retrieved scripture
   excerpts, verse metadata, and verified citations.
4. Sidebar controls for chapter filtering, top-k retrieval depth, temperature,
   and quick-start starter questions.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from query_db import HybridRetriever, SearchResult
from prompt_engine import (
    SYSTEM_PROMPT_TEMPLATE,
    build_rag_messages,
    extract_citations,
    align_query_terminology
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Hatha Yoga Scripture AI Scholar",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #B85D19;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #FFF3CD;
        color: #856404;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.4rem;
        border: 1px solid #FFEEBA;
    }
    .source-card {
        background-color: #F8F9FA;
        border-left: 4px solid #B85D19;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Cached Resource Initialization
# ============================================================================

@st.cache_resource(show_spinner="Initializing Scripture Retrieval Engine & Vector DB...")
def get_retriever():
    """Initializes and caches HybridRetriever across browser sessions."""
    return HybridRetriever(config_or_path="configs/hatha_yoga_pradipika.yaml")


@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ `OPENAI_API_KEY` not found in environment or `.env` file. Please configure it to enable chat.")
        st.stop()
    return OpenAI(api_key=api_key)


retriever = get_retriever()
client = get_openai_client()


# ============================================================================
# Session State Initialization
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}


# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.markdown("### 🕉️ Scripture Controls")
    st.markdown("**Book**: *Hatha Yoga Pradipika*  \n*(Yogi Svatmarama / Swami Muktibodhananda)*")

    # Chapter filter
    chapter_options = {
        "All Chapters (Holistic)": None,
        "Chapter 1: Asana (Postures)": 1,
        "Chapter 2: Shatkarma & Pranayama": 2,
        "Chapter 3: Mudra & Bandha (Seals/Locks)": 3,
        "Chapter 4: Samadhi (Liberation)": 4
    }
    selected_ch_label = st.selectbox("Filter by Chapter:", list(chapter_options.keys()))
    selected_chapter = chapter_options[selected_ch_label]

    # Model & Retrieval settings
    st.markdown("---")
    st.markdown("### ⚙️ Generation Settings")
    model_choice = st.selectbox("OpenAI Model:", ["gpt-4o-mini", "gpt-4o"], index=0)
    top_k = st.slider("Retrieval Depth (Chunks $k$):", min_value=1, max_value=8, value=4)
    temperature = st.slider("Creativity (Temperature):", min_value=0.0, max_value=0.6, value=0.2, step=0.05)

    st.markdown("---")
    st.markdown("### 💡 Quick-Start Questions")
    quick_questions = [
        "How to perform Moola Bandha?",
        "What are the six causes of failure in yoga?",
        "Explain the practice and stages of Bhastrika Pranayama.",
        "What is Mithahara and dietary rules for yogis?",
        "How does Viparita Karani mudra conquer death?"
    ]
    for q in quick_questions:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state.pending_query = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_history = {}
        st.rerun()


# ============================================================================
# Main Header
# ============================================================================

st.markdown('<div class="main-title">🕉️ Hatha Yoga Scripture AI Scholar</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Faithful, citation-grounded conversational RAG based on the <i>Hatha Yoga Pradipika</i> '
    '(Bihar School of Yoga edition).</div>',
    unsafe_allow_html=True
)


# ============================================================================
# Render Existing Conversation
# ============================================================================

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # If assistant, display associated sources accordion if available
        if msg["role"] == "assistant" and idx in st.session_state.sources_history:
            sources_data = st.session_state.sources_history[idx]
            sources = sources_data.get("sources", [])
            citations = sources_data.get("citations", [])

            with st.expander(f"🔎 Scriptural Sources & Verified Citations ({len(sources)} excerpts retrieved)"):
                if citations:
                    st.markdown("**Verified Citations in Response:**")
                    badges_html = " ".join([f'<span class="citation-badge">✓ {c["raw"]}</span>' for c in citations])
                    st.markdown(badges_html, unsafe_allow_html=True)
                    st.markdown("")

                for s_idx, s in enumerate(sources, start=1):
                    meta = s.metadata
                    ch_num = meta.get("chapter_number", "?")
                    ch_title = meta.get("chapter_title", "Unknown")
                    v_label = meta.get("verse_label") or "N/A"
                    p_start = meta.get("page_start", "?")
                    p_end = meta.get("page_end", "?")

                    st.markdown(f"**Excerpt {s_idx}: [{s.chunk_id}]** — Chapter {ch_num} ({ch_title}), Verse {v_label} (pp. {p_start}-{p_end})")
                    st.text(s.content)
                    st.markdown("---")


# ============================================================================
# Chat Input & Streaming Execution
# ============================================================================

# Handle button-clicked pending queries or manual user chat input
prompt = st.chat_input("Ask a question about asana, pranayama, mudra, or philosophy...")
if "pending_query" in st.session_state and st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

if prompt:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Contextual query enrichment for pronouns if applicable
    search_query = prompt
    if len(st.session_state.messages) > 1:
        prev_user_msg = next((m["content"] for m in reversed(st.session_state.messages[:-1]) if m["role"] == "user"), None)
        if prev_user_msg:
            aligned_terms = align_query_terminology(prev_user_msg)
            if aligned_terms and any(p in prompt.lower().split() for p in ["it", "its", "this", "that", "these", "practice", "technique"]):
                search_query = f"{prompt} ({', '.join(aligned_terms)})"

    # 3. Retrieve Scripture Chunks
    with st.spinner("Consulting classical scripture index..."):
        retrieved_sources = retriever.search(
            query=search_query,
            top_k=top_k,
            chapter=selected_chapter
        )

    # 4. Construct Prompt Messages
    recent_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ][-6:]

    messages = build_rag_messages(
        query=prompt,
        retrieved_chunks=retrieved_sources,
        chat_history=recent_history
    )

    # 5. Stream Assistant Response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            temperature=temperature,
            stream=True
        )

        def stream_generator():
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content

        full_response = st.write_stream(stream_generator)

        # 6. Extract and verify citations
        citations = extract_citations(full_response)

        # 7. Render source attribution accordion
        with st.expander(f"🔎 Scriptural Sources & Verified Citations ({len(retrieved_sources)} excerpts retrieved)"):
            if citations:
                st.markdown("**Verified Citations in Response:**")
                badges_html = " ".join([f'<span class="citation-badge">✓ {c["raw"]}</span>' for c in citations])
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown("")

            for s_idx, s in enumerate(retrieved_sources, start=1):
                meta = s.metadata
                ch_num = meta.get("chapter_number", "?")
                ch_title = meta.get("chapter_title", "Unknown")
                v_label = meta.get("verse_label") or "N/A"
                p_start = meta.get("page_start", "?")
                p_end = meta.get("page_end", "?")

                st.markdown(f"**Excerpt {s_idx}: [{s.chunk_id}]** — Chapter {ch_num} ({ch_title}), Verse {v_label} (pp. {p_start}-{p_end})")
                st.text(s.content)
                st.markdown("---")

    # 8. Record Assistant Turn in Session State
    assistant_idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.sources_history[assistant_idx] = {
        "sources": retrieved_sources,
        "citations": citations
    }
