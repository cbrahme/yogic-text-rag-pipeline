"""
Hath_Yog_RAG Core Source Package
================================
Provides:
- Ingestion & extraction: `ingestion_pipeline`
- ChromaDB management: `chroma_manager`
- Embedding & vector indexing: `embed_and_index`
- Hybrid retrieval: `query_db`
- Prompt architecture & guardrails: `prompt_engine`
- Conversational chat assistant: `chat`
- Streamlit web application: `app`
"""

import os
import sys

# Ensure src directory is in sys.path for direct submodule imports
_src_dir = os.path.abspath(os.path.dirname(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
