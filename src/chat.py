#!/usr/bin/env python3
"""
Interactive Conversational Scripture Assistant CLI (Step 9)
===========================================================
Provides:
1. `ChatSession`: Multi-turn conversational manager integrating:
   - `HybridRetriever` (dense vector + sparse BM25 via RRF)
   - `prompt_engine` (scholar persona, citation contracts, safety guardrails,
     canonical scripture sorting, terminology alignment)
   - OpenAI streaming API (`gpt-4o-mini`)
2. Contextual follow-up query enrichment for conversational pronouns ("its", "these").
3. Source attribution inspector (`:sources`) displaying exact verse references,
   ChromaDB scores, and page numbers.
4. Terminal REPL loop with streaming output and command switches.
"""

import os
import sys
import re
import argparse
from typing import Dict, Any, List, Optional, Tuple, Iterator

from dotenv import load_dotenv
from openai import OpenAI

from query_db import HybridRetriever, SearchResult, format_search_results
from prompt_engine import (
    SYSTEM_PROMPT_TEMPLATE,
    build_rag_messages,
    extract_citations,
    align_query_terminology
)

load_dotenv()


class ChatSession:
    """
    Manages multi-turn conversational interaction with the Hatha Yoga
    scripture RAG system.
    """

    def __init__(
        self,
        config_path: str = "configs/hatha_yoga_pradipika.yaml",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        top_k: int = 4,
        max_history_turns: int = 6,
        api_key: Optional[str] = None
    ):
        self.config_path = config_path
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.max_history_turns = max_history_turns

        # Initialize API client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your environment or .env file."
            )
        self.client = OpenAI(api_key=self.api_key)

        # Initialize retriever
        self.retriever = HybridRetriever(config_or_path=self.config_path)

        # Session state
        self.history: List[Dict[str, str]] = []
        self.chapter_filter: Optional[int] = None
        self.last_query: Optional[str] = None
        self.last_sources: List[SearchResult] = []
        self.last_citations: List[Dict[str, str]] = []

    def _resolve_conversational_query(self, query: str) -> str:
        """
        Enriches short pronoun-dependent follow-up queries
        (e.g., 'What are its benefits?', 'How to do it?')
        with context from the previous turn so hybrid search hits the right entity.
        """
        if not self.last_query:
            return query

        pronoun_pattern = r'\b(it|its|this|that|these|those|the practice|the technique)\b'
        words = query.strip().split()
        if len(words) <= 7 and re.search(pronoun_pattern, query, re.IGNORECASE):
            # Extract key topic words from previous query
            prev_terms = align_query_terminology(self.last_query)
            if prev_terms:
                return f"{query} ({', '.join(prev_terms)})"
            # Fallback: append previous query as context
            clean_prev = re.sub(r'^(what is|how to perform|how to do|tell me about)\s*', '', self.last_query, flags=re.IGNORECASE).strip(' ?.')
            if clean_prev:
                return f"{query} ({clean_prev})"

        return query

    def stream_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        chapter: Optional[int] = None
    ) -> Iterator[str]:
        """
        Retrieves context, constructs prompt payload, and yields streaming
        tokens from the LLM. Updates session history upon completion.
        """
        k = top_k or self.top_k
        ch = chapter if chapter is not None else self.chapter_filter

        # 1. Resolve query dependencies
        search_query = self._resolve_conversational_query(query)

        # 2. Retrieve grounded scriptural context
        sources = self.retriever.search(
            query=search_query,
            top_k=k,
            chapter=ch
        )
        self.last_sources = sources
        self.last_query = query

        # 3. Assemble prompt payload with sliding-window history
        recent_history = self.history[-self.max_history_turns:] if self.history else None
        messages = build_rag_messages(
            query=query,
            retrieved_chunks=sources,
            chat_history=recent_history
        )

        # 4. Stream response from OpenAI
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )

        full_content: List[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_content.append(delta)
                yield delta

        # 5. Record session history and extract citations
        assistant_reply = "".join(full_content)
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": assistant_reply})
        self.last_citations = extract_citations(assistant_reply)

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        chapter: Optional[int] = None
    ) -> Tuple[str, List[SearchResult], List[Dict[str, str]]]:
        """
        Synchronous non-streaming query execution. Returns (answer, sources, citations).
        """
        tokens = list(self.stream_answer(query=query, top_k=top_k, chapter=chapter))
        answer = "".join(tokens)
        return answer, self.last_sources, self.last_citations

    def clear_history(self) -> None:
        """Resets conversational history buffer."""
        self.history = []
        self.last_query = None
        self.last_sources = []
        self.last_citations = []


# ============================================================================
# Interactive Terminal REPL Loop
# ============================================================================

def run_chat_loop(session: ChatSession) -> None:
    """Runs interactive terminal conversational loop with streaming responses."""
    print("\n" + "=" * 65)
    print("   🕉️  HATHA YOGA SCRIPTURE CONVERSATIONAL ASSISTANT")
    print("=" * 65)
    print(f"  Model:       {session.model} (temp={session.temperature})")
    print(f"  Retrieval:   Top {session.top_k} Chunks (Hybrid RRF)")
    print("  Commands:    'exit' or 'quit' to terminate")
    print("               ':sources' to inspect chunks retrieved for last answer")
    print("               ':sources full' to view uncut source texts")
    print("               ':clear' to reset conversation memory")
    print("               ':k N' to change retrieval depth (e.g. ':k 5')")
    print("               ':ch N' to filter by Chapter (e.g. ':ch 2', ':all' to clear)")
    print("               ':model <name>' to switch model")
    print("               ':history' to view conversation turns")
    print("=" * 65 + "\n")

    while True:
        try:
            status_parts = [session.model]
            if session.chapter_filter:
                status_parts.append(f"Ch {session.chapter_filter}")
            status_parts.append(f"k={session.top_k}")
            if session.history:
                status_parts.append(f"{len(session.history)//2} turns")
            status_tag = f"[{' | '.join(status_parts)}]"

            user_input = input(f"{status_tag} Ask > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n[*] Ending session. Om Shanti! Namaste.\n")
                break

            # Handle interactive commands
            if user_input.startswith(":"):
                cmd = user_input.lower().split()
                if cmd[0] == ":clear":
                    session.clear_history()
                    print("[*] Conversation memory cleared.\n")
                elif cmd[0] == ":sources":
                    if not session.last_sources:
                        print("[!] No sources available yet. Ask a question first.\n")
                    else:
                        show_full = (len(cmd) > 1 and cmd[1] == "full")
                        print(format_search_results(session.last_sources, full_content=show_full))
                        if session.last_citations:
                            print("[*] Verified Citations in Response:")
                            for cit in session.last_citations:
                                print(f"    ✓ {cit['raw']} (Chapter {cit['chapter']}, Verse {cit['verse']}, pp. {cit['pages']})")
                            print()
                elif cmd[0] == ":k" and len(cmd) > 1 and cmd[1].isdigit():
                    session.top_k = int(cmd[1])
                    print(f"[*] Retrieval depth set to: {session.top_k}\n")
                elif cmd[0] == ":ch" and len(cmd) > 1 and cmd[1].isdigit():
                    session.chapter_filter = int(cmd[1])
                    print(f"[*] Filter set to Chapter {session.chapter_filter}\n")
                elif cmd[0] == ":all":
                    session.chapter_filter = None
                    print("[*] Cleared chapter filter.\n")
                elif cmd[0] == ":model" and len(cmd) > 1:
                    session.model = cmd[1]
                    print(f"[*] Switched model to: {session.model}\n")
                elif cmd[0] == ":history":
                    if not session.history:
                        print("[*] Conversation history is empty.\n")
                    else:
                        print("\n" + "-" * 50)
                        print(f"  CONVERSATION HISTORY ({len(session.history)//2} turns)")
                        print("-" * 50)
                        for turn in session.history:
                            role = turn["role"].upper()
                            preview = turn["content"][:120].replace("\n", " ")
                            print(f"  {role}: {preview}...")
                        print("-" * 50 + "\n")
                elif cmd[0] in [":help", ":h"]:
                    print("\nCommands:")
                    print("  :sources        Show retrieved scripture excerpts")
                    print("  :sources full   Show complete uncut chunk texts")
                    print("  :clear          Reset conversation memory")
                    print("  :k <N>          Set top-k retrieval depth")
                    print("  :ch <N>         Filter search to Chapter N")
                    print("  :all            Remove chapter filter")
                    print("  :history        Show past turns")
                    print("  exit / quit     Exit assistant\n")
                else:
                    print("[!] Unknown command. Type ':help' for options.\n")
                continue

            # Stream generation
            print("\n" + "-" * 60)
            for token in session.stream_answer(user_input):
                sys.stdout.write(token)
                sys.stdout.flush()
            print("\n" + "-" * 60 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n[*] Session interrupted. Om Shanti! Namaste.\n")
            break
        except Exception as e:
            print(f"\n[!] Error during execution: {e}\n")


# ============================================================================
# Main CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Conversational Hatha Yoga Scripture Assistant")
    parser.add_argument("--config", default="configs/hatha_yoga_pradipika.yaml", help="Path to YAML config")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--top-k", "-k", type=int, default=4, help="Retrieval depth (default: 4)")
    parser.add_argument("--chapter", "-c", type=int, default=None, help="Filter by chapter number (1-4)")
    parser.add_argument("--query", "-q", default=None, help="Single question mode (non-interactive)")
    parser.add_argument("--sources", action="store_true", help="Print retrieved sources after answer")
    args = parser.parse_args()

    session = ChatSession(
        config_path=args.config,
        model=args.model,
        temperature=args.temperature,
        top_k=args.top_k
    )
    if args.chapter is not None:
        session.chapter_filter = args.chapter

    if args.query:
        # Single-question non-interactive mode
        print(f"\n[Question]: {args.query}\n")
        print("-" * 60)
        for token in session.stream_answer(args.query):
            sys.stdout.write(token)
            sys.stdout.flush()
        print("\n" + "-" * 60 + "\n")

        if args.sources and session.last_sources:
            print(format_search_results(session.last_sources))
            if session.last_citations:
                print("[*] Verified Citations in Response:")
                for cit in session.last_citations:
                    print(f"    ✓ {cit['raw']} (Chapter {cit['chapter']}, Verse {cit['verse']}, pp. {cit['pages']})")
                print()
        return

    # Interactive REPL mode
    run_chat_loop(session)


if __name__ == "__main__":
    main()
