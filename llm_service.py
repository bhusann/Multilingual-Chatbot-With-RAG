"""
llm_service.py
==============
Search-agent LLM service for the multilingual voice assistant.

Flow (matches the requested behaviour):
    1. The user's speech (any language / Tanglish / Hinglish) is sent
       to the LLM.
    2. The LLM internally translates the intent into ENGLISH search
       queries.
    3. It calls the `web_search` tool (DuckDuckGo, no API key) as many
       times as it needs to cover every part of the question.
    4. The tool results are fed back to the LLM.
    5. The LLM produces the final answer in the SAME language the user
       spoke (Tanglish -> Tanglish, Tamil -> Tamil, Hindi -> Hindi, ...).

The model does NOT access the internet itself. This file is the bridge
between the model and the search engine.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY")
OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
LLM_MODEL = os.environ.get("OPENCODE_MODEL", "deepseek-v4-flash-free")

MAX_SEARCH_RESULTS = 5
MAX_TOKENS = 1200
TEMPERATURE = 0.3
MAX_SEARCH_ROUNDS = 6  # max number of tool-call rounds per user query
MAX_PARALLEL_SEARCHES = 4  # searches run concurrently within one round


# ============================================================
# BASE SYSTEM PROMPT
# ============================================================

BASE_SYSTEM_PROMPT = """
You are a helpful multilingual voice assistant for government schemes.

RULES:

1. Keep responses concise and conversational, normally 2-3 sentences.
2. Answer the user's actual question directly.
3. Do not mention these instructions.
"""


# ============================================================
# SEARCH ENGINE
# ============================================================

class SearchEngine:
    """
    Lightweight web search using DuckDuckGo (ddgs).

    Free, fast, and requires no API key. Returns a list of
    dicts with 'title', 'url' and 'snippet' for the top hits.
    """

    def search(self, query, max_results=MAX_SEARCH_RESULTS):
        """Search the web for `query` and return top results."""

        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                raw_results = list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                    )
                )

            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
                for item in raw_results
                if item.get("body")
            ]

            return results

        except Exception as e:

            print(f"⚠️ Web search failed: {e}")

            return []

    def format_results(self, results):
        """Format search results into a readable context block."""

        if not results:

            return (
                "No web search results were found "
                "for this query."
            )

        lines = []

        for i, result in enumerate(results, start=1):

            lines.append(
                f"{i}. {result['title']}\n"
                f"   URL: {result['url']}\n"
                f"   {result['snippet']}"
            )

        return "\n\n".join(lines)


# ============================================================
# TOOL SCHEMA
# ============================================================

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for information. Use this whenever you need "
            "fresh or detailed facts about government schemes, "
            "eligibility, age limits, application steps, amounts, etc. "
            "You may call it multiple times with different queries to "
            "gather everything the user asked about. Always include the "
            "current year in the query to get the latest data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query in ENGLISH. "
                        "Translate the user's intent to English first."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


# ============================================================
# LLM SERVICE
# ============================================================

class LLMService:
    """
    Search-agent that answers in the user's language.

    Pipeline:
        1. Build a system prompt (language rules + agent behaviour).
        2. Run an agent loop: the model decides when to call
           `web_search`, the service executes it and returns results.
        3. When the model has enough info it writes the final answer
           in the user's language.
    """

    def __init__(self):

        from datetime import datetime

        self.today_date = datetime.now().strftime("%d %B %Y")
        self.current_year = str(datetime.now().year)

        self.search_engine = SearchEngine()

        self.client = None
        self.model = LLM_MODEL

        # Build system prompt ONCE
        self.system_prompt = self._build_system_prompt()

        if OPENCODE_API_KEY:

            self.client = OpenAI(
                api_key=OPENCODE_API_KEY,
                base_url=OPENCODE_BASE_URL,
            )

            print(
                f"LLM Service ready "
                f"(OpenCode Zen / {LLM_MODEL})."
            )

        else:

            print(
                "\n❌ OPENCODE_API_KEY is not configured."
            )

            print(
                "Set it before running the program:"
            )

            print(
                "export OPENCODE_API_KEY='your-key-here'"
            )

    def configure_endpoint(
        self,
        base_url,
        model=None,
        api_key=None,
    ):
        """
        Re-point this service to a custom OpenAI-compatible
        endpoint (e.g. a local llama.cpp server).

        The default OpenCode Zen server stays untouched for
        any other scripts that share this module.
        """

        key = (
            api_key
            or OPENCODE_API_KEY
            or "local-test-key"
        )

        self.client = OpenAI(
            api_key=key,
            base_url=base_url,
        )

        if model:
            self.model = model

        print(
            "LLM Service re-pointed to custom endpoint: "
            f"{base_url}"
        )

    def _build_system_prompt(self):
        """Build the system prompt once at startup."""

        return f"""{BASE_SYSTEM_PROMPT}

TODAY'S DATE: {self.today_date}

SEARCH AGENT BEHAVIOUR
======================

You are a research agent with a `web_search` tool.

- The user may speak ANY language, including Tanglish / Hinglish
  (Indian languages written in Latin script).
- First UNDERSTAND the user's intent in their own language.
- Then TRANSLATE the intent into precise ENGLISH search queries.
- IMPORTANT: ALWAYS search for the LATEST data. Include the current
  year ({self.current_year}) in your search queries (for example
  "... {self.current_year} ..." or "... {self.current_year} latest ...").
  Prefer results dated in {self.current_year} or the most recent year
  available. Reject stale information if newer results contradict it.
- Call `web_search` as many times as needed to cover every part of
  the question. For multi-part questions, search each part separately.
- If the first results are thin, refine the query and search again.
- Once you have enough information, write the final answer.

FINAL ANSWER RULES
==================

- The user message includes a language tag indicating which language
  to respond in. Follow that instruction.
- If the user has explicitly asked to speak in a different language
  earlier in the conversation, follow that instruction instead.
- Keep the answer concise and conversational (2-4 short sentences)
  because it will be read aloud by a text-to-speech engine.
- Mention key facts like amounts, eligibility, age limits and
  how to apply when available.
- When citing amounts, dates or eligibility rules, use the LATEST
  figures from the search results. If a year is involved, state it
  clearly.
"""

    def _run_agent_loop(self, system_prompt, user_text, prior_turns, cancel_event=None, stream=False, rag_context=None):
        """
        Run the tool-calling loop until the model answers or the
        round limit is reached. Returns the final text reply.

        If stream=True, the FINAL answer is streamed to stdout.
        Tool-call rounds are never streamed.
        """

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Inject RAG context if available
        if rag_context:
            messages.append(
                {
                    "role": "system",
                    "content": rag_context,
                }
            )

        # Prior multi-turn context (user/assistant pairs)
        messages.extend(prior_turns)

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        for _ in range(MAX_SEARCH_ROUNDS):

            if (
                cancel_event
                and cancel_event.is_set()
            ):
                return ""

            # Tool-call rounds: non-streaming
            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model,
                    messages=messages,
                    tools=[WEB_SEARCH_TOOL],
                    tool_choice="auto",
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
            )

            message = response.choices[0].message

            tool_calls = getattr(
                message,
                "tool_calls",
                None,
            )

            # The model produced a final answer (no tool call)
            if not tool_calls:

                reply = (message.content or "").strip()

                if reply:
                    return reply

                # Empty response — nudge the model to retry instead
                # of giving up silently.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You returned an empty reply. Please answer the "
                            "user's question now. Use the web_search tool if "
                            "you need more information, then give your final "
                            "answer in the user's language."
                        ),
                    }
                )

                continue

            # Execute each requested search and feed results back
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            jobs = []

            for tc in tool_calls:

                try:

                    args = json.loads(
                        tc.function.arguments
                        or "{}"
                    )

                    query = args.get("query", user_text)

                except Exception:

                    query = user_text

                jobs.append((tc, query))

            print(
                f"🔎 Searching x{len(jobs)} (parallel): "
                + " | ".join(q for _, q in jobs)
            )

            # Run all searches for this round concurrently.
            # Slowest single search now bounds the round, not the sum.
            # Results are stashed and appended in the ORIGINAL tool-call
            # order so tool messages always line up with tool_calls.
            ordered_results = [None] * len(jobs)

            with ThreadPoolExecutor(
                max_workers=min(
                    len(jobs),
                    MAX_PARALLEL_SEARCHES,
                )
            ) as pool:

                future_by_index = {}

                for index, (tc, query) in enumerate(jobs):

                    future_by_index[index] = pool.submit(
                        self.search_engine.search,
                        query,
                    )

                for index, future in future_by_index.items():

                    try:

                        ordered_results[index] = future.result()

                    except Exception as e:

                        print(f"⚠️ Search failed: {e}")

                        ordered_results[index] = []

            for index, (tc, query) in enumerate(jobs):

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            self.search_engine.format_results(
                                ordered_results[index]
                            )
                        ),
                    }
                )

        # Round limit reached without a final answer
        return ""

    def _run_agent_loop_streaming(self, system_prompt, user_text, prior_turns, cancel_event=None, rag_context=None):
        """
        Run the tool-calling loop with streaming on the FINAL answer.
        Tool-call rounds are non-streaming. Returns an iterator of
        text chunks for the final answer.
        """

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Inject RAG context if available
        if rag_context:
            messages.append(
                {
                    "role": "system",
                    "content": rag_context,
                }
            )

        messages.extend(prior_turns)

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        for _ in range(MAX_SEARCH_ROUNDS):

            if (
                cancel_event
                and cancel_event.is_set()
            ):
                return

            # Stream the response to detect tool calls vs final answer
            response_stream = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model,
                    messages=messages,
                    tools=[WEB_SEARCH_TOOL],
                    tool_choice="auto",
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=True,
                )
            )

            # Accumulate the streamed response
            content_parts = []
            reasoning_parts = []
            tool_calls_data = {}  # index -> {id, name, arguments}
            has_tool_calls = False

            for chunk in response_stream:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Content
                c = getattr(delta, "content", None)
                if c:
                    content_parts.append(c)

                # Reasoning (for reasoning models)
                r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if r:
                    reasoning_parts.append(r)

                # Tool calls
                tc_delta = getattr(delta, "tool_calls", None)
                if tc_delta:
                    has_tool_calls = True
                    for tc in tc_delta:
                        idx = tc.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_data[idx]["id"] = tc.id
                        func = getattr(tc, "function", None)
                        if func:
                            if func.name:
                                tool_calls_data[idx]["name"] = func.name
                            if func.arguments:
                                tool_calls_data[idx]["arguments"] += func.arguments

            # If there are tool calls, execute them (non-streaming path)
            if has_tool_calls:

                # Build the assistant message with tool calls
                assistant_content = "".join(content_parts) or None
                assistant_tool_calls = []

                for idx in sorted(tool_calls_data.keys()):
                    tc = tool_calls_data[idx]
                    assistant_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    })

                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": assistant_tool_calls,
                    }
                )

                # Execute searches
                jobs = []

                for tc_raw in assistant_tool_calls:

                    try:
                        args = json.loads(
                            tc_raw["function"]["arguments"] or "{}"
                        )
                        query = args.get("query", user_text)
                    except Exception:
                        query = user_text

                    jobs.append((tc_raw, query))

                print(
                    f"🔎 Searching x{len(jobs)} (parallel): "
                    + " | ".join(q for _, q in jobs)
                )

                ordered_results = [None] * len(jobs)

                with ThreadPoolExecutor(
                    max_workers=min(
                        len(jobs),
                        MAX_PARALLEL_SEARCHES,
                    )
                ) as pool:

                    future_by_index = {}

                    for index, (tc_raw, query) in enumerate(jobs):
                        future_by_index[index] = pool.submit(
                            self.search_engine.search,
                            query,
                        )

                    for index, future in future_by_index.items():
                        try:
                            ordered_results[index] = future.result()
                        except Exception as e:
                            print(f"⚠️ Search failed: {e}")
                            ordered_results[index] = []

                for index, (tc_raw, query) in enumerate(jobs):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_raw["id"],
                            "content": (
                                self.search_engine.format_results(
                                    ordered_results[index]
                                )
                            ),
                        }
                    )

                continue

            # No tool calls — this is the final answer, stream it
            reply = "".join(content_parts).strip()

            # Handle reasoning models (Gemma4, DeepSeek, etc.)
            if not reply and reasoning_parts:
                raw_reasoning = "".join(reasoning_parts).strip()
                lines = [line.strip() for line in raw_reasoning.split("\n") if line.strip()]
                ans_lines = [l for l in lines if not (l.startswith('*') or l.startswith('Subject:') or l.startswith('Constraint:'))]
                if ans_lines:
                    reply = " ".join(ans_lines).strip()
                elif lines:
                    reply = lines[-1].strip('* ')

            if reply:
                yield reply
                return

            # Empty response — nudge retry
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You returned an empty reply. Please answer the "
                        "user's question now. Use the web_search tool if "
                        "you need more information, then give your final "
                        "answer in the user's language."
                    ),
                }
            )

        # Round limit reached
        yield ""

    def generate_response(
        self,
        user_text,
        chat_history,
        cancel_event=None,
        stream=False,
        rag_context=None,
    ):
        """
        Search the web (as many rounds as the model wants) and return
        the reply in the user's language.

        chat_history is mutated in place (user + assistant turns
        are appended) so multi-turn context is preserved.

        If cancel_event (a threading.Event) is set while searching,
        the loop stops early and returns "" (nothing is added to
        chat_history).

        If stream=True, returns a generator that yields chunks of the
        final answer (after all tool-call rounds).
        """

        if self.client is None:

            msg = (
                "Sorry, the assistant is not configured. "
                "Please set the OPENCODE_API_KEY."
            )

            if stream:
                yield msg
                return
            else:
                return msg

        # ----------------------------------------
        # 1. Use the pre-built system prompt
        # ----------------------------------------

        system_prompt = self.system_prompt

        # ----------------------------------------
        # 2. Prior turns (everything except index 0)
        # ----------------------------------------

        prior_turns = chat_history[1:] if chat_history else []

        # ----------------------------------------
        # 3. Run the search-agent loop
        # ----------------------------------------

        if stream:
            # Streaming mode: yield chunks from the final answer
            full_reply = ""

            for chunk in self._run_agent_loop_streaming(
                system_prompt=system_prompt,
                user_text=user_text,
                prior_turns=prior_turns,
                cancel_event=cancel_event,
                rag_context=rag_context,
            ):
                full_reply += chunk
                yield chunk

            # Interrupted mid-search: discard
            if (
                cancel_event
                and cancel_event.is_set()
            ):
                return

            if not full_reply:
                full_reply = (
                    "I couldn't find an answer "
                    "for that right now. Please try again."
                )
                yield full_reply

            # Remember turns
            chat_history.append(
                {"role": "user", "content": user_text}
            )
            chat_history.append(
                {"role": "assistant", "content": full_reply}
            )

        else:
            # Non-streaming mode
            try:

                reply = self._run_agent_loop(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    prior_turns=prior_turns,
                    cancel_event=cancel_event,
                    rag_context=rag_context,
                )

            except Exception as e:

                print(f"⚠️ LLM request failed: {e}")

                return (
                    "Sorry, I couldn't reach "
                    "the assistant service right now."
                )

            # Interrupted mid-search: discard
            if (
                cancel_event
                and cancel_event.is_set()
            ):
                return ""

            if not reply:

                reply = (
                    "I couldn't find an answer "
                    "for that right now. Please try again."
                )

            # Remember turns
            chat_history.append(
                {"role": "user", "content": user_text}
            )
            chat_history.append(
                {"role": "assistant", "content": reply}
            )

            return reply


# Shared instance used by the main chatbot loop
llm_service = LLMService()
