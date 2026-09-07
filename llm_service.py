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
import uuid
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY")
OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
LLM_MODEL = os.environ.get("OPENCODE_MODEL", "nemotron-3.5-lightning-free")

MAX_SEARCH_RESULTS = 5
MAX_TOKENS = 1200
TEMPERATURE = 0.3
MAX_SEARCH_ROUNDS = 6  # max number of tool-call rounds per user query
MAX_PARALLEL_SEARCHES = 4  # searches run concurrently within one round
MAX_HISTORY_TURNS = 8  # max conversation turns to keep in history


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

RAG_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "Search the uploaded government scheme documents. "
            "Use this to find specific details from the user's "
            "uploaded documents — eligibility, amounts, steps, "
            "deadlines, etc. The query MUST be in English keywords "
            "(the embedding model handles multilingual matching). "
            "You may call it multiple times with different keyword "
            "queries to cover different aspects of the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "English keyword query for document search. "
                        "Extract key terms from the user's question "
                        "and translate to English if needed."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

GRIEVANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "grievance_tool",
        "description": (
            "File a grievance complaint or check complaint status. "
            "Action 'create_ticket': register a complaint. You MUST "
            "have collected user_name, mobile, state, district and "
            "description from the user first (see the GRIEVANCE SKILL "
            "GUIDE in your instructions), and infer 'category' "
            "yourself from the complaint context. "
            "Action 'check_status': look up tickets using ticket_id "
            "or the mobile number given at filing time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create_ticket", "check_status"],
                    "description": (
                        "create_ticket to register a new complaint, "
                        "check_status to look up existing tickets."
                    ),
                },
                "user_name": {
                    "type": "string",
                    "description": "Complainant's full name (create only).",
                },
                "mobile": {
                    "type": "string",
                    "description": (
                        "10-digit mobile number. Required for create; "
                        "also accepted as lookup key for check_status."
                    ),
                },
                "state": {
                    "type": "string",
                    "description": "State name (create only).",
                },
                "district": {
                    "type": "string",
                    "description": "District name (create only).",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Inferred from complaint context (create only): "
                        "PACS, PMFBY, LOAN, SUBSIDY or OTHER."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "What happened, in the user's own words "
                        "(create only)."
                    ),
                },
                "ticket_id": {
                    "type": "string",
                    "description": (
                        "Ticket id like GRV-20260907-A3F9 "
                        "(check_status only, if known)."
                    ),
                },
            },
            "required": ["action"],
        },
    },
}


def load_grievance_skill():
    """
    Load the grievance skill guide for the system prompt.
    Best-effort: returns empty string if the file is missing.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "skills",
        "grievance_tool_usage_guide.txt",
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


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

        # True while pointed at OpenCode Zen (set False by
        # configure_endpoint when re-pointed to local/server).
        self._is_zen = False

        # Fallback Zen session when the caller has no chat
        # session id (stable for the process lifetime).
        self._zen_session = f"ses_{uuid.uuid4().hex[:24]}"

        # Build system prompt ONCE
        self.system_prompt = self._build_system_prompt()

        if OPENCODE_API_KEY:

            # Identify as the official OpenCode CLI (same headers
            # the `pi` agent sends). Zen gates free-tier models on
            # these — without them requests fail with
            # MissingSessionID ("free tier can only be used in
            # OpenCode").
            session_id = self._zen_session
            self.client = OpenAI(
                api_key=OPENCODE_API_KEY,
                base_url=OPENCODE_BASE_URL,
                default_headers={
                    "x-opencode-client": "pi",
                    "x-opencode-session": session_id,
                    "x-opencode-project": "global",
                    "x-opencode-request": f"msg_{uuid.uuid4().hex[:24]}",
                    "User-Agent": "opencode/latest/cli",
                },
            )
            self._is_zen = True

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

        # Re-pointed away from Zen (e.g. local llama.cpp) —
        # Zen session headers no longer apply.
        self._is_zen = False

        print(
            "LLM Service re-pointed to custom endpoint: "
            f"{base_url}"
        )

    # ------------------------------------------------
    # RAG search (called as a tool by the LLM)
    # ------------------------------------------------

    def rag_search(self, query):
        """
        Search uploaded documents using English keywords.
        The retriever handles embedding + Chroma lookup.
        Returns formatted context string for the LLM.
        """
        from rag.retriever import Retriever

        try:
            retriever = Retriever()
            context, sources = retriever.get_context(query)

            if not context:
                return (
                    "No relevant documents found for "
                    f"'{query}'. Use web_search instead."
                )

            n = len(sources)
            print(
                f"📚 RAG tool: {n} chunks for '{query}'"
            )
            return context

        except Exception as e:
            print(f"⚠️ RAG tool error: {e}")
            return (
                f"RAG search failed: {e}. "
                "Use web_search instead."
            )

    def _zen_headers(self, zen_session=None):
        """
        Per-request Zen headers for session affinity (cache hits).

        The session id follows OUR chat session: every turn of the
        same conversation carries the same x-opencode-session, so
        Zen routes it to the same provider with warm KV cache.
        x-opencode-request is fresh on every call.
        Returns None when not on Zen (local servers ignore it).
        """
        if not self._is_zen:
            return None

        sid = (zen_session or "").strip()
        if not sid.startswith("ses_"):
            sid = f"ses_{sid}" if sid else self._zen_session

        return {
            "x-opencode-client": "pi",
            "x-opencode-session": sid,
            "x-opencode-project": "global",
            "x-opencode-request": f"msg_{uuid.uuid4().hex[:24]}",
            "User-Agent": "opencode/latest/cli",
        }

    def grievance_tool_call(self, args):
        """
        Handle grievance_tool actions.

        create_ticket: one-shot registration. All fields must be
        present — if anything is missing, return what is missing
        so the LLM asks the user instead of filing a bad ticket.
        check_status: look up by ticket_id or mobile number.

        Returns a formatted string for the LLM.
        """
        from grievance_store import get_grievance_store

        action = (args.get("action") or "").strip()
        store = get_grievance_store()

        if action == "create_ticket":
            required = [
                "user_name", "mobile", "state",
                "district", "description",
            ]
            missing = [
                f for f in required
                if not (args.get(f) or "").strip()
            ]
            if missing:
                print(
                    "🎫 Grievance tool: create_ticket "
                    f"missing {missing}"
                )
                return (
                    "Cannot create the ticket yet — missing: "
                    + ", ".join(missing)
                    + ". Ask the user for these and call "
                    "create_ticket again with all fields."
                )

            digits = "".join(
                c for c in args["mobile"] if c.isdigit()
            )
            if len(digits) != 10:
                return (
                    "The mobile number must contain exactly "
                    "10 digits. Ask the user to confirm it and "
                    "call create_ticket again."
                )

            try:
                t = store.create_ticket(
                    user_name=args["user_name"],
                    mobile=args["mobile"],
                    state=args["state"],
                    district=args["district"],
                    category=args.get("category", "OTHER"),
                    description=args["description"],
                )
            except Exception as e:
                print(f"⚠️ Ticket creation failed: {e}")
                return f"Ticket creation failed: {e}."

            print(
                f"🎫 Grievance tool: created {t['ticket_id']}"
            )
            return (
                "Ticket registered successfully:\n"
                f"Ticket ID: {t['ticket_id']}\n"
                f"Name: {t['user_name']}\n"
                f"Category: {t['category']}\n"
                f"Status: {t['status']}\n"
                "Read the ticket ID back to the user and tell "
                "them to keep it safe for status checks."
            )

        elif action == "check_status":
            ticket_id = (args.get("ticket_id") or "").strip()
            mobile = (args.get("mobile") or "").strip()

            tickets = []
            if ticket_id:
                t = store.get_ticket(ticket_id)
                if t:
                    tickets = [t]
            elif mobile:
                tickets = store.find_by_mobile(mobile)
            else:
                return (
                    "Need a ticket_id or mobile number to check "
                    "status. Ask the user for one of them and call "
                    "check_status again."
                )

            n = len(tickets)
            print(
                f"🎫 Grievance tool: check_status "
                f"found {n} ticket(s)"
            )
            if not tickets:
                return (
                    "No tickets found for that lookup. Tell the "
                    "user no complaint is registered under it and "
                    "offer to file a new complaint."
                )

            lines = []
            for i, t in enumerate(tickets, start=1):
                lines.append(
                    f"{i}. Ticket {t['ticket_id']} "
                    f"({t['category']}, filed {t['created_at'][:10]}): "
                    f"status {t['status']} — "
                    f"{t['description'][:120]}"
                )
            return (
                f"Found {n} ticket(s):\n"
                + "\n".join(lines)
                + "\nReport each ticket's status in plain words "
                "in the user's language."
            )

        else:
            return (
                f"Unknown grievance action '{action}'. "
                "Use 'create_ticket' or 'check_status'."
            )

    def _dispatch_tool(self, tool_name, query, user_text, args=None):
        """Dispatch a single tool call to the right handler."""
        if tool_name == "web_search":
            return self.search_engine.search(query)
        elif tool_name == "rag_search":
            return self.rag_search(query)
        elif tool_name == "grievance_tool":
            return self.grievance_tool_call(args or {})
        else:
            print(f"⚠️ Unknown tool: {tool_name}")
            return []

    def _format_tool_result(self, tool_name, raw_result):
        """Format raw tool result for the LLM message."""
        if tool_name == "web_search":
            return self.search_engine.format_results(
                raw_result
            )
        elif tool_name == "rag_search":
            # rag_search already returns formatted string
            return raw_result
        elif tool_name == "grievance_tool":
            # grievance_tool already returns formatted string
            return raw_result
        else:
            return str(raw_result)

    def _build_system_prompt(self):
        """Build the system prompt once at startup."""

        return f"""{BASE_SYSTEM_PROMPT}

TODAY'S DATE: {self.today_date}

SEARCH AGENT BEHAVIOUR
======================

You are a research agent with THREE tools: `web_search`, `rag_search`
and `grievance_tool`.

TOOLS
-----
- `web_search(query)` — Search the internet for fresh facts.
  Query MUST be in ENGLISH. Include the current year ({self.current_year}).
- `rag_search(query)` — Search uploaded government scheme documents.
  Query MUST be in ENGLISH keywords (the embedding model handles
  multilingual matching). Use this for specific scheme details,
  eligibility, amounts, application steps from uploaded docs.
- `grievance_tool(action, ...)` — File a grievance complaint
  (`action="create_ticket"`) or check complaint status
  (`action="check_status"`). When the user wants to complain, FIRST
  read the GRIEVANCE SKILL GUIDE below and collect every required
  field before calling. When the user asks about their complaint,
  ask for their ticket_id or mobile number, then call check_status.

PARALLEL TOOL CALLS
-------------------
- You can call BOTH tools in the same round (parallel).
  For example: rag_search("maize subsidy eligibility") AND
  web_search("Tamil Nadu maize scheme 2026") simultaneously.
- Use rag_search FIRST to check uploaded docs, then web_search
  for latest补充 info. Or call both at once.

LANGUAGE RULES
--------------
- The user may speak ANY language, including Tanglish / Hinglish
  (Indian languages written in Latin script).
- First UNDERSTAND the user's intent in their own language.
- Then TRANSLATE the intent into precise ENGLISH search queries.
- The user message includes a language tag indicating which language
  to respond in. Follow that instruction.

SEARCH STRATEGY
---------------
- ALWAYS search for the LATEST data. Include the current year
  ({self.current_year}) in web_search queries.
- Call tools as many times as needed to cover every part of
  the question. For multi-part questions, search each part separately.
- If the first results are thin, refine the query and search again.
- Combine rag_search results (from docs) with web_search results
  (from internet) to give a complete answer.

FINAL ANSWER RULES
==================
- Keep the answer concise and conversational (2-4 short sentences)
  because it will be read aloud by a text-to-speech engine.
- Mention key facts like amounts, eligibility, age limits and
  how to apply when available.
- When citing amounts, dates or eligibility rules, use the LATEST
  figures from the search results. If a year is involved, state it
  clearly.

GRIEVANCE SKILL GUIDE
=====================
{load_grievance_skill()}
"""

    def _trim_history(self, chat_history):
        """
        Trim old turns from history while keeping system prompt.

        Keeps the last MAX_HISTORY_TURNS user turns with all their
        associated tool calls, tool results, and assistant replies.
        This preserves the exact message sequence for KV cache hits.
        """
        if len(chat_history) <= 2:
            return

        user_indices = [
            i for i, m in enumerate(chat_history)
            if m.get("role") == "user"
        ]

        if len(user_indices) <= MAX_HISTORY_TURNS:
            return

        keep_from = user_indices[-MAX_HISTORY_TURNS]
        del chat_history[1:keep_from]

    def _run_agent_loop(
        self, system_prompt, user_text, prior_turns,
        cancel_event=None, stream=False,
        rag_context=None, new_messages=None,
        zen_session=None,
    ):
        """
        Run the tool-calling loop until the model answers or the
        round limit is reached. Returns the final text reply.

        If new_messages is provided (a list), all new messages
        generated during this turn (user, tool calls, tool results,
        assistant) are appended to it for history tracking.
        """

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(prior_turns)

        user_msg = {"role": "user", "content": user_text}
        messages.append(user_msg)
        if new_messages is not None:
            new_messages.append(user_msg)

        for _ in range(MAX_SEARCH_ROUNDS):

            if (
                cancel_event
                and cancel_event.is_set()
            ):
                return ""

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model,
                    messages=messages,
                    tools=[WEB_SEARCH_TOOL, RAG_SEARCH_TOOL, GRIEVANCE_TOOL],
                    tool_choice="auto",
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    extra_headers=self._zen_headers(zen_session),
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
                    if new_messages is not None:
                        new_messages.append(
                            {"role": "assistant", "content": reply}
                        )
                    return reply

                # Empty response — nudge the model to retry.
                # Nudge messages are NOT saved to new_messages
                # so they don't pollute the persistent history.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You returned an empty reply. Please answer the "
                            "user's question now. Use web_search or rag_search "
                            "tools if you need more information, then give your "
                            "final answer in the user's language."
                        ),
                    }
                )

                continue

            # Execute each requested tool and feed results back
            asst_msg = {
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
            messages.append(asst_msg)
            if new_messages is not None:
                new_messages.append(asst_msg)

            jobs = []

            for tc in tool_calls:

                try:

                    args = json.loads(
                        tc.function.arguments
                        or "{}"
                    )

                    query = args.get("query", user_text)

                except Exception:

                    args = {}
                    query = user_text

                jobs.append((tc, query, args))

            tool_names = [tc.function.name for tc in tool_calls]
            print(
                f"\U0001f527 Tools x{len(jobs)} (parallel): "
                + " | ".join(
                    f"{name}({q})" for name, (_, q, _) in zip(tool_names, jobs)
                )
            )

            ordered_results = [None] * len(jobs)

            with ThreadPoolExecutor(
                max_workers=min(
                    len(jobs),
                    MAX_PARALLEL_SEARCHES,
                )
            ) as pool:

                future_by_index = {}

                for index, (tc, query, args) in enumerate(jobs):
                    future_by_index[index] = pool.submit(
                        self._dispatch_tool,
                        tc.function.name,
                        query,
                        user_text,
                        args,
                    )

                for index, future in future_by_index.items():

                    try:

                        ordered_results[index] = future.result()

                    except Exception as e:

                        print(f"\u26a0\ufe0f Tool failed: {e}")

                        ordered_results[index] = []

            for index, (tc, query, args) in enumerate(jobs):
                raw = ordered_results[index] or []
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": (
                        self._format_tool_result(
                            tc.function.name,
                            raw,
                        )
                    ),
                }
                messages.append(tool_msg)
                if new_messages is not None:
                    new_messages.append(tool_msg)

        # Round limit reached without a final answer
        return ""

    def _run_agent_loop_streaming(
        self, system_prompt, user_text, prior_turns,
        cancel_event=None, rag_context=None,
        new_messages=None, zen_session=None,
    ):
        """
        Run the tool-calling loop with streaming on the FINAL answer.
        Tool-call rounds are non-streaming. Returns an iterator of
        text chunks for the final answer.

        If new_messages is provided (a list), all new messages
        generated during this turn are appended to it for KV cache
        preservation.
        """

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        messages.extend(prior_turns)

        user_msg = {"role": "user", "content": user_text}
        messages.append(user_msg)
        if new_messages is not None:
            new_messages.append(user_msg)

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
                    tools=[WEB_SEARCH_TOOL, RAG_SEARCH_TOOL, GRIEVANCE_TOOL],
                    tool_choice="auto",
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=True,
                    extra_headers=self._zen_headers(zen_session),
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

                asst_msg = {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": assistant_tool_calls,
                }
                messages.append(asst_msg)
                if new_messages is not None:
                    new_messages.append(asst_msg)

                # Execute tools in parallel
                jobs = []

                for tc_raw in assistant_tool_calls:

                    try:
                        args = json.loads(
                            tc_raw["function"]["arguments"] or "{}"
                        )
                        query = args.get("query", user_text)
                    except Exception:
                        args = {}
                        query = user_text

                    jobs.append((tc_raw, query, args))

                tool_names = [tc["function"]["name"] for tc in assistant_tool_calls]
                print(
                    f"\U0001f527 Tools x{len(jobs)} (parallel): "
                    + " | ".join(
                        f"{name}({q})" for name, (_, q, _) in zip(tool_names, jobs)
                    )
                )

                ordered_results = [None] * len(jobs)

                with ThreadPoolExecutor(
                    max_workers=min(
                        len(jobs),
                        MAX_PARALLEL_SEARCHES,
                    )
                ) as pool:

                    future_by_index = {}

                    for index, (tc_raw, query, args) in enumerate(jobs):
                        future_by_index[index] = pool.submit(
                            self._dispatch_tool,
                            tc_raw["function"]["name"],
                            query,
                            user_text,
                            args,
                        )

                    for index, future in future_by_index.items():
                        try:
                            ordered_results[index] = future.result()
                        except Exception as e:
                            print(f"\u26a0\ufe0f Tool failed: {e}")
                            ordered_results[index] = []

                for index, (tc_raw, query, args) in enumerate(jobs):
                    raw = ordered_results[index] or []
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc_raw["id"],
                        "content": (
                            self._format_tool_result(
                                tc_raw["function"]["name"],
                                raw,
                            )
                        ),
                    }
                    messages.append(tool_msg)
                    if new_messages is not None:
                        new_messages.append(tool_msg)

                continue

            # No tool calls — this is the final answer
            reply = "".join(content_parts).strip()

            # Handle reasoning models (Gemma4, DeepSeek, etc.)
            if not reply and reasoning_parts:
                raw_reasoning = "".join(reasoning_parts).strip()
                rlines = [line.strip() for line in raw_reasoning.split("\n") if line.strip()]
                ans_lines = [l for l in rlines if not (l.startswith('*') or l.startswith('Subject:') or l.startswith('Constraint:'))]
                if ans_lines:
                    reply = " ".join(ans_lines).strip()
                elif rlines:
                    reply = rlines[-1].strip('* ')

            if reply:
                if new_messages is not None:
                    new_messages.append(
                        {"role": "assistant", "content": reply}
                    )
                yield reply
                return

            # Empty response — nudge retry (NOT saved to new_messages)
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
        zen_session=None,
    ):
        """
        Search the web/docs (tool-calling loop) and return reply in user's language.

        KV CACHE PRESERVATION:
        chat_history is mutated in place — all new messages from this turn
        (user, tool calls, tool results, assistant) are appended in exact order
        so that subsequent turns preserve the EXACT token prefix sequence.
        This ensures llama.cpp reuses cached KV states across conversation turns.

        If stream=True: returns a generator that yields text chunks.
        If stream=False: returns the complete reply string.
        """
        if stream:
            return self._generate_response_stream(
                user_text=user_text,
                chat_history=chat_history,
                cancel_event=cancel_event,
                rag_context=rag_context,
                zen_session=zen_session,
            )
        else:
            return self._generate_response_sync(
                user_text=user_text,
                chat_history=chat_history,
                cancel_event=cancel_event,
                rag_context=rag_context,
                zen_session=zen_session,
            )

    def _generate_response_sync(
        self,
        user_text,
        chat_history,
        cancel_event=None,
        rag_context=None,
        zen_session=None,
    ):
        """Synchronous non-streaming implementation that returns a string."""
        if self.client is None:
            return (
                "Sorry, the assistant is not configured. "
                "Please set the OPENCODE_API_KEY."
            )

        system_prompt = self.system_prompt
        if chat_history and chat_history[0].get("role") == "system":
            chat_history[0]["content"] = system_prompt

        prior_turns = chat_history[1:] if chat_history else []
        new_messages = []

        try:
            reply = self._run_agent_loop(
                system_prompt=system_prompt,
                user_text=user_text,
                prior_turns=prior_turns,
                cancel_event=cancel_event,
                rag_context=rag_context,
                new_messages=new_messages,
                zen_session=zen_session,
            )
        except Exception as e:
            print(f"⚠️ LLM request failed: {e}")
            return "Sorry, I couldn't reach the assistant service right now."

        if cancel_event and cancel_event.is_set():
            return ""

        if not reply:
            reply = (
                "I couldn't find an answer for that right now. "
                "Please try again."
            )
            new_messages.append({"role": "assistant", "content": reply})

        chat_history.extend(new_messages)
        self._trim_history(chat_history)
        return reply

    def _generate_response_stream(
        self,
        user_text,
        chat_history,
        cancel_event=None,
        rag_context=None,
        zen_session=None,
    ):
        """Streaming generator implementation that yields text chunks."""
        if self.client is None:
            yield (
                "Sorry, the assistant is not configured. "
                "Please set the OPENCODE_API_KEY."
            )
            return

        system_prompt = self.system_prompt
        if chat_history and chat_history[0].get("role") == "system":
            chat_history[0]["content"] = system_prompt

        prior_turns = chat_history[1:] if chat_history else []
        new_messages = []
        full_reply = ""

        for chunk in self._run_agent_loop_streaming(
            system_prompt=system_prompt,
            user_text=user_text,
            prior_turns=prior_turns,
            cancel_event=cancel_event,
            rag_context=rag_context,
            new_messages=new_messages,
            zen_session=zen_session,
        ):
            full_reply += chunk
            yield chunk

        if cancel_event and cancel_event.is_set():
            return

        if not full_reply:
            full_reply = (
                "I couldn't find an answer for that right now. "
                "Please try again."
            )
            yield full_reply
            new_messages.append({"role": "assistant", "content": full_reply})

        chat_history.extend(new_messages)
        self._trim_history(chat_history)


# Shared instance used by the main chatbot loop
llm_service = LLMService()
