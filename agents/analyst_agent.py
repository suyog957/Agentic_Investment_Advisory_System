# Research agent. Given a task from the Advisor, it loops through tool calls
# (knowledge base first, web search if needed) until it has enough to write
# a structured report. Emits events so the UI can show what it's doing.

import json
from typing import Callable
from groq import Groq, BadRequestError
from config import GROQ_API_KEY, MODEL_TOOL as MODEL, MAX_ANALYST_TOOL_ROUNDS
from schemas import AnalystTask, AnalystReport
from tools.knowledge_retrieval import search_knowledge_base, KNOWLEDGE_BASE_TOOL_SCHEMA
from tools.web_search import web_search, WEB_SEARCH_TOOL_SCHEMA
from tools.tool_call_fallback import extract_failed_generation, parse_failed_generation, make_synthetic_tool_messages
from events import Event


_SYSTEM_PROMPT = """
You are a professional investment research analyst. You have access to:
1. An internal knowledge base with risk profiles, ETF guides, asset allocation strategies, and investing principles.
2. Web search for current market data, news, and up-to-date financial information.

Your job:
- Receive a research task from the Advisor.
- Use your tools to gather relevant information (search knowledge base first, then web if needed).
- Synthesize findings into a structured research memo.

Output format:
Always end with a JSON block enclosed in <report>...</report> tags with this exact structure:
{
  "market_summary": "Brief current market context relevant to the client's situation",
  "recommendations": ["Specific actionable recommendation 1", "Recommendation 2"],
  "risks": ["Key risk 1", "Key risk 2"],
  "etf_suggestions": [
    {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "rationale": "..."}
  ],
  "allocation_breakdown": {"US Equities": 45, "International Equities": 15, "US Bonds": 30, "REITs": 5, "Cash": 5},
  "sources": ["Knowledge base: risk_profiles.md", "Web: ..."]
}

Guidelines:
- Be specific and data-driven.
- Always respect client constraints (e.g., no crypto, ETF preference).
- Reference specific ETF tickers when recommending funds.
- Use the knowledge base for strategy/principles, web for current market data.
""".strip()

_TOOLS = [KNOWLEDGE_BASE_TOOL_SCHEMA, WEB_SEARCH_TOOL_SCHEMA]


class AnalystAgent:
    """Research agent with Groq tool-use agentic loop."""

    def __init__(self):
        self._client = Groq(api_key=GROQ_API_KEY)

    def research(
        self,
        task: AnalystTask,
        on_event: Callable[[Event], None] | None = None,
    ) -> AnalystReport:
        """
        Execute research task using tools, return structured AnalystReport.
        Emits tool_call / tool_result events via on_event if provided.
        """
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._format_task(task)},
        ]

        last_content = ""

        for _ in range(MAX_ANALYST_TOOL_ROUNDS):
            try:
                response = self._client.chat.completions.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=messages,
                    tools=_TOOLS,
                    tool_choice="auto",
                )
            except BadRequestError as e:
                parsed = parse_failed_generation(e)
                if parsed is None:
                    # Plain text in failed_generation — treat as final analyst output.
                    plain_text = extract_failed_generation(e).strip()
                    if plain_text:
                        last_content = plain_text
                        break
                    raise
                tool_name, tool_args = parsed
                self._emit_tool_call(on_event, tool_name, tool_args)
                result = self._execute_tool(tool_name, tool_args)
                result = self._truncate(result)
                self._emit_tool_result(on_event, tool_name, result)
                asst_msg, tool_msg = make_synthetic_tool_messages(tool_name, tool_args, result)
                messages.append(asst_msg)
                messages.append(tool_msg)
                continue

            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            messages.append(self._message_to_dict(message))

            if finish_reason == "stop":
                last_content = message.content or ""
                break

            if finish_reason == "tool_calls" and message.tool_calls:
                for tool_call in message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    self._emit_tool_call(on_event, tool_call.function.name, args)
                    result = self._execute_tool(tool_call.function.name, args)
                    if len(result) > 3000:
                        result = result[:3000] + "\n... [truncated]"
                    self._emit_tool_result(on_event, tool_call.function.name, result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            else:
                last_content = message.content or ""
                break

        report = self._parse_report(last_content, task)
        if on_event:
            on_event(Event("analyst_report", {"report": report}))
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_tool_call(on_event, tool_name: str, args: dict) -> None:
        if on_event:
            on_event(Event("tool_call", {"tool": tool_name, "args": args}))

    @staticmethod
    def _emit_tool_result(on_event, tool_name: str, result: str) -> None:
        if on_event:
            on_event(Event("tool_result", {"tool": tool_name, "result": result}))

    @staticmethod
    def _truncate(text: str, limit: int = 3000) -> str:
        return text[:limit] + "\n... [truncated]" if len(text) > limit else text

    def _format_task(self, task: AnalystTask) -> str:
        lines = [f"**Research Objective:** {task.objective}", "", "**Research Questions:**"]
        for i, q in enumerate(task.research_questions, 1):
            lines.append(f"{i}. {q}")
        if task.client_profile_summary:
            lines += ["", f"**Client Context:** {task.client_profile_summary}"]
        lines += ["", "Please research this thoroughly using your tools, then provide your structured report."]
        return "\n".join(lines)

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "search_knowledge_base":
            return search_knowledge_base(
                query=tool_input["query"],
                n_results=tool_input.get("n_results", 5),
            )
        elif tool_name == "web_search":
            return web_search(
                query=tool_input["query"],
                num_results=tool_input.get("num_results", 5),
            )
        return f"Unknown tool: {tool_name}"

    @staticmethod
    def _message_to_dict(message) -> dict:
        d: dict = {"role": message.role, "content": message.content or ""}
        if message.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        return d

    def _parse_report(self, text: str, task: AnalystTask) -> AnalystReport:
        try:
            start = text.find("<report>")
            end = text.find("</report>")
            if start != -1 and end != -1:
                data = json.loads(text[start + len("<report>"):end].strip())
                return AnalystReport(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return AnalystReport(
            market_summary=text[:500] if text else "Research completed.",
            recommendations=["Please see the full analyst notes for detailed recommendations."],
            risks=["Unable to parse structured report — see raw output."],
            sources=[f"Research for: {task.objective}"],
        )
