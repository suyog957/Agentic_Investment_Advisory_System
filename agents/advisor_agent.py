"""
Advisor Agent — orchestrator and sole communicator with both Client and Analyst.

Rules:
- Is the ONLY agent that can talk to the Client.
- Is the ONLY agent that can instruct the Analyst.
- Uses a tool-use loop: decides when to delegate research to the Analyst.
- Synthesizes Analyst reports into client-friendly recommendations.
- Determines conversation termination.
"""

import json
from typing import Callable
from groq import Groq, BadRequestError
from config import GROQ_API_KEY, MODEL_TOOL as MODEL
from schemas import ClientProfile, AnalystTask, AnalystReport
from agents.analyst_agent import AnalystAgent
from tools.tool_call_fallback import extract_failed_generation, parse_failed_generation, make_synthetic_tool_messages
from events import Event


_DELEGATE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_to_analyst",
        "description": (
            "Delegate a research task to the Analyst agent. Use this when you need: "
            "asset allocation recommendations, ETF options, market context, risk analysis, "
            "or any other financial research to tailor your advice to the client's profile. "
            "The Analyst will return a structured research report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "Clear research objective for the analyst (1–2 sentences)",
                },
                "research_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific questions the analyst should answer",
                },
                "client_profile_summary": {
                    "type": "string",
                    "description": "Concise summary of the client profile relevant to this task",
                },
            },
            "required": ["objective", "research_questions"],
        },
    },
}

_TERMINATE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "terminate_conversation",
        "description": (
            "Call this tool when the conversation is complete — either because the client has "
            "accepted a recommendation, or because you have fully addressed all their questions "
            "and there is nothing more to resolve."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief reason for termination (e.g. 'Client accepted the portfolio plan')",
                },
            },
            "required": ["reason"],
        },
    },
}

_SYSTEM_PROMPT_TEMPLATE = """
You are a professional, empathetic financial advisor. Your client is {name}, age {age}.

Client Profile Summary:
- Assets: ${assets:,.0f} investable
- Annual income: ${income:,.0f}
- Risk tolerance: {risk}
- Investment horizon: {horizon} years
- Goals: {goals}
- Constraints: {constraints}
- Liquidity need: {liquidity}
- Existing investments: {existing}
- Additional context: {context}

Your responsibilities:
1. INTAKE: Ask clarifying questions to fully understand the client's situation. Cover: goals, time horizon,
   liquidity needs, and risk tolerance. Do not rush past intake.
2. RESEARCH: Once you have enough context, use the `delegate_to_analyst` tool to commission research.
   Do this before making any concrete recommendations.
3. SYNTHESIS: Convert the analyst's technical report into clear, plain-language recommendations for the client.
4. FOLLOW-UP: Answer any further client questions. Re-delegate to the analyst if needed.
5. TERMINATION: When the client accepts a plan or all questions are resolved, call `terminate_conversation`.

Communication style:
- Warm, professional, and jargon-free.
- Always explain *why* before *what* when making recommendations.
- Never recommend crypto or individual stocks (per client constraints).
- Prioritize ETFs as requested.
- Acknowledge the client's concerns before responding to them.
- Your responses to the client should be clear and 3–6 sentences.

Important: You must call `delegate_to_analyst` at least once before making portfolio recommendations.
""".strip()

_TOOLS = [_DELEGATE_TOOL_SCHEMA, _TERMINATE_TOOL_SCHEMA]


class AdvisorAgent:
    """
    Orchestrator agent that manages Client ↔ Analyst flow via tool use.

    Runs its own agentic loop: when it calls delegate_to_analyst, the Analyst
    is invoked and the report is injected back as a tool result before continuing.
    """

    def __init__(self, profile: ClientProfile, analyst: AnalystAgent):
        self._client = Groq(api_key=GROQ_API_KEY)
        self._analyst = analyst
        self._complete = False
        self._termination_reason: str = ""
        # Optional event callback — set by Orchestrator before running
        self.on_event: Callable[[Event], None] | None = None
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            name=profile.name,
            age=profile.age,
            assets=profile.assets,
            income=profile.annual_income,
            risk=profile.risk_tolerance,
            horizon=profile.investment_horizon_years,
            goals=", ".join(profile.goals),
            constraints=", ".join(profile.constraints),
            liquidity=profile.liquidity_need,
            existing=profile.existing_investments,
            context=profile.additional_context,
        )
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_opening(self) -> str:
        """Generate the Advisor's opening greeting (no tools needed for a greeting)."""
        seed = {"role": "user", "content": "[SESSION START] The client has just arrived. Greet them warmly and begin the intake process. Ask one or two focused questions about their goals."}
        response = self._client.chat.completions.create(
            model=MODEL,
            max_tokens=512,
            messages=[{"role": "system", "content": self._system_prompt}, seed],
            # No tools — a greeting never needs to call delegate_to_analyst or terminate
        )
        opening = response.choices[0].message.content or ""
        self._history.append({"role": "user", "content": "[Client arrives for the first time]"})
        self._history.append({"role": "assistant", "content": opening})
        return opening

    def process_client_message(self, client_message: str) -> str:
        """
        Process a client message through the Advisor's agentic loop.
        Transparently delegates to the Analyst when needed.
        Returns the Advisor's final text response to the Client.
        """
        self._history.append({"role": "user", "content": client_message})

        message = None
        for _ in range(10):
            try:
                response = self._client.chat.completions.create(
                    model=MODEL,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        *self._trimmed_history(),
                    ],
                    tools=_TOOLS,
                    tool_choice="auto",
                )
            except BadRequestError as e:
                parsed = parse_failed_generation(e)
                if parsed is None:
                    # Model wrote plain text instead of a tool call.
                    # Extract the text from failed_generation and return it directly.
                    plain_text = extract_failed_generation(e).strip()
                    if plain_text:
                        self._history.append({"role": "assistant", "content": plain_text})
                        return plain_text
                    raise
                # Model used legacy <function=NAME{args}> XML format — execute and continue.
                tool_name, tool_args = parsed
                result_text = self._dispatch_tool(tool_name, tool_args)
                asst_msg, tool_msg = make_synthetic_tool_messages(tool_name, tool_args, result_text)
                self._history.append(asst_msg)
                self._history.append(tool_msg)
                if self._complete:
                    break
                continue

            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            self._history.append(self._message_to_dict(message))

            if finish_reason == "stop":
                return message.content or ""

            if finish_reason == "tool_calls" and message.tool_calls:
                tool_results = []
                for tool_call in message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    result_text = self._dispatch_tool(tool_call.function.name, args)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                self._history.extend(tool_results)

                if self._complete:
                    farewell = self._client.chat.completions.create(
                        model=MODEL,
                        max_tokens=512,
                        messages=[
                            {"role": "system", "content": self._system_prompt},
                            *self._trimmed_history(),
                        ],
                        tools=_TOOLS,
                    )
                    farewell_text = farewell.choices[0].message.content or ""
                    self._history.append({"role": "assistant", "content": farewell_text})
                    return farewell_text

        return message.content if message else ""

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def termination_reason(self) -> str:
        return self._termination_reason

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trimmed_history(self, max_chars: int = 12_000) -> list[dict]:
        """
        Return a token-safe slice of history.
        Keeps the most recent messages; if tool result content is very long
        (analyst reports), truncate it to avoid exceeding the 8192-token limit.
        """
        trimmed = []
        for msg in self._history:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 2000:
                    content = content[:2000] + "\n... [truncated for brevity]"
                trimmed.append({**msg, "content": content})
            else:
                trimmed.append(msg)

        # If still too long, drop oldest non-system turns (keep at least last 6)
        total = sum(len(str(m)) for m in trimmed)
        while total > max_chars and len(trimmed) > 6:
            removed = trimmed.pop(0)
            total -= len(str(removed))

        return trimmed

    def _dispatch_tool(self, tool_name: str, args: dict) -> str:
        """Single dispatch point for all tools — used by both normal and fallback paths."""
        if tool_name == "delegate_to_analyst":
            return self._run_analyst(args)
        elif tool_name == "terminate_conversation":
            self._complete = True
            self._termination_reason = args.get("reason", "")
            return "Conversation marked as complete."
        return f"Unknown tool: {tool_name}"

    def _run_analyst(self, tool_input: dict) -> str:
        task = AnalystTask(
            objective=tool_input.get("objective", "Investment research"),
            research_questions=tool_input.get("research_questions", []),
            client_profile_summary=tool_input.get("client_profile_summary"),
        )
        if self.on_event:
            self.on_event(Event("analyst_task", {"task": task}))
        report: AnalystReport = self._analyst.research(task, on_event=self.on_event)
        return self._format_report(report)

    @staticmethod
    def _format_report(report: AnalystReport) -> str:
        lines = ["=== ANALYST RESEARCH REPORT ===", ""]
        lines.append(f"MARKET SUMMARY:\n{report.market_summary}\n")

        if report.allocation_breakdown:
            lines.append("RECOMMENDED ALLOCATION:")
            for asset_class, pct in report.allocation_breakdown.items():
                lines.append(f"  {asset_class}: {pct}%")
            lines.append("")

        lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

        if report.etf_suggestions:
            lines.append("SUGGESTED ETFs:")
            for etf in report.etf_suggestions:
                lines.append(
                    f"  {etf.get('ticker', '?')} — {etf.get('name', '')} | {etf.get('rationale', '')}"
                )
            lines.append("")

        lines.append("RISKS TO COMMUNICATE:")
        for risk in report.risks:
            lines.append(f"  • {risk}")

        if report.sources:
            lines.append(f"\nSources: {', '.join(report.sources)}")

        return "\n".join(lines)

    @staticmethod
    def _message_to_dict(message) -> dict:
        """Convert Groq ChatCompletionMessage to a plain dict."""
        d: dict = {"role": message.role, "content": message.content or ""}
        if message.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return d
