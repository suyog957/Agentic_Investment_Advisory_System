"""
Event dataclass shared across all agents.

Agents emit events via a callback so the caller (Orchestrator / Streamlit)
can observe everything that happens — client turns, advisor turns, analyst
tool calls, analyst reports — without coupling the agents to any specific UI.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """
    kind values:
      "advisor"         — Advisor speaks to Client
      "client"          — Client speaks to Advisor
      "analyst_task"    — Advisor delegates a task to Analyst (includes task obj)
      "tool_call"       — Analyst calls a tool (name + args)
      "tool_result"     — Tool returns a result (name + truncated result)
      "analyst_report"  — Analyst returns structured AnalystReport to Advisor
      "session_complete"— Session has ended
    """
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
