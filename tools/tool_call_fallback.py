"""
Fallback parser for Groq tool_use_failed errors.

When Groq's Llama models generate tool calls in the legacy
<function=NAME{...args...}</function> format instead of proper JSON,
the API returns a 400 tool_use_failed error with a 'failed_generation' field.
This module extracts the intended call and lets execution continue.
"""

import re
import json
import uuid


def extract_failed_generation(error) -> str:
    """Return the raw failed_generation string from a tool_use_failed error, or ''."""
    try:
        return error.body.get("error", {}).get("failed_generation", "")
    except Exception:
        return ""


def parse_failed_generation(error) -> tuple[str, dict] | None:
    """
    Extract (tool_name, tool_args) from a tool_use_failed BadRequestError.
    Returns None when the failed_generation is plain text (not a function call).
    """
    failed_gen = extract_failed_generation(error)
    if not failed_gen:
        return None

    # Model emits one of two formats:
    #   <function=TOOL_NAME{"arg": val}</function>
    #   <function=TOOL_NAME={"arg": val}</function>  <- extra = before {
    match = re.search(r"<function=(\w+)=?(\{.*?\})\s*</function>", failed_gen, re.DOTALL)
    if not match:
        return None

    tool_name = match.group(1)
    try:
        tool_args = json.loads(match.group(2))
    except json.JSONDecodeError:
        return None

    return tool_name, tool_args


def make_synthetic_tool_messages(tool_name: str, tool_args: dict, result: str) -> tuple[dict, dict]:
    """
    Build the assistant + tool messages that would have been produced
    had the model called the tool correctly.
    """
    fake_id = f"call_{uuid.uuid4().hex[:12]}"
    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": fake_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_args),
            },
        }],
    }
    tool_msg = {
        "role": "tool",
        "tool_call_id": fake_id,
        "content": result,
    }
    return assistant_msg, tool_msg
