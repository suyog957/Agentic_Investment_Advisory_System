"""Tool: DuckDuckGo web search for current market information."""

from duckduckgo_search import DDGS


def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for current financial market information.

    Args:
        query: Search query string.
        num_results: Number of results to return (default 5).

    Returns:
        Formatted string of search results with title, URL, and snippet.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return "No web results found for this query."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            href = r.get("href", "")
            body = r.get("body", "No snippet available")
            formatted.append(f"[{i}] {title}\nURL: {href}\n{body}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Web search failed: {str(e)}. Rely on internal knowledge base instead."


# Groq / OpenAI-compatible tool schema
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for current financial data, ETF performance, market conditions, "
            "economic news, and up-to-date investment information. Use this for time-sensitive "
            "data that may not be in the internal knowledge base (e.g. current interest rates, "
            "recent market trends, latest ETF expense ratios)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of search results to return (default 5, max 10)",
                },
            },
            "required": ["query"],
        },
    },
}
