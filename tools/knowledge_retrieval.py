"""Tool: search the internal investment knowledge base via RAG."""

from knowledge_store import InvestmentKnowledgeStore

# Singleton — loaded once, reused across all Analyst calls
_store: InvestmentKnowledgeStore | None = None


def _get_store() -> InvestmentKnowledgeStore:
    global _store
    if _store is None:
        _store = InvestmentKnowledgeStore()
    return _store


def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Search the internal knowledge base for investment-related information.

    Args:
        query: Natural-language search query.
        n_results: Number of document chunks to retrieve (default 5).

    Returns:
        Concatenated relevant document excerpts with source labels.
    """
    store = _get_store()
    results = store.query(query_text=query, n_results=n_results)
    if not results:
        return "No relevant information found in the knowledge base."
    return results


# Groq / OpenAI-compatible tool schema
KNOWLEDGE_BASE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the internal investment knowledge base. Use this to find information about "
            "risk profiles, asset allocation strategies, ETF recommendations, and core investing "
            "principles. Always search here first before using the web."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g. 'moderate risk ETF allocation', 'bond ETFs for education fund')",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10)",
                },
            },
            "required": ["query"],
        },
    },
}
