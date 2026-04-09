# Agentic Investment Advisory System

A multi-agent system built for the QuadSci take-home assignment. Three LLM-powered agents collaborate to give a simulated client personalised investment advice — the client talks to the advisor, the advisor figures out what it needs to know, and the analyst goes and finds it.

## What it does

The advisor runs the conversation. When it needs research — asset allocation ideas, ETF options, current market context — it delegates a task to the analyst rather than making things up. The analyst has two tools: a local knowledge base (ChromaDB + sentence-transformers) and a live web search (DuckDuckGo). It uses both, then hands a structured report back to the advisor, who translates it into something the client can actually act on. The session ends when the client accepts a recommendation or when the advisor decides the conversation is resolved.

```
Client  <-->  Advisor  <-->  Analyst
                               |-- search_knowledge_base
                               `-- web_search
```

The Streamlit UI shows the full picture — not just the client/advisor chat, but every research task the advisor assigned, every tool the analyst called, and the structured report that came back.

## Project structure

```
agents/
  client_agent.py     # simulated investor persona (Sarah Chen, 42, $350K)
  advisor_agent.py    # orchestrator — the only agent that talks to both sides
  analyst_agent.py    # research agent with an agentic tool-use loop

knowledge_store/
  vector_store.py     # ChromaDB + sentence-transformers, auto-ingests on first run
  documents/          # four markdown knowledge docs (risk profiles, ETFs, allocation, principles)

tools/
  knowledge_retrieval.py   # RAG search over the knowledge store
  web_search.py            # DuckDuckGo search
  tool_call_fallback.py    # handles Groq's occasional malformed tool call output

schemas/models.py     # Pydantic models: ClientProfile, AnalystTask, AnalystReport
events.py             # event dataclass used to stream activity to the UI
orchestrator.py       # drives the conversation loop
streamlit_app.py      # UI
main.py               # terminal entry point
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your Groq API key to .env
```

First run will download the `all-MiniLM-L6-v2` embedding model (~80MB) and populate the ChromaDB store from the four knowledge documents. Subsequent runs are instant.

## Running it

**Streamlit (recommended):**
```bash
streamlit run streamlit_app.py
```

**Terminal:**
```bash
python main.py
```

## A few design decisions worth mentioning

**Why the advisor uses tool calls to delegate, not a direct function call.** The advisor is an LLM — it decides *when* to consult the analyst based on the conversation, not on a hard-coded rule. Using `delegate_to_analyst` as a tool means the model can reason about whether it has enough information before going off to do research. In practice this means it usually completes intake first, then delegates once, then synthesises — which is the right order.

**The fallback parser.** Groq's Llama 3.3 sometimes outputs tool calls in an older XML format (`<function=NAME{args}>`) that the API rejects. Rather than crashing, `tool_call_fallback.py` catches the 400 error, parses the intended call out of the `failed_generation` field, executes it, and injects synthetic tool messages so the conversation can continue. There's also a second case: the model occasionally writes a plain conversational response when it should have called a tool — that text gets extracted and returned directly.

**ChromaDB over a managed vector store.** Kept it local and dependency-light. The knowledge base is small (four documents, ~80 chunks) so there's no need for a hosted service. Everything persists to disk between runs.

**Single model for all agents.** All three agents use `llama-3.3-70b-versatile`. An earlier version tried a separate fine-tuned tool-use model for the advisor and analyst, but that model was decommissioned by Groq. The versatile model handles tool use well enough with the fallback parser in place.

## Limitations

- Groq's free tier has a 100K tokens/day limit. A full session uses roughly 15–30K tokens depending on how many turns it runs and how much the analyst searches. You'll hit the limit after 3–6 sessions in a day.
- The client profile is pre-configured — there's no form for a human to fill in their actual financial situation. The sidebar in the Streamlit app lets you adjust the parameters but the client is still simulated.
- Not financial advice.
