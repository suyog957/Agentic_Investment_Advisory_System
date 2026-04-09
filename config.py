import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. Copy .env.example to .env and fill in your key.")

# Single model for all agents — llama-3.3-70b-versatile is the current Groq flagship
MODEL = "llama-3.3-70b-versatile"
MODEL_CHAT = MODEL   # kept as alias so existing imports don't break
MODEL_TOOL = MODEL   # kept as alias so existing imports don't break

# Knowledge store
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_store", "db")
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "knowledge_store", "documents")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Conversation limits
MAX_TURNS = 12
MAX_ANALYST_TOOL_ROUNDS = 5
