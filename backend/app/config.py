"""Central configuration. Every external dependency is optional so the app runs
end to end on a laptop with zero keys (MOCK mode), and switches to real
providers as soon as keys are set in .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # optional
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# Database: SQLite locally, Supabase Postgres in production (just swap the URL)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signalos.db")

# Jev (TypeSafe AI) — typed decision model
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
JEV_URL = os.getenv("JEV_URL", "https://api.typesafe.ai/v1/systemone")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")

# Gemini — generative summaries / explanations / baseline classifier
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# Ticket destination (n8n webhook, Jira proxy, etc.). Empty = store locally only.
TICKET_WEBHOOK_URL = os.getenv("TICKET_WEBHOOK_URL", "")

# Priority formula weights (transparent + tunable, see priority_service.py)
W_SEVERITY = float(os.getenv("W_SEVERITY", 0.40))
W_VOLUME = float(os.getenv("W_VOLUME", 0.30))
W_CHURN = float(os.getenv("W_CHURN", 0.20))
W_RECENCY = float(os.getenv("W_RECENCY", 0.10))

# Aggregation: minimum similarity to merge feedback into an existing issue.
# SEMANTIC_THRESHOLD applies to embedding cosine; CLUSTER_THRESHOLD to the
# token-overlap fallback used when no embedding model is installed.
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", 0.45))
CLUSTER_THRESHOLD = float(os.getenv("CLUSTER_THRESHOLD", 0.15))

# Embeddings (local, free). Retrieval depth for /ask.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 8))


def jev_live() -> bool:
    return bool(TYPESAFE_API_KEY)


def gemini_live() -> bool:
    return bool(GEMINI_API_KEY)
