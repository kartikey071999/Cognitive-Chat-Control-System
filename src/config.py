import os
from dotenv import load_dotenv

load_dotenv()

# ===================== SECRETS (from .env) =====================
HF_TOKEN = os.environ.get("HF_TOKEN")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")

# ===================== CONFIG =====================
MODEL_NAME = "google/gemma-4-E4B-it"

STATE_FILE = "agent_state.json"
MAX_MEMORY_CHARS = 3000
MAX_NEW_TOKENS = 1024

# Emotion model (numeric, internal only)
EMOTIONS = {
    "neutral": 0.0,
    "curious": 0.0,
    "confident": 0.0,
    "empathetic": 0.0,
    "frustrated": 0.0,
}

SYSTEM_RULES = """
You are a helpful, calm AI assistant.
Follow user intent.
Be honest if unsure.
"""

# ===================== GENERATION KWARGS =====================
# Passed directly to model.generate() — not GenerationConfig
GEN_CREATIVE = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}

GEN_DETERMINISTIC = {
    "max_new_tokens": 50,
    "do_sample": False,
}
