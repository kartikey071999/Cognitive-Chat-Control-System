from transformers import GenerationConfig

# ===================== CONFIG =====================
MODEL_NAME = "gpt2"

STATE_FILE = "agent_state.json"
MAX_MEMORY_CHARS = 3000
MAX_NEW_TOKENS = 120

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

# ===================== GENERATION CONFIG =====================
GEN_CREATIVE = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)

GEN_DETERMINISTIC = GenerationConfig(
    max_new_tokens=50,
    temperature=0.0,
    do_sample=False,
)
