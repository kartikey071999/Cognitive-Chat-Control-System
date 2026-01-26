import os
import json
from src.config import STATE_FILE, EMOTIONS


# ===================== STATE =====================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "working_memory": [],
        "memory_summary": "",
        "important_facts": [],
        "emotions": EMOTIONS.copy(),
    }


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
