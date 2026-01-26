from src.config import GEN_DETERMINISTIC
from src.llm import run_llm


# ===================== MEMORY =====================
def compress_memory(state):
    convo = "\n".join(
        f"User: {m['user']}\nAI: {m['assistant']}" for m in state["working_memory"]
    )

    prompt = f"""
SYSTEM:
Summarize important long-term facts only.

CONVERSATION:
{convo}

SUMMARY:
"""
    summary = run_llm(prompt, GEN_DETERMINISTIC)

    state["memory_summary"] = summary
    state["working_memory"] = []
