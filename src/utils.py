from src.config import SYSTEM_RULES


# ===================== PROMPT =====================
def build_prompt(state, user_input, intent):
    emotions = ", ".join(f"{k}:{round(v, 2)}" for k, v in state["emotions"].items())

    return f"""
SYSTEM:
{SYSTEM_RULES}

Emotional posture:
{emotions}

Memory summary:
{state["memory_summary"]}

User intent:
{intent}

USER:
{user_input}

ASSISTANT:
"""
