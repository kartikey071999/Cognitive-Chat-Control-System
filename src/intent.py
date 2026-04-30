from src.config import GEN_DETERMINISTIC
from src.llm_diff import run_llm


# ===================== INTENT =====================
def infer_intent(user_input):
    prompt = f"""
SYSTEM:
Classify intent as one word:
question | instruction | storytelling | emotional_support | exploration | technical_problem

USER:
{user_input}

INTENT:
"""
    text = run_llm(prompt, GEN_DETERMINISTIC)
    return text.strip().split()[-1].lower()
