import json
from src.config import GEN_DETERMINISTIC
from src.llm_diff import run_llm


# ===================== EMOTION =====================
def update_emotions(emotions, user_input):
    prompt = f"""
SYSTEM:
Return JSON emotion deltas between -1 and 1.

Emotions: {list(emotions.keys())}

USER:
{user_input}

JSON:
"""
    try:
        delta = json.loads(run_llm(prompt, GEN_DETERMINISTIC))
        for k in emotions:
            emotions[k] = max(-5.0, min(5.0, emotions[k] + float(delta.get(k, 0))))
    except Exception:
        pass
    return emotions
