import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== CONFIG =====================
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

STATE_FILE = "agent_state.json"
MAX_WORKING_MEMORY = 8
MAX_MEMORY_CHARS = 3000
MAX_NEW_TOKENS = 220

# Emotion model (numeric, internal only)
EMOTIONS = {
    "neutral": 0.0,
    "curious": 0.0,
    "confident": 0.0,
    "empathetic": 0.0,
    "frustrated": 0.0
}

SYSTEM_RULES = """
You are an intelligent autonomous AI assistant.
Follow user intent carefully.
Be honest if unsure.
Do not mention internal states, emotions, or memory systems.
"""

# ===================== MODEL =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# ===================== STATE =====================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "working_memory": [],
        "memory_summary": "",
        "important_facts": [],
        "emotions": EMOTIONS.copy()
    }

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# ===================== UTILS =====================
def run_llm(prompt, max_tokens=200, temperature=0.4):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

# ===================== INTENT =====================
def infer_intent(user_input, context_summary):
    prompt = f"""
<|system|>
Classify the user's intent concisely.

Possible intents:
- question
- instruction
- storytelling
- emotional_support
- exploration
- technical_problem

Context:
{context_summary}

User:
{user_input}

<|assistant|>
Intent:
"""
    return run_llm(prompt, 20).lower()

# ===================== EMOTION UPDATE =====================
def update_emotions(emotions, user_input):
    prompt = f"""
Analyze emotional impact of user message.
Return JSON with emotion deltas between -1 and 1.

Emotions: {list(emotions.keys())}

User message:
{user_input}

<|assistant|>
"""
    try:
        delta = json.loads(run_llm(prompt, 60))
        for k in emotions:
            emotions[k] = max(-5, min(5, emotions[k] + delta.get(k, 0)))
    except:
        pass
    return emotions

# ===================== MEMORY IMPORTANCE =====================
def score_memory(text):
    prompt = f"""
Score importance of this information from 0 to 10.
Only return number.

Text:
{text}

<|assistant|>
"""
    try:
        return float(run_llm(prompt, 10))
    except:
        return 1.0

# ===================== MEMORY COMPRESSION =====================
def compress_memory(state):
    joined = "\n".join(
        f"User: {m['user']}\nAI: {m['assistant']}"
        for m in state["working_memory"]
    )

    prompt = f"""
Summarize the following conversation into key long-term facts.
Be concise.

Conversation:
{joined}

<|assistant|>
"""
    summary = run_llm(prompt, 150)

    # extract important facts
    facts_prompt = f"""
Extract important long-term facts as bullet points.

Summary:
{summary}

<|assistant|>
"""
    facts_text = run_llm(facts_prompt, 120)
    facts = [
        {"text": f.strip("- "), "weight": score_memory(f)}
        for f in facts_text.split("\n") if f.strip()
    ]

    state["memory_summary"] = summary
    state["important_facts"].extend(facts)
    state["working_memory"] = []

# ===================== PROMPT BUILDER =====================
def build_prompt(state, user_input, intent):
    facts = "\n".join(
        f"- ({f['weight']}) {f['text']}"
        for f in sorted(state["important_facts"], key=lambda x: -x["weight"])[:5]
    )

    emotions = ", ".join(
        f"{k}:{round(v,2)}" for k, v in state["emotions"].items()
    )

    prompt = f"""
<|system|>
{SYSTEM_RULES}

Current emotional state (internal):
{emotions}

Conversation summary:
{state["memory_summary"]}

Important facts:
{facts}

Inferred intent:
{intent}

<|user|>
{user_input}

<|assistant|>
"""
    return prompt

# ===================== MAIN LOOP =====================
def chat():
    state = load_state()
    print("🧠 Advanced Local Agent (type 'exit')\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            save_state(state)
            print("👋 State saved.")
            break

        # 1. Infer intent
        intent = infer_intent(user_input, state["memory_summary"])

        # 2. Update emotions
        state["emotions"] = update_emotions(state["emotions"], user_input)

        # 3. Build prompt
        prompt = build_prompt(state, user_input, intent)

        # 4. Generate response
        response = run_llm(prompt, MAX_NEW_TOKENS)

        print(f"\nAI: {response}\n")

        # 5. Save working memory
        state["working_memory"].append({
            "user": user_input,
            "assistant": response
        })

        # 6. Compress memory if too large
        total_chars = sum(len(m["user"]) + len(m["assistant"])
                          for m in state["working_memory"])
        if total_chars > MAX_MEMORY_CHARS:
            compress_memory(state)

        save_state(state)

if __name__ == "__main__":
    chat()
