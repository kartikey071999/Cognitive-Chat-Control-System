from src.config import MAX_MEMORY_CHARS, GEN_CREATIVE
from src.state import load_state, save_state
from src.llm import run_llm
from src.intent import infer_intent
from src.emotions import update_emotions
from src.memory import compress_memory
from src.utils import build_prompt


# ===================== MAIN LOOP =====================
def chat():
    state = load_state()
    print("🧠 Cognitive Chat Control System (GPT-2 test mode)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            save_state(state)
            print("👋 State saved.")
            break

        intent = infer_intent(user_input)
        state["emotions"] = update_emotions(state["emotions"], user_input)

        prompt = build_prompt(state, user_input, intent)
        response = run_llm(prompt, GEN_CREATIVE)

        print(f"\nAI: {response}\n")

        state["working_memory"].append({"user": user_input, "assistant": response})

        total_chars = sum(
            len(m["user"]) + len(m["assistant"]) for m in state["working_memory"]
        )

        if total_chars > MAX_MEMORY_CHARS:
            compress_memory(state)

        save_state(state)


if __name__ == "__main__":
    chat()
