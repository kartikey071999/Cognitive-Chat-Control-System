from src.llm.chat import chat
from src.engram.graph import KnowledgeGraph


def main():
    print("Chat (type 'exit' or 'quit' to stop)\n")
    messages = []
    kg = KnowledgeGraph()

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError, KeyboardInterrupt:
            print("\nBye!")
            kg.exit()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            kg.exit()
            break

        messages.append({"role": "user", "content": user_input})
        print("AI: ", end="", flush=True)
        response = chat(messages, stream=True)
        messages.append({"role": "assistant", "content": response})

        # Enrich KG with the conversation turn
        kg.ingest_chat(f"{user_input}\n{response}")
        print(f"\n[KG] {kg}")
        for node, data in kg.get_nodes():
            print(f"  - {node} ({data.get('label', '?')})")
        print()


if __name__ == "__main__":
    main()
