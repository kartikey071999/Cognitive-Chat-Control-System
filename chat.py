from src.llm.chat import chat
from src.engram.graph import KnowledgeGraph
from src.engram.context import KGContext


def main():
    print("Chat (type 'exit' or 'quit' to stop)\n")
    messages = []
    kg = KnowledgeGraph()
    ctx = KGContext()

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

        # Update context from KG based on user query
        ctx.update(kg, user_input)
        kg_context = ctx.get()

        # Build system prompt with KG context if available
        system_prompt = None
        if kg_context:
            print(f"[Context from KG]\n{kg_context}\n")
            system_prompt = f"Use this context from memory:\n{kg_context}"

        messages.append({"role": "user", "content": user_input})
        print("AI: ", end="", flush=True)
        response = chat(messages, system_prompt=system_prompt, stream=True)
        messages.append({"role": "assistant", "content": response})

        # Enrich KG with the conversation turn
        kg.ingest_chat(f"{user_input}\n{response}")
        # print(f"\n[KG] {kg}")
        # for node, data in kg.get_nodes():
        #     print(f"  - {node} ({data.get('label', '?')})")
        # print()


if __name__ == "__main__":
    main()
