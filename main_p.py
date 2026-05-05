from src.engram.extractor import extract_triples
from src.engram.ingest import GraphIngestor

# HELP_TEXT = """
# ━━━ Commands ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   /think          Toggle thinking/reasoning mode
#   /image <path>   Describe or ask about an image
#   /audio <path>   Transcribe audio (ASR)
#   /translate <path> <src_lang> <tgt_lang>
#                   Transcribe + translate audio
#   /video <path>   Describe video frames (folder of images)
#   /clear          Clear conversation history
#   /state          Show current emotional state
#   /help           Show this help
#   exit / quit     Save and exit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# """


# def _build_system_prompt(state):
#     """Build system prompt with cognitive state context."""
#     emotions = ", ".join(f"{k}:{round(v, 2)}" for k, v in state["emotions"].items())
#     parts = [SYSTEM_RULES.strip()]
#     if state.get("memory_summary"):
#         parts.append(f"Memory summary:\n{state['memory_summary']}")
#     parts.append(f"Current emotional posture: {emotions}")
#     return "\n\n".join(parts)


# def _state_to_messages(state):
#     """Convert working_memory to Gemma 4 message format."""
#     messages = []
#     for turn in state["working_memory"]:
#         messages.append({"role": "user", "content": turn["user"]})
#         messages.append({"role": "assistant", "content": turn["assistant"]})
#     return messages


# # ===================== MAIN LOOP =====================
# def chat_loop():
#     state = load_state()
#     thinking_enabled = False

#     print("🧠 Cognitive Chat Control System (Gemma 4 E4B)")
#     print("   Type /help for commands, /think to enable reasoning\n")

#     while True:
#         try:
#             user_input = input("You: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             save_state(state)
#             print("\n👋 State saved.")
#             break

#         if not user_input:
#             continue

#         # ── Exit ──
#         if user_input.lower() in {"exit", "quit"}:
#             save_state(state)
#             print("👋 State saved.")
#             break

#         # ── Commands ──
#         if user_input == "/help":
#             print(HELP_TEXT)
#             continue

#         if user_input == "/think":
#             thinking_enabled = not thinking_enabled
#             status = "ON 🧠" if thinking_enabled else "OFF"
#             print(f"  Thinking mode: {status}\n")
#             continue

#         if user_input == "/clear":
#             state["working_memory"] = []
#             print("  Conversation history cleared.\n")
#             continue

#         if user_input == "/state":
#             for k, v in state["emotions"].items():
#                 bar = "█" * int(abs(v) * 4)
#                 sign = "+" if v >= 0 else "-"
#                 print(f"  {k:>12}: {sign}{bar} ({v:.2f})")
#             print()
#             continue

#         if user_input.startswith("/image "):
#             path = user_input[7:].strip()
#             if not os.path.isfile(path):
#                 print(f"  File not found: {path}\n")
#                 continue
#             prompt = input("  Prompt (enter for default): ").strip()
#             prompt = prompt or "Describe this image in detail."
#             print("  Analyzing image...")
#             result = image_to_text(
#                 path, prompt=prompt, enable_thinking=thinking_enabled
#             )
#             print(f"\nAI: {result}\n")
#             continue

#         if user_input.startswith("/audio "):
#             path = user_input[7:].strip()
#             if not os.path.isfile(path):
#                 print(f"  File not found: {path}\n")
#                 continue
#             lang = input("  Language (default: English): ").strip() or "English"
#             print("  Transcribing audio...")
#             result = audio_to_text(
#                 path, language=lang, enable_thinking=thinking_enabled
#             )
#             print(f"\nAI: {result}\n")
#             continue

#         if user_input.startswith("/translate "):
#             parts = user_input[11:].strip().split()
#             if len(parts) < 1:
#                 print("  Usage: /translate <path> [src_lang] [tgt_lang]\n")
#                 continue
#             path = parts[0]
#             if not os.path.isfile(path):
#                 print(f"  File not found: {path}\n")
#                 continue
#             src = parts[1] if len(parts) > 1 else "English"
#             tgt = parts[2] if len(parts) > 2 else "Spanish"
#             print(f"  Translating {src} → {tgt}...")
#             result = audio_translate(
#                 path,
#                 source_language=src,
#                 target_language=tgt,
#                 enable_thinking=thinking_enabled,
#             )
#             print(f"\nAI: {result}\n")
#             continue

#         if user_input.startswith("/video "):
#             folder = user_input[7:].strip()
#             if not os.path.isdir(folder):
#                 print(f"  Directory not found: {folder}\n")
#                 continue
#             frames = sorted(
#                 os.path.join(folder, f)
#                 for f in os.listdir(folder)
#                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
#             )
#             if not frames:
#                 print("  No image frames found in directory.\n")
#                 continue
#             prompt = input("  Prompt (enter for default): ").strip()
#             prompt = prompt or "Describe what happens in this video."
#             print(f"  Analyzing {len(frames)} frames...")
#             result = video_to_text(
#                 frames, prompt=prompt, enable_thinking=thinking_enabled
#             )
#             print(f"\nAI: {result}\n")
#             continue

#         # ── Regular chat ──
#         # Update cognitive state
#         intent = infer_intent(user_input)
#         state["emotions"] = update_emotions(state["emotions"], user_input)

#         # Build conversation with history
#         history = _state_to_messages(state)
#         history.append({"role": "user", "content": user_input})

#         system = _build_system_prompt(state)
#         if intent:
#             system += f"\nDetected user intent: {intent}"

#         response = chat_stream(
#             history,
#             enable_thinking=thinking_enabled,
#             system_prompt=system,
#         )

#         print(f"\nAI: {response}\n")

#         # Save turn
#         state["working_memory"].append({"user": user_input, "assistant": response})

#         total_chars = sum(
#             len(m["user"]) + len(m["assistant"]) for m in state["working_memory"]
#         )
#         if total_chars > MAX_MEMORY_CHARS:
#             compress_memory(state)

#         save_state(state)

if __name__ == "__main__":
    text = """
    Kartikey is a backend developer at EXL.
    He lives in Kanpur and loves building AI agents.
    """

    extraction = extract_triples(text)

    print("Extracted:")
    print(extraction.model_dump_json(indent=2))

    ingestor = GraphIngestor()
    ingestor.ingest(extraction)
    ingestor.close()

# if __name__ == "__main__":
#     chat_loop()
