from src.llm.cloud import cloud_chat

CHAT_LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SYSTEM_PROMPT = (
    "You are a helpful, calm AI assistant. Follow user intent. Be honest if unsure"
)


def chat(messages, system_prompt=None, model=None, stream=False):
    model = model or CHAT_LLM_MODEL
    system_prompt = system_prompt or SYSTEM_PROMPT
    return cloud_chat(messages, model=model, system_prompt=system_prompt, stream=stream)
