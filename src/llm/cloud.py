from openai import OpenAI
from src.config import LLM_API_KEY, LLM_BASE_URL

_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def cloud_chat(messages, model, system_prompt=None, stream=False):
    """Send messages to the cloud LLM and return the full response text."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(messages)

    if stream:
        return _cloud_chat_stream(msgs, model)

    completion = _client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
    )
    return completion.choices[0].message.content


def _cloud_chat_stream(messages, model):
    """Stream response, printing tokens and returning full text."""
    completion = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        stream=True,
    )
    chunks = []
    for chunk in completion:
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)
        chunks.append(token)
    print()
    return "".join(chunks)
