import torch
from src.gemma4.model import processor, model


# ===================== CHAT (text → text) =====================
def chat(
    messages,
    enable_thinking=False,
    system_prompt=None,
    max_new_tokens=1024,
):
    """Chat with Gemma 4 using the official chat template.

    Args:
        messages: List of dicts with "role" and "content" keys.
                  e.g. [{"role": "user", "content": "Hello!"}]
                  Roles: "system", "user", "assistant".
        enable_thinking: Enable step-by-step reasoning before answering.
        system_prompt: Shorthand — prepended as a system message if provided.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Parsed response dict from processor.parse_response().
    """
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)


def chat_stream(
    messages,
    enable_thinking=False,
    system_prompt=None,
    max_new_tokens=1024,
):
    """Simple wrapper that returns just the text content from chat()."""
    result = chat(
        messages,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )
    if isinstance(result, dict):
        return result.get("content", str(result))
    return str(result)
