import torch
from src.gemma4.model import processor, model


# ===================== LLM CORE (backward-compatible) =====================
def run_llm(prompt, gen_kwargs):
    """Text-only generation. Used by emotions, intent, memory modules."""
    inputs = processor(
        text=prompt,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    generated_ids = output[0][input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()
