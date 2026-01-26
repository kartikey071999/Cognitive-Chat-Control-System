import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import MODEL_NAME

# ===================== MODEL =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 requirement

use_cuda = torch.cuda.is_available()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if use_cuda else "cpu",
    max_memory={0: "5GB", "cpu": "16GB"} if use_cuda else None,
    dtype=torch.float16 if use_cuda else torch.float32,
)


# ===================== LLM CORE =====================
def run_llm(prompt, gen_config):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024 - gen_config.max_new_tokens,  # reserve space
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs, generation_config=gen_config, pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("ASSISTANT:")[-1].strip()
