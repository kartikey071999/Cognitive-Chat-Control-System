import torch
from huggingface_hub import snapshot_download, model_info
from transformers import AutoProcessor, AutoModelForCausalLM
from src.config import MODEL_NAME, HF_TOKEN


def _model_cached(model_name):
    """Check if model is already downloaded locally."""
    try:
        info = model_info(model_name, token=HF_TOKEN)
        cache_dir = snapshot_download(
            model_name, token=HF_TOKEN, local_files_only=True
        )
        print(f"✅ Model found in cache: {cache_dir}")
        return True
    except Exception:
        return False


# ===================== MODEL LOADING =====================
if not _model_cached(MODEL_NAME):
    print(f"⬇️  Downloading model {MODEL_NAME}...")

use_cuda = torch.cuda.is_available()

processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    device_map="auto" if use_cuda else None,
    token=HF_TOKEN,
)
