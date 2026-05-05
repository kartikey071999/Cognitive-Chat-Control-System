import torch
from PIL import Image
from src.gemma4.model import processor, model

# Max video length: 60 seconds at 1 fps = 60 frames
MAX_VIDEO_FRAMES = 60


# ===================== VIDEO → TEXT =====================
def video_to_text(
    frame_paths,
    prompt="Describe what happens in this video.",
    enable_thinking=False,
    system_prompt=None,
    max_new_tokens=1024,
    image_token_budget=280,
):
    """Analyze a video (as a sequence of frames) and generate text.

    Gemma 4 processes videos as image frame sequences. Max 60 seconds
    at 1 frame per second. Extract frames beforehand and pass as paths.

    Args:
        frame_paths: List of paths to video frame images (in order).
                     Max 60 frames recommended.
        prompt: Text prompt/question about the video.
        enable_thinking: Enable step-by-step reasoning.
        system_prompt: Optional system instruction.
        max_new_tokens: Maximum tokens to generate.
        image_token_budget: Visual token budget per frame.
                           Use lower budgets (140-280) for video.

    Returns:
        Parsed response dict from processor.parse_response().
    """
    frame_paths = frame_paths[:MAX_VIDEO_FRAMES]
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    # Frames (images) before text
    user_content = [{"type": "image", "image": f} for f in frames]
    user_content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = processor(
        text=text,
        images=frames,
        return_tensors="pt",
        image_token_budget=image_token_budget,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)
