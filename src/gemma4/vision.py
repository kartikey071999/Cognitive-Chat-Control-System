import torch
from PIL import Image
from src.gemma4.model import processor, model


# ===================== IMAGE → TEXT =====================
def image_to_text(
    image_path,
    prompt="Describe this image in detail.",
    enable_thinking=False,
    system_prompt=None,
    max_new_tokens=1024,
    image_token_budget=560,
):
    """Analyze an image and generate a text response.

    Supports: object detection, OCR, document/PDF parsing, chart comprehension,
    screen/UI understanding, handwriting recognition, pointing.
    Variable aspect ratios and resolutions are handled automatically.

    Args:
        image_path: Path to the image file.
        prompt: Text prompt/question about the image.
        enable_thinking: Enable step-by-step reasoning.
        system_prompt: Optional system instruction.
        max_new_tokens: Maximum tokens to generate.
        image_token_budget: Visual token budget (70, 140, 280, 560, 1120).
                           Higher = more detail, lower = faster inference.

    Returns:
        Parsed response dict from processor.parse_response().
    """
    image = Image.open(image_path).convert("RGB")

    # Image before text for optimal performance (per Gemma 4 docs)
    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]

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
        images=image,
        return_tensors="pt",
        image_token_budget=image_token_budget,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)


def images_to_text(
    image_paths,
    prompt="Describe these images.",
    enable_thinking=False,
    system_prompt=None,
    max_new_tokens=1024,
    image_token_budget=560,
):
    """Interleaved multimodal: analyze multiple images with text.

    Args:
        image_paths: List of image file paths.
        prompt: Text prompt/question about the images.
        enable_thinking: Enable step-by-step reasoning.
        system_prompt: Optional system instruction.
        max_new_tokens: Maximum tokens to generate.
        image_token_budget: Visual token budget per image.

    Returns:
        Parsed response dict from processor.parse_response().
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]

    # Images before text
    user_content = [{"type": "image", "image": img} for img in images]
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
        images=images,
        return_tensors="pt",
        image_token_budget=image_token_budget,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)
