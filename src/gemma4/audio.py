import torch
import librosa
from src.gemma4.model import processor, model


# Max audio length supported by Gemma 4 E4B
MAX_AUDIO_SECONDS = 30


# ===================== AUDIO → TEXT (ASR) =====================
def audio_to_text(
    audio_path,
    language="English",
    max_new_tokens=512,
    sr=16000,
    enable_thinking=False,
):
    """Automatic Speech Recognition (ASR) — transcribe audio to text.

    Supported natively on E2B and E4B models only. Max 30 seconds.

    Args:
        audio_path: Path to the audio file (wav, mp3, flac, etc.).
        language: Language of the speech for transcription.
        max_new_tokens: Maximum tokens to generate.
        sr: Sample rate (default 16000 Hz).
        enable_thinking: Enable step-by-step reasoning.

    Returns:
        Parsed response dict from processor.parse_response().
    """
    audio, _ = librosa.load(audio_path, sr=sr)

    # Enforce 30-second limit
    max_samples = MAX_AUDIO_SECONDS * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    prompt = (
        f"Transcribe the following speech segment in {language} into {language} text.\n\n"
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
        "one point seven, and write 3 instead of three."
    )

    # Audio before text for optimal performance
    user_content = [
        {"type": "audio", "audio": audio, "sample_rate": sr},
        {"type": "text", "text": prompt},
    ]

    messages = [{"role": "user", "content": user_content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = processor(
        text=text,
        audios=audio,
        sampling_rate=sr,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)


# ===================== AUDIO → TRANSLATED TEXT (AST) =====================
def audio_translate(
    audio_path,
    source_language="English",
    target_language="Spanish",
    max_new_tokens=512,
    sr=16000,
    enable_thinking=False,
):
    """Automatic Speech Translation (AST) — transcribe and translate audio.

    Supported natively on E2B and E4B models only. Max 30 seconds.

    Args:
        audio_path: Path to the audio file.
        source_language: Language of the source speech.
        target_language: Language to translate into.
        max_new_tokens: Maximum tokens to generate.
        sr: Sample rate (default 16000 Hz).
        enable_thinking: Enable step-by-step reasoning.

    Returns:
        Parsed response dict from processor.parse_response().
    """
    audio, _ = librosa.load(audio_path, sr=sr)

    max_samples = MAX_AUDIO_SECONDS * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    prompt = (
        f"Transcribe the following speech segment in {source_language}, "
        f"then translate it into {target_language}.\n"
        f"When formatting the answer, first output the transcription in {source_language}, "
        f"then one newline, then output the string '{target_language}: ', "
        f"then the translation in {target_language}."
    )

    user_content = [
        {"type": "audio", "audio": audio, "sample_rate": sr},
        {"type": "text", "text": prompt},
    ]

    messages = [{"role": "user", "content": user_content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = processor(
        text=text,
        audios=audio,
        sampling_rate=sr,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)
