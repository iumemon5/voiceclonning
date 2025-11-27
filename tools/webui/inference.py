import html
from functools import partial
from typing import Any, Callable

from lucy_speech.i18n import i18n
from lucy_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


def inference_wrapper(
    text,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    """

    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)
    else:
        references = []

    req = ServeTTSRequest(
        text=text,
        reference_id=reference_id if reference_id else None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
        use_memory_cache=use_memory_cache,
    )

    for result in engine.inference(req):
        match result.code:
            case "final":
                stats = format_stats(result.meta)
                return result.audio, None, stats
            case "error":
                return None, build_html_error_message(i18n(result.error)), None
            case _:
                pass

    return None, i18n("No audio generated"), None


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )


def format_stats(meta: dict | None) -> str:
    if not meta:
        return "No stats available"
    tokens = meta.get("tokens", 0)
    gen_time = meta.get("gen_time_sec", 0.0)
    audio_sec = meta.get("audio_sec", 0.0)
    tps = meta.get("tokens_per_sec", None)
    parts = [
        f"Tokens: {tokens}",
        f"Generation time: {gen_time:.2f}s",
        f"Audio length: {audio_sec:.2f}s",
    ]
    if tps is not None:
        parts.append(f"Tokens/sec: {tps:.2f}")
    return " | ".join(parts)
