from typing import Callable

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from core.vector_store import new_session_id
from utils.audio_processor import cleanup_audio_artifacts, process_input
from utils.validation import validate_pipeline_request


PipelineCallback = Callable[[str, str], None]


def _notify(callback: PipelineCallback | None, stage: str, state: str) -> None:
    if callback:
        callback(stage, state)


def _run_optional_stage(
    stage: str,
    callback: PipelineCallback | None,
    errors: dict[str, str],
    fallback,
    fn,
):
    _notify(callback, stage, "active")
    try:
        result = fn()
    except Exception as exc:
        errors[stage] = str(exc)
        _notify(callback, stage, "failed")
        return fallback

    _notify(callback, stage, "done")
    return result


def run_pipeline(source: str, language: str = "english", callback: PipelineCallback | None = None) -> dict:
    source, language = validate_pipeline_request(source, language)
    session_id = new_session_id()
    errors: dict[str, str] = {}
    chunks: list[str] = []

    try:
        _notify(callback, "audio", "active")
        chunks = process_input(source)
        _notify(callback, "audio", "done")

        _notify(callback, "transcript", "active")
        transcript = transcribe_all(chunks, language)
        if not transcript.strip():
            raise RuntimeError("Transcription completed, but the transcript is empty.")
        _notify(callback, "transcript", "done")

        title = _run_optional_stage(
            "title",
            callback,
            errors,
            "Untitled Meeting",
            lambda: generate_title(transcript),
        )

        summary = _run_optional_stage(
            "summary",
            callback,
            errors,
            "Summary unavailable. The transcript is still available for review.",
            lambda: summarize(transcript),
        )

        def extract_all() -> tuple[str, str, str]:
            return (
                extract_action_items(transcript),
                extract_key_decisions(transcript),
                extract_questions(transcript),
            )

        action_items, decisions, questions = _run_optional_stage(
            "extract",
            callback,
            errors,
            (
                "Action item extraction unavailable.",
                "Decision extraction unavailable.",
                "Open question extraction unavailable.",
            ),
            extract_all,
        )

        rag_chain = _run_optional_stage(
            "rag",
            callback,
            errors,
            None,
            lambda: build_rag_chain(transcript, session_id=session_id, source=source),
        )

        return {
            "session_id": session_id,
            "source": source,
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
            "errors": errors,
        }
    finally:
        cleanup_audio_artifacts(chunks)
