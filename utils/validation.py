import os
from pathlib import Path
from urllib.parse import urlparse

from utils.ffmpeg_tools import get_ffmpeg_executable


SUPPORTED_LANGUAGES = {"english", "hinglish"}
SUPPORTED_MEDIA_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
    ".wma",
}


class ValidationError(ValueError):
    """Raised when a pipeline request cannot be processed safely."""


def _has_real_value(name: str) -> bool:
    value = (os.getenv(name) or "").strip()
    if not value:
        return False
    return not value.lower().startswith("your_")


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_pipeline_request(source: str, language: str = "english") -> tuple[str, str]:
    source = (source or "").strip()
    language = (language or "english").strip().lower()

    if not source:
        raise ValidationError("Please provide a YouTube URL or a local media file path.")

    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValidationError(f"Unsupported language '{language}'. Supported values: {supported}.")

    try:
        get_ffmpeg_executable()
    except Exception as exc:
        raise ValidationError("FFmpeg is not available. Install FFmpeg or imageio-ffmpeg.") from exc

    if not _has_real_value("MISTRAL_API_KEY"):
        raise ValidationError("MISTRAL_API_KEY is not set in your environment or .env file.")

    if language == "hinglish" and not _has_real_value("SARVAM_API_KEY"):
        raise ValidationError("SARVAM_API_KEY is required when language is set to hinglish.")

    if is_url(source):
        return source, language

    media_path = Path(source).expanduser()
    if not media_path.exists():
        raise ValidationError(f"Local media file was not found: {media_path}")

    if not media_path.is_file():
        raise ValidationError(f"Expected a media file, but got: {media_path}")

    if media_path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_MEDIA_EXTENSIONS))
        raise ValidationError(f"Unsupported file type '{media_path.suffix}'. Allowed types: {allowed}.")

    return str(media_path), language
