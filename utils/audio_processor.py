import yt_dlp
from pydub import AudioSegment
import os
from pathlib import Path

from utils.ffmpeg_tools import ensure_ffmpeg_on_path
from utils.retrying import retry_external_call

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

FFMPEG_EXECUTABLE = ensure_ffmpeg_on_path()
AudioSegment.converter = FFMPEG_EXECUTABLE

@retry_external_call
def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "ffmpeg_location": FFMPEG_EXECUTABLE,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


def cleanup_audio_artifacts(chunks: list[str]) -> None:
    """Remove generated chunk files and intermediate WAV files."""
    generated_paths: set[Path] = set()

    for chunk in chunks or []:
        chunk_path = Path(chunk)
        generated_paths.add(chunk_path)

        marker = "_chunk_"
        chunk_text = str(chunk_path)
        if marker in chunk_text:
            generated_paths.add(Path(chunk_text.split(marker, 1)[0]))

    for path in generated_paths:
        should_delete = (
            "_chunk_" in str(path)
            or path.name.endswith("_converted.wav")
            or path.parent.name == DOWNLOAD_DIR
        )
        if should_delete and path.exists() and path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                print(f"Could not remove generated audio artifact {path}: {exc}")
