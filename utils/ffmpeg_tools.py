import shutil
import os
from pathlib import Path

import imageio_ffmpeg


def get_ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    bundled_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    project_root = Path(__file__).resolve().parents[1]
    local_bin = project_root / ".local_bin"
    local_bin.mkdir(exist_ok=True)

    local_ffmpeg = local_bin / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not local_ffmpeg.exists():
        shutil.copy2(bundled_ffmpeg, local_ffmpeg)

    return str(local_ffmpeg)


def ensure_ffmpeg_on_path() -> str:
    ffmpeg_executable = get_ffmpeg_executable()
    ffmpeg_dir = str(Path(ffmpeg_executable).parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_dir not in path_parts:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg_executable
