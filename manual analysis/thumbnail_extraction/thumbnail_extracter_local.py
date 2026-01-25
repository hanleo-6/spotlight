import subprocess
from pathlib import Path

INPUT_DIR = Path("best_media")
OUTPUT_DIR = Path("thumbnails")
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

for video in INPUT_DIR.iterdir():
    if video.suffix.lower() not in VIDEO_EXTS:
        continue

    output = OUTPUT_DIR / f"{video.stem}.jpg"

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-ss", "00:00:01",
        "-vframes", "1",
        str(output)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Extracted {output}")
