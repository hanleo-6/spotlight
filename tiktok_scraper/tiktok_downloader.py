import json
from pathlib import Path
from yt_dlp import YoutubeDL

INPUT_JSON = Path("tiktok/tiktok_data_1.json")
OUTPUT_DIR = Path("tiktok_vids")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_urls(path: Path):
    with path.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    # JSON array
    if content.startswith("["):
        data = json.loads(content)
        return [
            item["webVideoUrl"]
            for item in data
            if isinstance(item, dict) and "webVideoUrl" in item
        ]

    # JSONL fallback
    urls = []
    for line in content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "webVideoUrl" in obj:
            urls.append(obj["webVideoUrl"])
    return urls

urls = load_urls(INPUT_JSON)

ydl_opts = {
    "cookiefile": "tiktok/cookies.txt",
    "outtmpl": str(OUTPUT_DIR / "%(uploader)s/%(id)s.%(ext)s"),
    "format": "bv*+ba/b",
    "writeinfojson": True,
    "writethumbnail": True,
    "merge_output_format": "mp4",
    "extractor_args": {
        "tiktok": ["watermark=0"]
    },
    "sleep_interval": 2,
    "max_sleep_interval": 6,
    "retries": 5,
    "fragment_retries": 5,
    "force_ipv4": True,
    "user_agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

from yt_dlp.utils import DownloadError

failed = []

with YoutubeDL(ydl_opts) as ydl:
    for url in urls:
        try:
            ydl.download([url])
        except DownloadError as e:
            msg = str(e)
            if "10231" in msg:
                failed.append({"url": url, "status": "unavailable_10231"})
            elif "403" in msg:
                failed.append({"url": url, "status": "forbidden_403"})
            else:
                failed.append({"url": url, "status": "other"})
