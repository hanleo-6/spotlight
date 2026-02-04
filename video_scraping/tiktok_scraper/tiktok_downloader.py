import json
from pathlib import Path
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

INPUT_JSON = Path("video_scraping/tiktok_scraper/tiktok_data_2.json")
OUTPUT_DIR = Path("data/input/tiktok_vids")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "cookiefile": "video_scraping/tiktok_scraper/cookies.txt",
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

failed = []

def is_already_downloaded(ydl: YoutubeDL, url: str) -> bool:
    try:
        info = ydl.extract_info(url, download=False)
    except DownloadError:
        return False

    filename = ydl.prepare_filename(info)
    video_path = Path(filename)
    merge_ext = ydl_opts.get("merge_output_format")
    if merge_ext and video_path.suffix != f".{merge_ext}":
        video_path = video_path.with_suffix(f".{merge_ext}")

    return video_path.exists()

with YoutubeDL(ydl_opts) as ydl:
    for url in urls:
        try:
            if is_already_downloaded(ydl, url):
                continue
            ydl.download([url])
        except DownloadError as e:
            msg = str(e)
            if "10231" in msg:
                failed.append({"url": url, "status": "unavailable_10231"})
            elif "403" in msg:
                failed.append({"url": url, "status": "forbidden_403"})
            else:
                failed.append({"url": url, "status": "other"})
