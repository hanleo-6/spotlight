import json
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------
INPUT_FILE = Path("tiktok/tiktok_data_1.json")       # your source JSON
SCRAPED_FILE = Path("scraped_videos.json")  # local store of scraped video IDs

# Load existing scraped IDs
if SCRAPED_FILE.exists():
    with SCRAPED_FILE.open("r", encoding="utf-8") as f:
        scraped_ids = set(json.load(f))
else:
    scraped_ids = set()

# Load input JSON
with INPUT_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

new_urls = []

for item in data:
    video_url = item.get("webVideoUrl")
    if not video_url:
        continue
    # Extract numeric video ID from URL
    video_id = video_url.rstrip("/").split("/")[-1]
    if video_id not in scraped_ids:
        scraped_ids.add(video_id)
        new_urls.append(video_url)

# Save updated scraped IDs
with SCRAPED_FILE.open("w", encoding="utf-8") as f:
    json.dump(list(scraped_ids), f, indent=2)

# Optional: print new URLs
for url in new_urls:
    print(url)
