import json
import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Needs env variable api token
# Set APIFY_API_TOKEN in .env file
API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
SCRAPED_VIDEOS_FILE = Path("video_scraping/tiktok_scraper/scraped_videos.json")
OUTPUT_FILE = Path("video_scraping/tiktok_scraper/tiktok_data_2.json")

# Load previously scraped video IDs
if SCRAPED_VIDEOS_FILE.exists():
    with SCRAPED_VIDEOS_FILE.open("r", encoding="utf-8") as f:
        scraped_video_ids = set(json.load(f))
else:
    scraped_video_ids = set()

# Actor input
run_input = {
    "hashtags": ["indiedev", "aitools"],
    "resultsPerPage": 200,
    "profileScrapeSections": ["videos"],
    "profileSorting": "latest",
    "excludePinnedPosts": False,
    "maxFollowersPerProfile": 0,
    "maxFollowingPerProfile": 0,
    "searchSection": "",
    "maxProfilesPerQuery": 10,
    "searchSorting": "0",
    "searchDatePosted": "0",
    "scrapeRelatedVideos": False,
    "shouldDownloadVideos": False,
    "shouldDownloadCovers": False,
    "shouldDownloadSubtitles": False,
    "shouldDownloadSlideshowImages": False,
    "shouldDownloadAvatars": False,
    "shouldDownloadMusicCovers": False,
    "commentsPerPost": 0,
    "maxRepliesPerComment": 0,
    "proxyCountryCode": "None",
}

# Run the actor using curl
print("Starting Apify actor run...")
curl_command = [
    "curl",
    "-X", "POST",
    f"https://api.apify.com/v2/acts/GdWCkxBtKWOsKjdch/runs?token={API_TOKEN}",
    "-H", "Content-Type: application/json",
    "-d", json.dumps(run_input)
]

result = subprocess.run(curl_command, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error starting actor: {result.stderr}")
    exit(1)

run_response = json.loads(result.stdout)
run_id = run_response["data"]["id"]
default_dataset_id = run_response["data"]["defaultDatasetId"]
print(f"Actor run started with ID: {run_id}")

# Wait for the run to complete
print("Waiting for actor run to complete...")
while True:
    status_command = [
        "curl",
        f"https://api.apify.com/v2/actor-runs/{run_id}?token={API_TOKEN}"
    ]
    result = subprocess.run(status_command, capture_output=True, text=True)
    status_response = json.loads(result.stdout)
    status = status_response["data"]["status"]
    
    if status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"]:
        print(f"Actor run finished with status: {status}")
        break
    
    print(f"Status: {status}, waiting...")
    time.sleep(5)

if status != "SUCCEEDED":
    print(f"Actor run did not succeed: {status}")
    exit(1)

# Fetch results from dataset
print("Fetching results...")
dataset_command = [
    "curl",
    f"https://api.apify.com/v2/datasets/{default_dataset_id}/items?token={API_TOKEN}"
]

result = subprocess.run(dataset_command, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error fetching dataset: {result.stderr}")
    exit(1)

items = json.loads(result.stdout)

# Filter out already scraped videos
new_items = []
for item in items:
    # Extract TikTok video ID from URL
    video_url = item.get("webVideoUrl") or item.get("postUrl")
    if not video_url:
        continue
    # TikTok video ID is numeric part at the end of the URL
    video_id = video_url.rstrip("/").split("/")[-1]
    if video_id in scraped_video_ids:
        continue
    # Keep new video
    scraped_video_ids.add(video_id)
    new_items.append(item)
    print(item)

print(f"\nFound {len(new_items)} new videos (filtered {len(items) - len(new_items)} duplicates)")

# Save new items (filtered) to tiktok_data_2.json
print(f"Saving {len(new_items)} new items to {OUTPUT_FILE}...")
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(new_items, f, indent=2)

# Save updated scraped video IDs
with SCRAPED_VIDEOS_FILE.open("w", encoding="utf-8") as f:
    json.dump(list(scraped_video_ids), f, indent=2)

print(f"Done! {len(new_items)} new items saved to {OUTPUT_FILE}")
