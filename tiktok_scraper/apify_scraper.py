import json
import os
from pathlib import Path
from apify_client import ApifyClient

# -------------------------------
# Configuration
# -------------------------------
API_TOKEN = os.getenv("APIFY_API_TOKEN", "")  # Set APIFY_API_TOKEN environment variable
SCRAPED_VIDEOS_FILE = Path("tiktok/scraped_videos.json")

# Load previously scraped video IDs
if SCRAPED_VIDEOS_FILE.exists():
    with SCRAPED_VIDEOS_FILE.open("r", encoding="utf-8") as f:
        scraped_video_ids = set(json.load(f))
else:
    scraped_video_ids = set()

# Initialize Apify client
client = ApifyClient(API_TOKEN)

# Actor input
run_input = {
    "hashtags": ["saastok", "apptok"],
    "resultsPerPage": 100,
    "profiles": None,
    "profileScrapeSections": ["videos"],
    "profileSorting": "latest",
    "excludePinnedPosts": False,
    "oldestPostDateUnified": None,
    "newestPostDate": None,
    "mostDiggs": None,
    "leastDiggs": None,
    "maxFollowersPerProfile": 0,
    "maxFollowingPerProfile": 0,
    "searchQueries": None,
    "searchSection": "",
    "maxProfilesPerQuery": 10,
    "searchSorting": "0",
    "searchDatePosted": "0",
    "postURLs": None,
    "scrapeRelatedVideos": False,
    "shouldDownloadVideos": False,
    "shouldDownloadCovers": False,
    "shouldDownloadSubtitles": False,
    "shouldDownloadSlideshowImages": False,
    "shouldDownloadAvatars": False,
    "shouldDownloadMusicCovers": False,
    "videoKvStoreIdOrName": None,
    "commentsPerPost": 30,
    "maxRepliesPerComment": 0,
    "proxyCountryCode": "None",
}

# Run the actor
run = client.actor("GdWCkxBtKWOsKjdch").call(run_input=run_input)

# Fetch results and filter out already scraped videos
new_items = []
dataset = client.dataset(run["defaultDatasetId"])

for item in dataset.iterate_items():
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

# Save updated scraped video IDs
with SCRAPED_VIDEOS_FILE.open("w", encoding="utf-8") as f:
    json.dump(list(scraped_video_ids), f, indent=2)
