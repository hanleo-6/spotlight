"""
Engagement Metrics Collector for TikTok Creators

This program collects performance metrics for all creators of videos in a given input folder
using Apify TikTok Profile Scraper API. It extracts both profile and video metrics
and outputs them to data/output/metrics/ for further analysis.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
INPUT_DIR = Path("data/input/tiktok_vids")
OUTPUT_DIR = Path("data/output/metrics")
OUTPUT_FILE = OUTPUT_DIR / "creator_metrics.json"

def get_creator_folders(input_dir):
    """Get all creator folders from the input directory."""
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return []
    
    creator_folders = [f for f in input_dir.iterdir() if f.is_dir()]
    print(f"Found {len(creator_folders)} creator folders")
    return creator_folders

def scrape_creator_profile(username):
    """Scrape a creator's profile using Apify TikTok Profile Scraper."""
    print(f"Scraping profile for @{username}...")
    
    # Actor input for profile scraping
    run_input = {
        "profiles": [f"@{username}"],
        "resultsPerPage": 100,  # Get up to 100 videos
        "profileScrapeSections": ["videos"],
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
        "shouldDownloadSubtitles": False,
        "shouldDownloadSlideshowImages": False,
        "shouldDownloadAvatars": False,
        "shouldDownloadMusicCovers": False,
        "commentsPerPost": 0,
        "maxRepliesPerComment": 0,
    }
    
    # Run the actor
    curl_command = [
        "curl", "-s",
        "-X", "POST",
        f"https://api.apify.com/v2/acts/GdWCkxBtKWOsKjdch/runs?token={API_TOKEN}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(run_input)
    ]
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Error starting actor for @{username}: {result.stderr}")
            return None
        
        run_response = json.loads(result.stdout)
        run_id = run_response["data"]["id"]
        default_dataset_id = run_response["data"]["defaultDatasetId"]
        
        # Wait for the run to complete
        max_wait = 300  # 5 minutes max
        wait_time = 0
        while wait_time < max_wait:
            status_command = [
                "curl", "-s",
                f"https://api.apify.com/v2/actor-runs/{run_id}?token={API_TOKEN}"
            ]
            result = subprocess.run(status_command, capture_output=True, text=True, timeout=30)
            status_response = json.loads(result.stdout)
            status = status_response["data"]["status"]
            
            if status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"]:
                if status != "SUCCEEDED":
                    print(f"Actor run for @{username} finished with status: {status}")
                    return None
                break
            
            time.sleep(5)
            wait_time += 5
        
        if wait_time >= max_wait:
            print(f"Timeout waiting for @{username} scrape to complete")
            return None
        
        # Fetch results
        dataset_command = [
            "curl", "-s",
            f"https://api.apify.com/v2/datasets/{default_dataset_id}/items?token={API_TOKEN}"
        ]
        
        result = subprocess.run(dataset_command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Error fetching dataset for @{username}: {result.stderr}")
            return None
        
        items = json.loads(result.stdout)
        return items
        
    except Exception as e:
        print(f"Exception scraping @{username}: {e}")
        return None

def extract_metrics(items):
    """Extract relevant metrics from scraped data."""
    if not items:
        return None
    
    # Get profile metrics from first item's authorMeta
    first_item = items[0] if items else None
    if not first_item or "authorMeta" not in first_item:
        return None
    
    author_meta = first_item["authorMeta"]
    
    profile_metrics = {
        "username": author_meta.get("name"),
        "nickname": author_meta.get("nickName"),
        "followerCount": author_meta.get("fans", 0),  # "fans" is followers
        "videoCount": author_meta.get("video", 0),
        "totalLikes": author_meta.get("heart", 0),
        "verified": author_meta.get("verified", False),
        "signature": author_meta.get("signature", ""),
        # Account creation date is typically not available via Apify
        "profileUrl": author_meta.get("profileUrl"),
    }
    
    # Extract video metrics
    videos = []
    for item in items:
        video_meta = {
            "videoId": item.get("id"),
            "createTime": item.get("createTime"),
            "createTimeISO": item.get("createTimeISO"),
            "playCount": item.get("videoMeta", {}).get("playCount", 0),  # views
            "diggCount": item.get("diggCount", 0),  # likes
            "commentCount": item.get("commentCount", 0),
            "shareCount": item.get("shareCount", 0),
            "text": item.get("text", ""),
            "url": item.get("webVideoUrl"),
            "duration": item.get("videoMeta", {}).get("duration", 0),
        }
        videos.append(video_meta)
    
    return {
        "profile": profile_metrics,
        "videos": videos,
        "scraped_at": datetime.now().isoformat(),
        "total_videos_scraped": len(videos)
    }

def main():
    """Main function to collect metrics for all creators."""
    if not API_TOKEN:
        print("Error: APIFY_API_TOKEN environment variable not set")
        print("Please set it in a .env file or environment")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all creator folders
    creator_folders = get_creator_folders(INPUT_DIR)
    if not creator_folders:
        print("No creator folders found")
        return
    
    # Load existing metrics if available
    all_metrics = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            all_metrics = json.load(f)
        print(f"Loaded {len(all_metrics)} existing metrics")
    
    # Process each creator
    for i, folder in enumerate(creator_folders, 1):
        username = folder.name
        print(f"\n[{i}/{len(creator_folders)}] Processing @{username}")
        
        # Skip if already scraped recently (within 24 hours)
        if username in all_metrics:
            scraped_at = datetime.fromisoformat(all_metrics[username].get("scraped_at", "2000-01-01"))
            hours_since_scrape = (datetime.now() - scraped_at).total_seconds() / 3600
            if hours_since_scrape < 24:
                print(f"  Skipping (scraped {hours_since_scrape:.1f} hours ago)")
                continue
        
        # Scrape creator profile and videos
        items = scrape_creator_profile(username)
        if items:
            metrics = extract_metrics(items)
            if metrics:
                all_metrics[username] = metrics
                print(f"  ✓ Collected metrics: {metrics['profile']['followerCount']} followers, {len(metrics['videos'])} videos")
                
                # Save after each creator (in case of interruption)
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_metrics, f, indent=2)
            else:
                print(f"  ✗ Failed to extract metrics")
        else:
            print(f"  ✗ Failed to scrape")
        
        # Rate limiting - wait between requests
        if i < len(creator_folders):
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"Metrics collection complete!")
    print(f"Total creators processed: {len(all_metrics)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
