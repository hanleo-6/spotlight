# TikTok Data Scraping Tools

This folder contains tools specifically for scraping and downloading TikTok video data.

## Files

### Core Scraping Tools

- **`apify_scraper.py`** - Uses Apify API to scrape TikTok metadata (hashtags, profiles, etc.)
- **`tiktok_downloader.py`** - Downloads TikTok videos using yt-dlp
- **`temp.py`** - Temporary/test scripts

### Configuration Files

- **`credentials.json`** - API credentials and configuration
- **`cookies.txt`** - Browser cookies for authenticated scraping
- **`scraped_videos.json`** - Tracking file for previously scraped video IDs

### Data Files

- **`tiktok_data_1.json`** - Scraped metadata and video information

## Usage

### 1. Scrape TikTok Metadata

```bash
python apify_scraper.py
```

This will:
- Use Apify API to scrape TikTok posts by hashtag or profile
- Save metadata to `tiktok_data_1.json`
- Track scraped video IDs to avoid duplicates

### 2. Download Videos

```bash
python tiktok_downloader.py
```

This will:
- Read URLs from `tiktok_data_1.json`
- Download videos to `../tiktok_vids/` organized by uploader
- Save metadata and thumbnails
- Use cookies for authentication

## Data Flow

```
[Apify API] → apify_scraper.py → tiktok_data_1.json
                                        ↓
                              tiktok_downloader.py
                                        ↓
                            ../tiktok_vids/<uploader>/<video_id>.mp4
```

## Integration with Analysis Pipeline

Once videos are downloaded, use the analysis pipeline:

```bash
cd ../analysis_pipeline
python pipeline_driver.py --video-dir ../tiktok_vids
```

See `../analysis_pipeline/README.md` for details.

## Configuration

### Apify Setup
1. Get API token from apify.com
2. Update `API_TOKEN` in `apify_scraper.py`
3. Configure hashtags/profiles to scrape

### Cookies for Downloads
1. Export cookies from browser using extension (e.g., "Get cookies.txt")
2. Save to `cookies.txt`
3. Required for some videos that need authentication

## Notes

- Respect TikTok's rate limits and terms of service
- Use delays between requests to avoid blocking
- Keep credentials secure (don't commit to git)
- Monitor scraping logs for errors

## Troubleshooting

**No videos downloaded:**
- Check cookies are fresh (regenerate if expired)
- Verify URLs in tiktok_data_1.json are valid
- Check internet connection

**Apify errors:**
- Verify API token is valid
- Check account has sufficient credits
- Review Apify dashboard for run logs

**Rate limiting:**
- Increase sleep intervals in downloaders
- Reduce batch sizes
- Use proxies if needed
