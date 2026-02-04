# TikTok Creator Engagement Analysis

This module provides tools to collect and analyze engagement metrics for TikTok creators using the Apify API.

## Files

### 1. `engagement_ranker.py`
Collects performance metrics for all creators whose videos are in the input folder.

**Metrics Collected:**

Profile Metrics:
- `followerCount`: Number of followers
- `videoCount`: Total number of videos posted
- `totalLikes`: Total likes across all videos
- `verified`: Verification status
- `signature`: Bio/description

Video Metrics (per video):
- `playCount`: Number of views
- `diggCount`: Number of likes
- `commentCount`: Number of comments
- `shareCount`: Number of shares
- `createTime`: Unix timestamp of creation
- `createTimeISO`: ISO format timestamp

**Output:** `data/output/metrics/creator_metrics.json`

### 2. `engagement_analyzer.py`
Analyzes the collected metrics to identify top N creators by various engagement metrics.

**Features:**
- Calculates engagement rates (overall, like rate, comment rate, share rate)
- Ranks creators by multiple metrics
- Filters creators by niche keywords
- Generates comprehensive analysis reports

**Output:** 
- `data/output/metrics/engagement_analysis.json`
- Niche-specific analysis files (e.g., `ai_tech_engagement_analysis.json`)

## Setup

1. Install required dependencies:
```bash
pip install python-dotenv
```

2. Set up your Apify API token:
   - Create a `.env` file in the project root
   - Add your token: `APIFY_API_TOKEN=your_token_here`
   - Get your token from: https://console.apify.com/account/integrations

3. Ensure your video folders are in `data/input/tiktok_vids/`:
   - Each folder should be named after the creator's username
   - Example: `data/input/tiktok_vids/username/`

## Usage

### Collecting Metrics

Run the engagement ranker to collect metrics for all creators:

```bash
python video_scraping/engagement_ranker.py
```

This will:
- Scan all folders in `data/input/tiktok_vids/`
- Use Apify to scrape each creator's profile and video metrics
- Save results to `data/output/metrics/creator_metrics.json`
- Skip creators scraped within the last 24 hours
- Auto-save after each creator (safe for interruptions)

**Note:** This may take a while if you have many creators. The script includes rate limiting and automatic retries.

### Analyzing Metrics

After collecting metrics, run the analyzer:

```bash
python video_scraping/engagement_analyzer.py
```

This will generate rankings by:
- Overall engagement rate
- Average views per video
- Total views
- Follower engagement rate
- Average likes, comments, shares
- Like rate, comment rate, share rate

### Custom Niche Analysis

To analyze a specific niche:

```bash
python video_scraping/engagement_analyzer.py --niche 'Niche Name' 'keyword1,keyword2,keyword3' [metric] [top_n]
```

**Examples:**

```bash
# Gaming niche, top 15 by engagement rate
python video_scraping/engagement_analyzer.py --niche 'Gaming' 'game,gaming,gamer' engagement_rate 15

# Fitness niche, top 10 by average views
python video_scraping/engagement_analyzer.py --niche 'Fitness' 'fitness,workout,gym' avg_views 10

# AI/Tech niche, top 20 by total views
python video_scraping/engagement_analyzer.py --niche 'AI Tech' 'ai,tech,coding' total_views 20
```

## Metrics Explained

### Engagement Rates

- **Engagement Rate**: `(likes + comments + shares) / views * 100`
  - Measures overall audience interaction
  - Higher = more engaged audience

- **Like Rate**: `likes / views * 100`
  - Percentage of viewers who liked
  - Industry average: ~3-9%

- **Comment Rate**: `comments / views * 100`
  - Percentage of viewers who commented
  - Indicates content that sparks discussion

- **Share Rate**: `shares / views * 100`
  - Percentage of viewers who shared
  - Indicates viral potential

- **Follower Engagement**: `avg_views / followers * 100`
  - How many followers watch each video
  - Higher = more loyal audience

## Output Format

### creator_metrics.json
```json
{
  "username": {
    "profile": {
      "username": "creator_name",
      "nickname": "Display Name",
      "followerCount": 50000,
      "videoCount": 150,
      "totalLikes": 1000000,
      "verified": false
    },
    "videos": [
      {
        "videoId": "1234567890",
        "createTime": 1749643870,
        "playCount": 100000,
        "diggCount": 5000,
        "commentCount": 200,
        "shareCount": 100
      }
    ],
    "scraped_at": "2026-02-02T12:00:00",
    "total_videos_scraped": 100
  }
}
```

### engagement_analysis.json
```json
{
  "generated_at": "2026-02-02T12:00:00",
  "rankings": {
    "engagement_rate": [
      {
        "username": "creator_name",
        "follower_count": 50000,
        "engagement_rate": 8.5,
        "avg_views": 150000,
        "total_views": 15000000,
        ...
      }
    ],
    "avg_views": [...],
    ...
  }
}
```

## Tips

1. **Rate Limits**: The Apify actor has rate limits. The script includes delays between requests.

2. **Costs**: Check your Apify usage limits. Each profile scrape consumes compute units.

3. **Incremental Updates**: Re-running `engagement_ranker.py` will only update creators not scraped in the last 24 hours.

4. **Filtering**: Use niche analysis to focus on specific creator categories.

5. **Best Metrics**: For finding viral potential, focus on:
   - Share rate (viral content)
   - Engagement rate (audience quality)
   - Follower engagement (audience loyalty)

## Troubleshooting

**"APIFY_API_TOKEN environment variable not set"**
- Create a `.env` file with your token
- Or export it: `export APIFY_API_TOKEN=your_token`

**"No creator folders found"**
- Check that `data/input/tiktok_vids/` exists
- Ensure folders are named after creator usernames

**"Failed to scrape"**
- Check your Apify account has available compute units
- Verify the username exists on TikTok
- Check internet connection

**Slow performance**
- This is normal - scraping takes time
- Results are saved after each creator
- You can interrupt and resume later
