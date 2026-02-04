"""
Engagement Metrics Analyzer for TikTok Creators

This program analyzes engagement metric data to identify top N creators
in a niche by various engagement metrics. It calculates engagement rates
and provides comprehensive rankings.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configuration
METRICS_FILE = Path("data/output/metrics/creator_metrics.json")
OUTPUT_DIR = Path("data/output/metrics")

def load_metrics():
    """Load creator metrics from JSON file."""
    if not METRICS_FILE.exists():
        print(f"Error: Metrics file not found at {METRICS_FILE}")
        print("Please run engagement_ranker.py first to collect metrics")
        return None
    
    with open(METRICS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_engagement_rate(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Calculate various engagement rates for a creator."""
    profile = metrics.get("profile", {})
    videos = metrics.get("videos", [])
    
    follower_count = profile.get("followerCount", 0)
    if follower_count == 0:
        follower_count = 1  # Avoid division by zero
    
    # Calculate aggregate video metrics
    total_views = sum(v.get("playCount", 0) for v in videos)
    total_likes = sum(v.get("diggCount", 0) for v in videos)
    total_comments = sum(v.get("commentCount", 0) for v in videos)
    total_shares = sum(v.get("shareCount", 0) for v in videos)
    
    num_videos = len(videos)
    avg_views = total_views / num_videos if num_videos > 0 else 0
    avg_likes = total_likes / num_videos if num_videos > 0 else 0
    avg_comments = total_comments / num_videos if num_videos > 0 else 0
    avg_shares = total_shares / num_videos if num_videos > 0 else 0
    
    # Engagement rates
    engagement_rate = ((total_likes + total_comments + total_shares) / total_views * 100) if total_views > 0 else 0
    like_rate = (total_likes / total_views * 100) if total_views > 0 else 0
    comment_rate = (total_comments / total_views * 100) if total_views > 0 else 0
    share_rate = (total_shares / total_views * 100) if total_views > 0 else 0
    
    # Follower engagement rate
    follower_engagement = (avg_views / follower_count * 100) if follower_count > 0 else 0
    
    return {
        "total_views": total_views,
        "total_likes": total_likes,
        "total_comments": total_comments,
        "total_shares": total_shares,
        "avg_views": avg_views,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "avg_shares": avg_shares,
        "engagement_rate": engagement_rate,
        "like_rate": like_rate,
        "comment_rate": comment_rate,
        "share_rate": share_rate,
        "follower_engagement": follower_engagement,
        "num_videos": num_videos,
        "follower_count": follower_count
    }

def analyze_by_metric(all_metrics: Dict, metric_key: str, top_n: int = 20) -> List[Dict]:
    """Analyze and rank creators by a specific metric."""
    rankings = []
    
    for username, metrics in all_metrics.items():
        engagement = calculate_engagement_rate(metrics)
        profile = metrics.get("profile", {})
        
        creator_data = {
            "username": username,
            "nickname": profile.get("nickname", ""),
            "follower_count": profile.get("followerCount", 0),
            "video_count": profile.get("videoCount", 0),
            "verified": profile.get("verified", False),
            "profile_url": profile.get("profileUrl", ""),
            **engagement
        }
        
        rankings.append(creator_data)
    
    # Sort by the specified metric
    rankings.sort(key=lambda x: x.get(metric_key, 0), reverse=True)
    
    return rankings[:top_n]

def filter_by_niche(all_metrics: Dict, niche_keywords: List[str]) -> Dict:
    """Filter creators by niche keywords in their bio or video content."""
    filtered = {}
    
    for username, metrics in all_metrics.items():
        profile = metrics.get("profile", {})
        signature = profile.get("signature", "").lower()
        
        # Check if any keyword appears in bio
        if any(keyword.lower() in signature for keyword in niche_keywords):
            filtered[username] = metrics
            continue
        
        # Check video descriptions
        videos = metrics.get("videos", [])
        for video in videos:
            text = video.get("text", "").lower()
            if any(keyword.lower() in text for keyword in niche_keywords):
                filtered[username] = metrics
                break
    
    return filtered

def print_rankings(rankings: List[Dict], title: str, metric_key: str):
    """Print rankings in a formatted table."""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'Username':<20} {'Followers':<12} {'Videos':<8} {metric_key.replace('_', ' ').title():<20} {'✓' if 'verified' in rankings[0] else '':<3}")
    print(f"{'-'*100}")
    
    for i, creator in enumerate(rankings, 1):
        username = creator['username'][:19]
        followers = f"{creator['follower_count']:,}"
        videos = creator['num_videos']
        metric_value = creator[metric_key]
        verified = "✓" if creator.get('verified') else ""
        
        # Format metric value based on type
        if 'rate' in metric_key or 'engagement' in metric_key:
            metric_str = f"{metric_value:.2f}%"
        elif isinstance(metric_value, float):
            metric_str = f"{metric_value:,.0f}"
        else:
            metric_str = f"{metric_value:,}"
        
        print(f"{i:<6} {username:<20} {followers:<12} {videos:<8} {metric_str:<20} {verified:<3}")
    
    print(f"{'='*100}\n")

def save_analysis(rankings: Dict[str, List[Dict]], filename: str = "engagement_analysis.json"):
    """Save analysis results to JSON file."""
    output_file = OUTPUT_DIR / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "rankings": rankings
        }, f, indent=2)
    
    print(f"Analysis saved to: {output_file}")

def main():
    """Main analysis function."""
    print("Loading creator metrics...")
    all_metrics = load_metrics()
    
    if not all_metrics:
        return
    
    print(f"Loaded metrics for {len(all_metrics)} creators\n")
    
    # Analyze by different metrics
    analysis_configs = [
        ("engagement_rate", "Top Creators by Overall Engagement Rate"),
        ("avg_views", "Top Creators by Average Views per Video"),
        ("total_views", "Top Creators by Total Views"),
        ("follower_engagement", "Top Creators by Follower Engagement Rate"),
        ("avg_likes", "Top Creators by Average Likes per Video"),
        ("like_rate", "Top Creators by Like Rate"),
        ("comment_rate", "Top Creators by Comment Rate"),
        ("share_rate", "Top Creators by Share Rate"),
    ]
    
    all_rankings = {}
    
    for metric_key, title in analysis_configs:
        rankings = analyze_by_metric(all_metrics, metric_key, top_n=20)
        all_rankings[metric_key] = rankings
        print_rankings(rankings, title, metric_key)
    
    # Save all rankings
    save_analysis(all_rankings)
    
    # Optional: Filter by niche
    print("\n" + "="*100)
    print("NICHE-SPECIFIC ANALYSIS")
    print("="*100)
    
    # Example: AI/Tech niche
    ai_keywords = ["ai", "artificial intelligence", "machine learning", "tech", "coding", "developer"]
    ai_metrics = filter_by_niche(all_metrics, ai_keywords)
    print(f"\nFound {len(ai_metrics)} creators in AI/Tech niche")
    
    if ai_metrics:
        ai_rankings = analyze_by_metric(ai_metrics, "engagement_rate", top_n=10)
        print_rankings(ai_rankings, "Top AI/Tech Creators by Engagement Rate", "engagement_rate")
        
        # Save niche-specific analysis
        save_analysis({"ai_tech_niche": ai_rankings}, "ai_tech_engagement_analysis.json")

def analyze_custom_niche(niche_name: str, keywords: List[str], metric: str = "engagement_rate", top_n: int = 20):
    """Analyze a custom niche with specified keywords."""
    all_metrics = load_metrics()
    if not all_metrics:
        return None
    
    niche_metrics = filter_by_niche(all_metrics, keywords)
    print(f"\nFound {len(niche_metrics)} creators in {niche_name} niche")
    
    if not niche_metrics:
        print(f"No creators found for {niche_name}")
        return None
    
    rankings = analyze_by_metric(niche_metrics, metric, top_n=top_n)
    print_rankings(rankings, f"Top {niche_name} Creators by {metric.replace('_', ' ').title()}", metric)
    
    # Save analysis
    filename = f"{niche_name.lower().replace(' ', '_')}_analysis.json"
    save_analysis({niche_name: rankings}, filename)
    
    return rankings

if __name__ == "__main__":
    import sys
    
    # Support command-line arguments for custom niche analysis
    if len(sys.argv) > 1:
        if sys.argv[1] == "--niche" and len(sys.argv) >= 4:
            niche_name = sys.argv[2]
            keywords = sys.argv[3].split(",")
            metric = sys.argv[4] if len(sys.argv) > 4 else "engagement_rate"
            top_n = int(sys.argv[5]) if len(sys.argv) > 5 else 20
            
            print(f"Analyzing niche: {niche_name}")
            print(f"Keywords: {', '.join(keywords)}")
            analyze_custom_niche(niche_name, keywords, metric, top_n)
        else:
            print("Usage:")
            print("  python engagement_analyzer.py")
            print("  python engagement_analyzer.py --niche 'niche_name' 'keyword1,keyword2' [metric] [top_n]")
            print("\nExample:")
            print("  python engagement_analyzer.py --niche 'Gaming' 'game,gaming,gamer' engagement_rate 15")
    else:
        main()
