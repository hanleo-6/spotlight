#!/usr/bin/env python3
"""
Extract features from best_media metadata CSV and create JSON feature files.

This script reads the metadata.csv file and converts it into individual JSON
feature files for each video, matching the expected format for the pipeline.
"""

import csv
import json
from pathlib import Path
import re


def extract_video_id_from_shortcode(shortcode: str) -> str:
    """Extract video ID from Instagram shortcode."""
    return shortcode


def create_feature_json(row: dict) -> dict:
    """
    Convert a metadata row into a feature JSON structure.
    
    Args:
        row: A row from the metadata CSV
        
    Returns:
        Dictionary with feature data
    """
    features = {
        "video_id": extract_video_id_from_shortcode(row.get("shortcode", "")),
        "title": "",  # No title in metadata
        "desc": row.get("owner_username", "") + " - Instagram Video",
        "caption": "",  # No caption in metadata
        "text": f"Creator: {row.get('owner_username', '')}",
        "hashtags": [],
        "challenges": [],
        
        # Metadata from CSV
        "metadata": {
            "owner_username": row.get("owner_username", ""),
            "owner_profile_id": row.get("owner_profile_id", ""),
            "product_type": row.get("product_type", "clips"),
            "timestamp": row.get("timestamp", ""),
            "duration_sec": float(row.get("duration_sec", 0)) if row.get("duration_sec") else 0,
            "width": int(row.get("width", 0)) if row.get("width") else 0,
            "height": int(row.get("height", 0)) if row.get("height") else 0,
        },
        
        # Engagement metrics
        "engagement": {
            "video_view_count": int(row.get("video_view_count", 0)) if row.get("video_view_count") else 0,
            "likes_count": int(row.get("likes_count", 0)) if row.get("likes_count") else 0,
            "comments_count": int(row.get("comments_count", 0)) if row.get("comments_count") else 0,
            "followers_count": int(row.get("followers_count", 0)) if row.get("followers_count") else 0,
            "engagement_rate": float(row.get("engagement_rate", 0)) if row.get("engagement_rate") else 0,
            "view_rate": float(row.get("view_rate", 0)) if row.get("view_rate") else 0,
            "composite_score": float(row.get("composite_score", 0)) if row.get("composite_score") else 0,
        },
        
        # Storage info
        "storage": {
            "storage_path": row.get("storage_path", ""),
            "public_url": row.get("public_url", ""),
            "video_url": row.get("video_url", ""),
            "status": row.get("status", "done"),
        }
    }
    
    return features


def extract_features_from_metadata(metadata_csv_path: Path, output_dir: Path):
    """
    Extract features from metadata CSV and save as individual JSON files.
    
    Args:
        metadata_csv_path: Path to the metadata.csv file
        output_dir: Directory to save feature JSON files
    """
    
    if not metadata_csv_path.exists():
        print(f"Metadata file not found: {metadata_csv_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading metadata from: {metadata_csv_path}")
    print(f"Writing features to: {output_dir}")
    
    count = 0
    with metadata_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = create_feature_json(row)
            video_id = features["video_id"]
            owner = features["metadata"]["owner_username"]
            shortcode = row.get("shortcode", "")
            
            # Create filename: {shortcode}_{owner}_{composite_score}.json
            score = row.get("composite_score", "0")
            filename = f"{shortcode}_{owner}_score{score}.json"
            
            output_path = output_dir / filename
            
            with output_path.open("w", encoding="utf-8") as out_f:
                json.dump(features, out_f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ {filename}")
            count += 1
    
    print(f"\nSuccessfully created {count} feature JSON files")


def main():
    """Main entry point."""
    workspace_root = Path(__file__).resolve().parent.parent
    
    metadata_csv = workspace_root / "data" / "input" / "best_media" / "metadata.csv"
    features_dir = workspace_root / "features"
    
    extract_features_from_metadata(metadata_csv, features_dir)


if __name__ == "__main__":
    main()
