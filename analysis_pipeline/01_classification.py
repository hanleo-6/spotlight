"""
Module 1: Video Classification
Classifies videos by niche using content analysis and taxonomy.
"""
import json
from pathlib import Path
import csv
import re
from collections import Counter
from typing import Dict, List

from niche_taxonomy import NicheTaxonomy


class VideoClassifier:
    """Classify videos by niche based on content and metadata."""
    
    def __init__(self, workspace_root: Path = None):
        """Initialize classifier with workspace root directory."""
        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parent.parent
        
        self.workspace_root = workspace_root
        self.taxonomy = NicheTaxonomy()
        
    def extract_video_id(self, filename: str) -> str:
        """Extract video ID from filename like 'Cutsg0uhk0l_calvincheungtc_score0.9124.mp4'."""
        name = filename.replace(".mp4", "")
        match = re.match(r"^([A-Za-z0-9_-]+?)_[^_]+_score[\d.]+$", name)
        if match:
            return match.group(1)
        return name

    def load_feature_data(self, video_id: str, features_dir: Path) -> dict:
        """Load feature/metadata from corresponding JSON file if it exists."""
        for json_file in features_dir.glob(f"{video_id}_*.json"):
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_text_from_features(self, features: dict) -> str:
        """Extract relevant text fields from feature data."""
        text_parts = []

        def _coerce(value) -> List[str]:
            if isinstance(value, str):
                return [value]
            elif isinstance(value, list):
                out = []
                for item in value:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        out.extend(_coerce(item))
                return out
            elif isinstance(value, dict):
                out = []
                for v in value.values():
                    out.extend(_coerce(v))
                return out
            return []

        for key in ["title", "desc", "text", "caption", "hashtags", "challenges"]:
            if key in features:
                text_parts.extend(_coerce(features[key]))

        return " ".join(text_parts)

    def classify_videos(self, video_dir: Path, features_dir: Path, output_csv: Path):
        """
        Classify all videos in a directory by niche.
        
        Args:
            video_dir: Directory containing video files
            features_dir: Directory containing feature JSON files
            output_csv: Path to output CSV file with classifications
        """
        if not video_dir.exists():
            print(f"Video directory not found: {video_dir}")
            return

        results = []
        video_files = sorted(video_dir.glob("**/*.mp4"))
        
        print(f"Classifying {len(video_files)} videos...")
        
        for video_path in video_files:
            video_id = self.extract_video_id(video_path.name)
            features = self.load_feature_data(video_id, features_dir)
            text_content = self.get_text_from_features(features)
            
            niche = self.taxonomy.classify_text(text_content)
            
            results.append({
                "video_id": video_id,
                "filename": video_path.name,
                "niche": niche,
                "relative_path": str(video_path.relative_to(video_dir)),
                "text_sample": text_content[:100] if text_content else ""
            })
            
            print(f"  {video_path.name} → {niche}")

        # Write results to CSV
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "filename", "niche", "relative_path", "text_sample"])
            writer.writeheader()
            writer.writerows(results)
        
        # Print summary
        niche_counts = Counter(r["niche"] for r in results)
        print(f"\nClassification Summary:")
        print(f"Total videos: {len(results)}")
        print("\nBy niche:")
        for niche, count in niche_counts.most_common():
            print(f"  {niche}: {count}")
        
        print(f"\nResults saved to: {output_csv}")
        
        return results


def main():
    """Standalone execution for best_media classification."""
    workspace_root = Path(__file__).resolve().parent.parent
    
    classifier = VideoClassifier(workspace_root)
    classifier.classify_videos(
        video_dir=workspace_root / "best_media",
        features_dir=workspace_root / "features",
        output_csv=workspace_root / "data" / "output" / "video_classifications.csv"
    )


if __name__ == "__main__":
    main()
