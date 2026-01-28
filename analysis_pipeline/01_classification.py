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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def classify_videos(self, video_dir: Path, features_dir: Path, output_csv: Path, num_workers: int = None):
        """
        Classify all videos in a directory by niche using parallel processing.
        
        Args:
            video_dir: Directory containing video files
            features_dir: Directory containing feature JSON files
            output_csv: Path to output CSV file with classifications
            num_workers: Number of parallel workers (default: CPU count / 2)
        """
        # Ensure paths are resolved
        video_dir = Path(video_dir).resolve()
        features_dir = Path(features_dir).resolve()
        output_csv = Path(output_csv).resolve()
        
        if not video_dir.exists():
            print(f"Video directory not found: {video_dir}")
            return []

        # Set default worker count
        if num_workers is None:
            num_workers = max(1, os.cpu_count() // 2)
        
        video_files = sorted(video_dir.glob("**/*.mp4"))
        
        print(f"Classifying {len(video_files)} videos using {num_workers} parallel workers...")
        
        results = []
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(
                    self._classify_single_video, 
                    video_path, 
                    features_dir, 
                    video_dir
                ): video_path 
                for video_path in video_files
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_video):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        completed += 1
                        print(f"  [{completed}/{len(video_files)}] {result['filename']} → {result['niche']}")
                except Exception as e:
                    video_path = future_to_video[future]
                    print(f"  ❌ Error processing {video_path.name}: {e}")

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

    def _classify_single_video(self, video_path: Path, features_dir: Path, video_dir: Path) -> Dict:
        """
        Classify a single video. Designed to be called from thread pool.
        
        Args:
            video_path: Path to video file
            features_dir: Directory containing feature JSON files
            video_dir: Root video directory for relative paths
            
        Returns:
            dict: Classification result for this video
        """
        video_id = self.extract_video_id(video_path.name)
        features = self.load_feature_data(video_id, features_dir)
        text_content = self.get_text_from_features(features)
        
        classification = self.taxonomy.classify_content(text_content)
        niche = classification.get("category", "unknown")
        
        return {
            "video_id": video_id,
            "filename": video_path.name,
            "niche": niche,
            "relative_path": str(video_path.relative_to(video_dir)),
            "text_sample": text_content[:100] if text_content else ""
        }


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
