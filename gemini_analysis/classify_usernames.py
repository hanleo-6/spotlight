#!/usr/bin/env python3
"""
Generate username-to-niche mapping using the niche taxonomy classifier.
Reads feature files and classifies each username's content.
"""

import json
from pathlib import Path
import sys
from collections import defaultdict

# Add analysis_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis_pipeline"))

from niche_taxonomy import NicheTaxonomy
from classification import VideoClassifier

FEATURES_DIR = Path("features")
GEMINI_INPUTS = Path("data/output/gemini_inputs")
OUTPUT_FILE = Path("data/username_niche_mapping.json")


def main():
    """Generate username-to-niche mapping using content classification."""
    
    if not FEATURES_DIR.exists():
        print(f"❌ Features directory not found: {FEATURES_DIR}")
        return
    
    if not GEMINI_INPUTS.exists():
        print(f"❌ Gemini inputs directory not found: {GEMINI_INPUTS}")
        return
    
    # Initialize classifier
    classifier = VideoClassifier()
    taxonomy = NicheTaxonomy()
    
    # Get all usernames
    usernames = sorted([d.name for d in GEMINI_INPUTS.iterdir() if d.is_dir()])
    print(f"Found {len(usernames)} usernames to classify...")
    
    # Classify each username
    username_niches = {}
    username_details = {}
    
    for i, username in enumerate(usernames, 1):
        print(f"\n[{i}/{len(usernames)}] Classifying: {username}")
        
        # Find all feature files for this username
        feature_files = list(FEATURES_DIR.glob(f"*_{username}_score*.json"))
        
        if not feature_files:
            print(f"  ⚠️  No feature files found, using 'unknown'")
            username_niches[username] = "unknown"
            continue
        
        # Aggregate text from all videos for this username
        all_text = []
        for feature_file in feature_files[:5]:  # Sample first 5 videos
            try:
                with open(feature_file, "r") as f:
                    features = json.load(f)
                text = classifier.get_text_from_features(features)
                if text:
                    all_text.append(text)
            except Exception as e:
                print(f"  ⚠️  Error reading {feature_file.name}: {e}")
        
        if not all_text:
            print(f"  ⚠️  No text content found, using 'unknown'")
            username_niches[username] = "unknown"
            continue
        
        # Classify combined text
        combined_text = " ".join(all_text)
        classification = taxonomy.classify_content(combined_text)
        
        # Use full_path or construct from parts
        if classification.get("full_path"):
            niche = classification["full_path"]
        elif classification.get("micro_niche"):
            niche = f"{classification['category']} > {classification['subcategory']} > {classification['micro_niche']}"
        elif classification.get("subcategory"):
            niche = f"{classification['category']} > {classification['subcategory']}"
        else:
            niche = classification['category']
        
        username_niches[username] = niche
        username_details[username] = {
            "niche": niche,
            "confidence": classification.get("confidence", 0.0),
            "matched_keywords": classification.get("matched_keywords", []),
            "sample_count": len(feature_files)
        }
        
        print(f"  ✓ {niche} (confidence: {classification.get('confidence', 0):.2f})")
        if classification.get("matched_keywords"):
            print(f"    Keywords: {', '.join(classification['matched_keywords'][:5])}")
    
    # Save mapping
    mapping = {
        "_info": "Auto-generated username-to-niche mapping using NicheTaxonomy classifier",
        "_total_usernames": len(usernames),
        "_classification_date": "2026-01-26"
    }
    mapping.update(username_niches)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    
    # Save detailed report
    details_file = OUTPUT_FILE.parent / "username_niche_details.json"
    with open(details_file, "w") as f:
        json.dump(username_details, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY")
    print("="*80)
    
    niche_counts = defaultdict(int)
    for niche in username_niches.values():
        # Group by top-level category
        top_level = niche.split(" > ")[0]
        niche_counts[top_level] += 1
    
    print(f"\nTotal usernames: {len(usernames)}")
    print(f"\nBy top-level category:")
    for niche, count in sorted(niche_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {niche}: {count}")
    
    print(f"\n✓ Mapping saved to: {OUTPUT_FILE}")
    print(f"✓ Detailed report saved to: {details_file}")
    print(f"\nNext step: Run the analyzer")
    print(f"  /Users/ying/Documents/hameo/spotlight/.venv/bin/python gemini_analysis/gemini_output_analyser.py")


if __name__ == "__main__":
    main()
