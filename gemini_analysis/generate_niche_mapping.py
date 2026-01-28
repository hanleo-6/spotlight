#!/usr/bin/env python3
"""Generate a username-to-niche mapping template from gemini_inputs directories."""

import json
from pathlib import Path

GEMINI_INPUTS = Path("data/output/gemini_inputs")
OUTPUT_FILE = Path("data/username_niche_mapping.json")

def main():
    # Get all username directories
    usernames = sorted([d.name for d in GEMINI_INPUTS.iterdir() if d.is_dir()])
    
    print(f"Found {len(usernames)} usernames")
    
    # Create mapping template
    mapping = {
        "_instructions": "Replace 'TO_CLASSIFY' with actual niches. Common niches: SaaS, E-commerce, Education, Fitness, Finance, Food, Travel, Fashion, Entertainment, Personal Brand, Content Creation, Marketing Tools, AI Tools, Productivity, etc.",
        "_note": f"Total usernames to classify: {len(usernames)}"
    }
    
    for username in usernames:
        mapping[username] = "TO_CLASSIFY"
    
    # Save template
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✓ Mapping template saved to {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"1. Edit {OUTPUT_FILE}")
    print(f"2. Replace 'TO_CLASSIFY' with actual niches for each username")
    print(f"3. Run the analyzer again: python gemini_analysis/gemini_output_analyser.py")

if __name__ == "__main__":
    main()
