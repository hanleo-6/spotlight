#!/usr/bin/env python3
"""
Convert video analysis JSON to Gemini Video Generation format.
Extracts only the essential information needed for video generation,
excluding metadata like confidence scores, niche targeting details, etc.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def extract_video_gen_spec(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract video generation specification from analysis JSON.
    
    Args:
        analysis: Full video analysis JSON
    
    Returns:
        Simplified video generation specification
    """
    template = analysis.get("template_data", {})
    
    # Extract basic identity
    identity = template.get("template_identity", {})
    
    # Extract structure information
    structure = template.get("structure", {})
    
    # Extract visual system
    visual = template.get("visual_system", {})
    
    # Extract motion language
    motion = template.get("motion_language", {})
    
    # Extract technical execution
    technical = template.get("technical_execution", {})
    
    # Build the video generation spec
    video_spec = {
        "video_id": analysis.get("video_id"),
        "username": analysis.get("username"),
        
        # Template overview
        "template_name": identity.get("template_name"),
        "category": identity.get("template_category"),
        "personality": identity.get("personality"),
        
        # Story structure
        "story_structure": {
            "intro": structure.get("intro_characteristics"),
            "hook": structure.get("hook_strategy"),
            "content_flow": structure.get("content_flow", []),
            "outro": structure.get("outro_characteristics"),
            "pacing": structure.get("pacing_rhythm"),
            "transition_style": structure.get("transition_style"),
            "storytelling_type": template.get("content_formula", {}).get("storytelling_structure")
        },
        
        # Visual design
        "visual_design": {
            "layout": {
                "grid_system": visual.get("layout", {}).get("grid_system"),
                "subject_positioning": visual.get("layout", {}).get("subject_positioning"),
                "layer_hierarchy": visual.get("layout", {}).get("layer_hierarchy", [])
            },
            "color_system": {
                "primary": {
                    "purpose": visual.get("color_system", {}).get("primary_purpose"),
                    "appearance": visual.get("color_system", {}).get("primary_appearance")
                },
                "secondary": {
                    "purpose": visual.get("color_system", {}).get("secondary_purpose"),
                    "appearance": visual.get("color_system", {}).get("secondary_appearance")
                },
                "background_treatment": visual.get("color_system", {}).get("background_treatment"),
                "contrast_strategy": visual.get("color_system", {}).get("contrast_strategy")
            },
            "typography": {
                "headline_style": visual.get("typography", {}).get("headline_style"),
                "body_style": visual.get("typography", {}).get("body_style"),
                "alignment": visual.get("typography", {}).get("alignment_pattern"),
                "text_animation": visual.get("typography", {}).get("text_animation"),
                "uppercase_pattern": visual.get("typography", {}).get("uppercase_pattern")
            },
            "graphic_style": visual.get("graphic_style"),
            "shape_language": visual.get("shape_language")
        },
        
        # Motion and camera
        "motion_design": {
            "camera_behavior": motion.get("camera_behavior"),
            "subject_movement": motion.get("subject_movement_pattern"),
            "text_entry": motion.get("text_entry_style"),
            "text_exit": motion.get("text_exit_style"),
            "transitions": motion.get("transition_vocabulary", []),
            "animation_speed": motion.get("animation_speed")
        },
        
        # Technical settings
        "technical_settings": {
            "lighting_style": technical.get("lighting_style"),
            "color_grading": technical.get("color_grading"),
            "depth_of_field": technical.get("depth_of_field"),
            "camera_quality": technical.get("camera_quality_feel")
        },
        
        # Content format
        "content_format": {
            "primary_format": template.get("content_formula", {}).get("primary_format"),
            "format_mix_ratio": template.get("content_formula", {}).get("format_mix_ratio"),
            "text_visual_balance": template.get("content_formula", {}).get("text_visual_balance"),
            "information_density": template.get("content_formula", {}).get("information_density")
        }
    }
    
    return video_spec


def convert_file(input_path: Path, output_path: Path = None) -> bool:
    """
    Convert a video analysis JSON file to video generation spec.
    
    Args:
        input_path: Path to input JSON file
        output_path: Optional path to output file. If not provided, 
                     defaults to input_path with '_video_gen_spec' suffix
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load input JSON
        with open(input_path, 'r') as f:
            analysis = json.load(f)
        
        # Extract video generation spec
        video_spec = extract_video_gen_spec(analysis)
        
        # Determine output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_video_gen_spec.json"
        
        # Save output JSON
        with open(output_path, 'w') as f:
            json.dump(video_spec, f, indent=2)
        
        return True
    except Exception as e:
        print(f"✗ Error converting {input_path}: {e}")
        return False


def convert_directory(input_dir: Path, recursive: bool = True) -> None:
    """
    Convert all JSON files in a directory to video generation specs.
    
    Args:
        input_dir: Path to directory containing JSON files
        recursive: Whether to search subdirectories
    """
    # Find all JSON files
    if recursive:
        json_files = list(input_dir.rglob("*.json"))
    else:
        json_files = list(input_dir.glob("*.json"))
    
    # Filter out files that are already video_gen_spec files
    json_files = [f for f in json_files if not f.stem.endswith("_video_gen_spec")]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing {json_file.relative_to(input_dir)}...", end=" ")
        if convert_file(json_file):
            print("✓")
            successful += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Complete: {successful} successful, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Convert video analysis JSON to Gemini video generation format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input JSON file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output file (only for single file mode)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories (only for directory mode)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return 1
    
    # Check if input is a directory or file
    if input_path.is_dir():
        # Directory mode
        if args.output:
            print("Warning: --output is ignored in directory mode")
        convert_directory(input_path, recursive=not args.no_recursive)
    else:
        # Single file mode
        output_path = Path(args.output) if args.output else None
        if convert_file(input_path, output_path):
            print(f"✓ Converted analysis to video generation spec")
            output = output_path if output_path else input_path.parent / f"{input_path.stem}_video_gen_spec.json"
            print(f"  Input:  {input_path}")
            print(f"  Output: {output}")
        else:
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
