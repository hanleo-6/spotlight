import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local paths
ALIGNMENT_DIR = Path("data/output/frame_mappings/tiktok_vids")
OUTPUT_DIR = Path("data/output/gemini_inputs/tiktok_vids")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GCS bucket path for storing frame URIs
GCS_BUCKET = "gs://spotlight-analysis-videos-tiktok-ew4/frames"


def generate_gemini_input_from_alignment(alignment_file: Path) -> bool:
    """
    Convert a frame alignment JSON to Gemini input format.
    
    Args:
        alignment_file: Path to frame mapping JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load aligned transcript JSON
        with open(alignment_file, "r", encoding='utf-8') as f:
            alignment_data = json.load(f)
        
        if not alignment_data:
            logger.warning(f"Empty alignment file: {alignment_file}")
            return False
        
        # Get relative path for output structure
        relative_path = alignment_file.relative_to(ALIGNMENT_DIR)
        output_subdir = OUTPUT_DIR / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate Gemini input format
        gemini_input = []
        
        for frame_entry in alignment_data:
            # Get transcript segments and combine text
            transcript_segments = frame_entry.get("transcript_segments", [])
            
            if not transcript_segments:
                # Use context segments if no direct match
                transcript_segments = []
                if frame_entry.get("previous_segment"):
                    transcript_segments.append(frame_entry["previous_segment"])
                if frame_entry.get("next_segment"):
                    transcript_segments.append(frame_entry["next_segment"])
            
            if not transcript_segments:
                logger.debug(f"No transcript segments for {frame_entry.get('frame_filename')}")
                continue
            
            transcript_text = " ".join(seg["text"] for seg in transcript_segments if seg and seg.get("text"))
            
            if not transcript_text.strip():
                logger.debug(f"Empty transcript text for {frame_entry.get('frame_filename')}")
                continue
            
            # Build GCS URI - construct path from alignment file location
            # e.g., frame_mappings/channel/video_id/video_id_frame_mapping.json
            # -> gs://bucket/frames/channel/video_id/frame_filename.jpg
            frame_filename = frame_entry.get("frame_filename", "")
            frame_uri = f"{GCS_BUCKET}/{relative_path.parent}/{frame_filename}"
            
            gemini_input.append({
                "frame": frame_uri,
                "transcript": transcript_text.strip()
            })
        
        if not gemini_input:
            logger.warning(f"No valid gemini inputs generated from {alignment_file}")
            return False
        
        # Write output JSON
        output_filename = alignment_file.stem.replace("_frame_mapping", "") + ".json"
        output_file = output_subdir / output_filename
        
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(gemini_input, f, indent=2)
        
        logger.info(f"Generated Gemini input for {relative_path.parent}: {len(gemini_input)} frames")
        logger.info(f"Output: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing {alignment_file}: {e}")
        return False


def main():
    """Process all frame alignment files and generate Gemini inputs."""
    if not ALIGNMENT_DIR.exists():
        logger.error(f"Alignment directory not found: {ALIGNMENT_DIR}")
        return
    
    # Find all frame mapping JSON files
    alignment_files = sorted(ALIGNMENT_DIR.rglob("*_frame_mapping.json"))
    
    if not alignment_files:
        logger.warning(f"No frame mapping files found in {ALIGNMENT_DIR}")
        return
    
    logger.info(f"Found {len(alignment_files)} frame mapping files to process")
    
    successful = 0
    failed = 0
    
    for alignment_file in alignment_files:
        if generate_gemini_input_from_alignment(alignment_file):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
