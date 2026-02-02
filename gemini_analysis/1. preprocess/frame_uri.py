"""
Frame-to-Transcript Alignment Tool
Maps video frames to corresponding transcript segments based on timestamp alignment.
Handles edge cases and provides context from adjacent segments.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_timestamp_from_filename(filename: str, sample_interval: int = 3) -> Optional[float]:
    """
    Extract timestamp from frame filename.
    Supports formats like: frame_t0000.jpg (explicit timestamp) or keyframe_00000.jpg (sequential index).
    For sequential index, calculates timestamp as: index * sample_interval
    
    Args:
        filename: Frame filename
        sample_interval: Sampling interval in seconds for sequential frames (default: 3)
        
    Returns:
        Timestamp in seconds or None if not found
    """
    # Try to extract 't' followed by digits pattern (e.g., t0006 for explicit timestamp)
    match = re.search(r't(\d+)', filename)
    if match:
        return float(match.group(1))
    
    # Try to extract sequential index from keyframe_XXXXX.jpg format
    match = re.search(r'keyframe_(\d+)', filename)
    if match:
        frame_index = int(match.group(1))
        return float(frame_index * sample_interval)
    
    # Try to extract leading digits (for formats like 00000.jpg)
    match = re.search(r'^(\d+)', filename)
    if match:
        frame_index = int(match.group(1))
        return float(frame_index * sample_interval)
    
    return None


def load_transcript(transcript_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load transcript from JSON file.
    
    Args:
        transcript_path: Path to transcript JSON file
        
    Returns:
        Transcript data or None if file not found
    """
    if not transcript_path.exists():
        logger.warning(f"Transcript file not found: {transcript_path}")
        return None
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading transcript {transcript_path}: {e}")
        return None


def find_matching_segments(frame_timestamp: float, segments: List[Dict[str, Any]], 
                          include_context: bool = True) -> Dict[str, Any]:
    """
    Find transcript segments that match a frame timestamp.
    
    Args:
        frame_timestamp: Timestamp of the frame in seconds
        segments: List of transcript segments with start, end, and text
        include_context: Whether to include previous and next segments for context
        
    Returns:
        Dict with matched segments and optional context
    """
    result = {
        "matching_segments": [],
        "previous_segment": None,
        "next_segment": None
    }
    
    if not segments:
        return result
    
    prev_segment = None
    next_segment = None
    
    for i, segment in enumerate(segments):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        
        # Check if frame timestamp falls within segment
        if start <= frame_timestamp <= end:
            result["matching_segments"].append(segment)
        elif frame_timestamp < start and next_segment is None:
            next_segment = segment
        elif frame_timestamp > end:
            prev_segment = segment
    
    if include_context:
        result["previous_segment"] = prev_segment
        result["next_segment"] = next_segment
    
    return result


def align_frames_to_transcript(frames_dir: str, transcript_path: str, 
                               output_path: str, include_context: bool = True,
                               frame_uri_prefix: str = "frames") -> bool:
    """
    Align video frames to transcript segments and output mapping as JSON.
    
    Args:
        frames_dir: Path to directory containing frame images
        transcript_path: Path to transcript JSON file
        output_path: Path to output JSON file
        include_context: Whether to include previous/next segments
        frame_uri_prefix: Prefix for frame URIs in output
        
    Returns:
        True if successful, False otherwise
    """
    frames_path = Path(frames_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load transcript
    transcript_data = load_transcript(Path(transcript_path))
    if not transcript_data:
        logger.error(f"Failed to load transcript from {transcript_path}")
        return False
    
    segments = transcript_data.get("segments", [])
    if not segments:
        logger.warning(f"No segments found in transcript {transcript_path}")
        return False
    
    # Find all frame files
    frame_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    frame_files = [
        f for f in sorted(frames_path.iterdir())
        if f.is_file() and f.suffix.lower() in frame_extensions
    ]
    
    if not frame_files:
        logger.warning(f"No frame files found in {frames_dir}")
        return False
    
    logger.info(f"Processing {len(frame_files)} frames from {frames_path.name}")
    logger.info(f"Transcript has {len(segments)} segments")
    if segments:
        logger.info(f"Transcript time range: {segments[0].get('start', 0)}s to {segments[-1].get('end', 0)}s")
    
    # Map frames to transcript segments
    frame_mappings = []
    matched_count = 0
    
    for frame_file in frame_files:
        frame_timestamp = extract_timestamp_from_filename(frame_file.name)
        
        if frame_timestamp is None:
            logger.warning(f"Could not extract timestamp from {frame_file.name}")
            continue
        
        # Find matching segments
        segment_match = find_matching_segments(frame_timestamp, segments, include_context)
        
        # Create frame mapping
        mapping = {
            "frame_filename": frame_file.name,
            "frame_timestamp": frame_timestamp,
            "frame_uri": f"{frame_uri_prefix}/{frame_file.name}",
            "transcript_segments": segment_match["matching_segments"]
        }
        
        # Add context if requested
        if include_context:
            mapping["previous_segment"] = segment_match["previous_segment"]
            mapping["next_segment"] = segment_match["next_segment"]
        
        if segment_match["matching_segments"]:
            matched_count += 1
        
        frame_mappings.append(mapping)
    
    # Write output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(frame_mappings, f, indent=2)
        logger.info(f"Mapped {matched_count}/{len(frame_mappings)} frames to transcript segments")
        logger.info(f"Output saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error writing output file {output_file}: {e}")
        return False


def process_all_videos(frames_base_dir: str, transcripts_base_dir: str, 
                      output_base_dir: str, include_context: bool = True) -> None:
    """
    Automatically match keyframes with transcripts for all videos in directories.
    Handles cases where transcript directories are at a higher level than frame directories.
    
    Args:
        frames_base_dir: Base directory containing subdirectories for each video's frames
        transcripts_base_dir: Base directory containing subdirectories for each video's transcripts
        output_base_dir: Base directory where output mappings will be saved
        include_context: Whether to include previous/next segments
    """
    frames_base = Path(frames_base_dir)
    transcripts_base = Path(transcripts_base_dir)
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    if not frames_base.exists():
        logger.error(f"Frames directory not found: {frames_base_dir}")
        return
    
    if not transcripts_base.exists():
        logger.error(f"Transcripts directory not found: {transcripts_base_dir}")
        return
    
    # Find all directories that contain frame images
    frame_dirs = set()
    for img_file in frames_base.rglob("*.jpg"):
        frame_dirs.add(img_file.parent)
    for img_file in frames_base.rglob("*.jpeg"):
        frame_dirs.add(img_file.parent)
    for img_file in frames_base.rglob("*.png"):
        frame_dirs.add(img_file.parent)
    
    frame_dirs = sorted(frame_dirs)
    
    if not frame_dirs:
        logger.warning(f"No frame directories found in {frames_base_dir}")
        return
    
    logger.info(f"Found {len(frame_dirs)} frame directories to process")
    
    successful = 0
    failed = 0
    
    for frames_dir in frame_dirs:
        # Get the relative path from frames_base
        relative_path = frames_dir.relative_to(frames_base)
        
        # Extract the account/channel name (first directory level)
        path_parts = relative_path.parts
        if not path_parts:
            logger.warning(f"Could not determine channel name for {frames_dir}")
            failed += 1
            continue
        
        channel_name = path_parts[0]
        
        # Find corresponding transcript directory (just the channel name level)
        transcript_dir = transcripts_base / channel_name
        
        if not transcript_dir.exists():
            logger.warning(f"No transcript directory found for {channel_name}")
            failed += 1
            continue
        
        # Find transcript JSON file in the directory
        transcript_files = list(transcript_dir.glob("*_transcript.json"))
        
        if not transcript_files:
            logger.warning(f"No transcript JSON file found in {transcript_dir}")
            failed += 1
            continue
        
        # Use the first transcript file found
        transcript_file = transcript_files[0]
        
        # Create output directory preserving the relative structure
        video_output_dir = output_base / relative_path
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{relative_path.name}_frame_mapping.json"
        output_file = video_output_dir / output_filename
        
        logger.info(f"Processing: {relative_path}")
        
        if align_frames_to_transcript(
            frames_dir=str(frames_dir),
            transcript_path=str(transcript_file),
            output_path=str(output_file),
            include_context=include_context,
            frame_uri_prefix=f"frames/{relative_path}"
        ):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")


def main():
    """Command-line interface for frame-transcript alignment."""
    parser = argparse.ArgumentParser(
        description="Align video frames to transcript segments. Can process single video or batch of videos."
    )
    parser.add_argument(
        "--frames-dir",
        help="Path to directory containing frame images (for single video processing)"
    )
    parser.add_argument(
        "--transcript",
        help="Path to transcript JSON file (for single video processing)"
    )
    parser.add_argument(
        "--output",
        help="Path to output JSON file (for single video processing)"
    )
    parser.add_argument(
        "--frames-base",
        help="Base path to directories containing video frames (for batch processing)"
    )
    parser.add_argument(
        "--transcripts-base",
        help="Base path to directories containing video transcripts (for batch processing)"
    )
    parser.add_argument(
        "--output-base",
        help="Base path for output mappings (for batch processing)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Do not include previous/next segments for context"
    )
    parser.add_argument(
        "--frame-uri-prefix",
        default="frames",
        help="Prefix for frame URIs in output (default: frames)"
    )
    
    args = parser.parse_args()
    
    # Determine mode: single video or batch processing
    if args.frames_base and args.transcripts_base and args.output_base:
        # Batch processing mode
        process_all_videos(
            frames_base_dir=args.frames_base,
            transcripts_base_dir=args.transcripts_base,
            output_base_dir=args.output_base,
            include_context=not args.no_context
        )
    elif args.frames_dir and args.transcript and args.output:
        # Single video processing mode
        align_frames_to_transcript(
            frames_dir=args.frames_dir,
            transcript_path=args.transcript,
            output_path=args.output,
            include_context=not args.no_context,
            frame_uri_prefix=args.frame_uri_prefix
        )
    else:
        print("Error: Must provide either:")
        print("  - Single video: --frames-dir, --transcript, --output")
        print("  - Batch videos: --frames-base, --transcripts-base, --output-base")
        parser.print_help()


if __name__ == "__main__":
    main()

