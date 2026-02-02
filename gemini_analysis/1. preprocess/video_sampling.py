"""
Preprocess: Sample keyframes at strategic intervals (every 3 seconds)
Extracts keyframes from videos in an input folder and saves them to output subfolders.
"""

import os
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_INTERVAL = 3  # seconds


def sample_keyframes(input_folder: str, output_folder: str, sample_interval: int = SAMPLE_INTERVAL) -> None:
    """
    Sample keyframes from all videos in input folder at regular intervals.
    
    Args:
        input_folder: Path to folder containing video files
        output_folder: Path to folder where keyframes will be saved
        sample_interval: Interval in seconds between sampled keyframes (default: 3)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported video formats
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    # Recursively find all video files in input folder and subfolders
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.rglob(f"*{ext}"))
    
    video_files = sorted(video_files)
    
    if not video_files:
        logger.warning(f"No video files found in {input_folder}")
        return
    
    for video_file in video_files:
        try:
            logger.info(f"Processing {video_file.name}")
            
            # Create output subfolder preserving the relative directory structure
            relative_path = video_file.relative_to(input_path)
            video_output_folder = output_path / relative_path.parent / video_file.stem
            video_output_folder.mkdir(parents=True, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(str(video_file))
            
            if not cap.isOpened():
                logger.error(f"Failed to open {video_file.name}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                logger.warning(f"Could not determine FPS for {video_file.name}")
                fps = 30  # Default fallback
            
            frame_interval = int(fps * sample_interval)
            frame_count = 0
            keyframe_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Save keyframe at regular intervals
                if frame_count % frame_interval == 0:
                    # Calculate actual timestamp in seconds
                    timestamp_seconds = int(frame_count / fps)
                    keyframe_path = video_output_folder / f"keyframe_t{timestamp_seconds:05d}.jpg"
                    cv2.imwrite(str(keyframe_path), frame)
                    keyframe_count += 1
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {keyframe_count} keyframes from {video_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {video_file.name}: {e}")


if __name__ == "__main__":
    # Default paths
    input_dir = "data/input"
    output_dir = "data/output/keyframes"
    
    sample_keyframes(input_dir, output_dir)
    logger.info("Keyframe sampling completed!")
