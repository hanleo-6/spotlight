"""
Extract audio from videos using ffmpeg.
Processes videos from an input folder and extracts audio tracks as MP3 files.
"""

import os
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_audio(input_folder: str, output_folder: str) -> None:
    """
    Extract audio from all videos in input folder using ffmpeg.
    
    Args:
        input_folder: Path to folder containing video files
        output_folder: Path to folder where audio files will be saved
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported video formats
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except FileNotFoundError:
        logger.error("ffmpeg is not installed. Please install it to use this function.")
        logger.info("Install via: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
        return
    
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
            logger.info(f"Extracting audio from {video_file.name}")
            
            # Output audio file path - preserve relative subfolder structure
            relative_path = video_file.relative_to(input_path)
            output_subfolder = output_path / relative_path.parent
            output_subfolder.mkdir(parents=True, exist_ok=True)
            audio_output = output_subfolder / f"{video_file.stem}.mp3"
            
            # Use ffmpeg to extract audio
            # -i: input file
            # -q:a 9: audio quality (0=best, 9=worst)
            # -map a: select audio stream
            command = [
                'ffmpeg',
                '-i', str(video_file),
                '-q:a', '9',
                '-map', 'a',
                str(audio_output),
                '-y'  # Overwrite without asking
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully extracted audio to {audio_output.name}")
            else:
                logger.error(f"Failed to extract audio from {video_file.name}")
                logger.error(f"Error: {result.stderr}")
        
        except Exception as e:
            logger.error(f"Error processing {video_file.name}: {e}")


if __name__ == "__main__":
    # Default paths
    input_dir = "data/input/best_media"
    output_dir = "data/output/audio"
    
    extract_audio(input_dir, output_dir)
    logger.info("Audio extraction completed!")
