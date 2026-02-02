"""
Extract transcripts from videos using Whisper.
Processes videos from an input folder and generates transcripts.
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_transcripts(input_folder: str, output_folder: str, model_size: str = "base") -> None:
    """
    Extract transcripts from all videos in input folder using Whisper.
    
    Args:
        input_folder: Path to folder containing video files
        output_folder: Path to folder where transcripts will be saved
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    """
    try:
        import whisper
    except ImportError:
        logger.error("Whisper is not installed. Install it via: pip install openai-whisper")
        return
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported video formats
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4a', '.mp3', '.wav'}
    
    # Load Whisper model
    logger.info(f"Loading Whisper model: {model_size}")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return
    
    # Recursively find all media files in input folder and subfolders
    media_files = []
    for ext in video_extensions:
        media_files.extend(input_path.rglob(f"*{ext}"))
    
    media_files = sorted(media_files)
    
    if not media_files:
        logger.warning(f"No media files found in {input_folder}")
        return
    
    for media_file in media_files:
        try:
            logger.info(f"Transcribing {media_file.name}")
            
            # Transcribe using Whisper
            result = model.transcribe(str(media_file), language="en", verbose=False)
            
            # Output transcript paths - preserve relative subfolder structure
            relative_path = media_file.relative_to(input_path)
            output_subfolder = output_path / relative_path.parent
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            # Save transcript as text
            text_output = output_subfolder / f"{media_file.stem}_transcript.txt"
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            # Save detailed transcript with timestamps as JSON
            import json
            json_output = output_subfolder / f"{media_file.stem}_transcript.json"
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Transcript saved to {text_output.name}")
            
        except Exception as e:
            logger.error(f"Error processing {media_file.name}: {e}")


if __name__ == "__main__":
    # Default paths
    input_dir = "data/input"
    output_dir = "data/output/transcripts"
    
    # You can change model_size to 'tiny', 'small', 'medium', or 'large' for better accuracy
    # Larger models are more accurate but slower
    extract_transcripts(input_dir, output_dir, model_size="base")
    logger.info("Transcript extraction completed!")
