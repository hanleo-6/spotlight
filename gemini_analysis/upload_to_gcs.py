"""
Upload gemini input files to Google Cloud Storage bucket.
"""
import json
import logging
from pathlib import Path
from google.cloud import storage
import os

# ================= CONFIG =================
PROJECT_ID = "project-5541b270-14df-45c9-9c4"
BUCKET_NAME = "spotlight-analysis-videos-ig-ew4"
LOCAL_INPUT_DIR = Path("data/output/gemini_inputs")
LOCAL_FRAMES_DIR = Path("data/output/keyframes/best_media")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize GCS client
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)


def upload_file_to_gcs(local_path: Path, gcs_path: str) -> bool:
    """Upload a single file to GCS."""
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded: {gcs_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return False


def upload_json_files():
    """Upload all JSON files from gemini_inputs directory to GCS."""
    if not LOCAL_INPUT_DIR.exists():
        logger.error(f"Input directory not found: {LOCAL_INPUT_DIR}")
        return

    # Find all JSON files (excluding frame_mapping files)
    json_files = [
        f for f in LOCAL_INPUT_DIR.rglob("*.json")
        if f.is_file() and "_frame_mapping" not in f.name
    ]

    if not json_files:
        logger.error(f"No JSON files found in {LOCAL_INPUT_DIR}")
        return

    logger.info(f"Found {len(json_files)} JSON files to upload")

    # Create directory structure in GCS and upload
    successful, failed = 0, 0
    for json_file in json_files:
        # Preserve directory structure relative to LOCAL_INPUT_DIR
        relative_path = json_file.relative_to(LOCAL_INPUT_DIR)
        gcs_path = f"gemini_inputs/{relative_path.as_posix()}"

        if upload_file_to_gcs(json_file, gcs_path):
            successful += 1
        else:
            failed += 1

    logger.info(f"Upload complete: {successful} successful, {failed} failed")
    logger.info(f"Files are now accessible at: gs://{BUCKET_NAME}/gemini_inputs/")


def upload_frame_images():
    """Upload all frame images from keyframes directory to GCS."""
    if not LOCAL_FRAMES_DIR.exists():
        logger.error(f"Frames directory not found: {LOCAL_FRAMES_DIR}")
        return

    # Find all JPG files in the keyframes directory
    frame_files = [
        f for f in LOCAL_FRAMES_DIR.rglob("*.jpg")
        if f.is_file()
    ]

    if not frame_files:
        logger.error(f"No frame images found in {LOCAL_FRAMES_DIR}")
        return

    logger.info(f"Found {len(frame_files)} frame images to upload")

    # Upload frames to gemini_inputs/username/video_id/frames/
    successful, failed = 0, 0
    for frame_file in frame_files:
        # Get relative path from LOCAL_FRAMES_DIR
        # e.g., goodieai/DK2OehgMToz_goodieai_score0.8948/keyframe_00000.jpg
        relative_path = frame_file.relative_to(LOCAL_FRAMES_DIR)
        parts = relative_path.parts
        
        if len(parts) >= 3:
            username = parts[0]
            video_id = parts[1]
            filename = parts[2]
            
            # Upload to gemini_inputs/username/video_id/frames/filename
            gcs_path = f"gemini_inputs/{username}/{video_id}/frames/{filename}"
            
            if upload_file_to_gcs(frame_file, gcs_path):
                successful += 1
            else:
                failed += 1
        else:
            logger.warning(f"Unexpected path structure: {relative_path}")
            failed += 1

    logger.info(f"Frame upload complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    logger.info("=== Uploading JSON files ===")
    upload_json_files()
    logger.info("\n=== Uploading frame images ===")
    upload_frame_images()
    logger.info(f"\nAll files are now accessible at: gs://{BUCKET_NAME}/gemini_inputs/")
