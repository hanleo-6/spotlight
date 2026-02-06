import json
import os
import subprocess
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE VALUES FOR YOUR USE CASE
# =============================================================================

TEMPLATE_JSON_PATH = "data/output/template_database/tiktok_vids/tiktok_vids/7124337427774311722_template.json"
TRANSCRIPT_OUTPUT = "video_gen_pipeline/prompts/talking_head_transcript.txt"

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================


def load_transcript(template: Dict[str, str]) -> Tuple[Optional[str], Optional[float]]:
    """Load the transcript file for a template and return (transcript_text, video_duration)."""
    if "transcript_path" in template:
        transcript_path = template["transcript_path"]
    elif "source_video_id" in template and "source_username" in template:
        source_username = template["source_username"]
        video_id = template["source_video_id"]
        base_transcript_dir = "data/output/transcripts"
        search_pattern = os.path.join(
            base_transcript_dir,
            source_username,
            "*",
            f"{video_id}_transcript.txt",
        )

        import glob

        matching_files = glob.glob(search_pattern)

        if matching_files:
            transcript_path = matching_files[0]
        else:
            print(f"Warning: No transcript found for video ID {video_id} in {source_username}")
            return None, None
    else:
        print("Warning: No transcript path or source metadata found in template.")
        return None, None

    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
    else:
        print(f"Warning: Transcript file not found at {transcript_path}")
        return None, None

    video_duration = None
    if "source_video_id" in template and "source_username" in template:
        source_username = template["source_username"]
        video_id = template["source_video_id"]

        try:
            result = subprocess.run(
                ["find", "data/input", "-name", f"{video_id}.info.json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            matching_info_files = [
                f.strip() for f in result.stdout.strip().split("\n") if f.strip()
            ]

            if matching_info_files:
                try:
                    with open(matching_info_files[0], "r", encoding="utf-8") as f:
                        info_data = json.load(f)
                        if "duration" in info_data:
                            video_duration = info_data["duration"]
                except Exception as e:
                    print(
                        f"Warning: Could not parse video duration from {matching_info_files[0]}: {e}"
                    )
        except Exception as e:
            print(f"Warning: Could not find video info file: {e}")

    return transcript_text, video_duration


def main() -> None:
    print(f"Loading template from {TEMPLATE_JSON_PATH}...")
    with open(TEMPLATE_JSON_PATH, "r") as f:
        template = json.load(f)

    transcript_text, _ = load_transcript(template)
    if not transcript_text:
        raise ValueError("No transcript found. Cannot proceed for talking head pipeline.")

    os.makedirs(os.path.dirname(TRANSCRIPT_OUTPUT), exist_ok=True)
    with open(TRANSCRIPT_OUTPUT, "w") as f:
        f.write(transcript_text)

    print(f"✓ Transcript saved to {TRANSCRIPT_OUTPUT}")


if __name__ == "__main__":
    main()
