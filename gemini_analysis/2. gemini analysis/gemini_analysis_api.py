import json
from pathlib import Path
import time
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import logging

# ================= CONFIG =================
# Change this constant to switch between data sources: "tiktok_vids", "best_media", etc.
DATA_SOURCE = "tiktok_vids"  # Options: "tiktok_vids", "best_media"

PROJECT_ID = "project-5541b270-14df-45c9-9c4"
REGION = "europe-west4"
BUCKET_NAME = "spotlight-analysis-videos-tiktok-ew4" # change this bucket for tiktok/ig
GCS_INPUT_PREFIX = f"gemini_inputs/{DATA_SOURCE}"
OUTPUT_DIR = Path(f"data/output/gemini_analysis/{DATA_SOURCE}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consolidated output directory for template database
TEMPLATE_DB_DIR = Path(f"data/output/template_database/{DATA_SOURCE}")
TEMPLATE_DB_DIR.mkdir(parents=True, exist_ok=True)

# How many frames to include in a single request
BATCH_SIZE = 10

# Prompt file (centralized prompt text)
PROMPT_FILE = Path(__file__).parent / "gemini_prompt.txt"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vertex AI with higher timeout for complex analysis
vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel("gemini-2.0-flash")

# Initialize GCS client
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

def load_prompt() -> str:
    """Load Gemini prompt from file, raising if missing."""
    if not PROMPT_FILE.exists():  # fail fast if prompt is not present
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


ANALYSIS_PROMPT = load_prompt()

# ================= FUNCTIONS =================
def load_json_from_gcs(gcs_path: str) -> dict:
    """Load JSON from GCS and fix frame URIs to point to frames in gemini_inputs."""
    try:
        blob = bucket.blob(gcs_path)
        json_data = json.loads(blob.download_as_string())
        
        # Update frame URIs: from gs://old-bucket/frames/user/video/frame.jpg
        # to gs://new-bucket/gemini_inputs/user/video/frames/frame.jpg
        if isinstance(json_data, list):
            for item in json_data:
                if "frame" in item:
                    frame_uri = item["frame"]
                    
                    # Extract path after 'frames/'
                    if "/frames/" in frame_uri:
                        path_after_frames = frame_uri.split("/frames/", 1)[1]
                        parts = path_after_frames.split('/')
                        
                        if len(parts) >= 3:
                            username = parts[0]
                            video_id = parts[1]
                            filename = parts[2]
                            
                            # Construct new URI with DATA_SOURCE
                            item["frame"] = f"gs://{BUCKET_NAME}/gemini_inputs/{DATA_SOURCE}/{username}/{video_id}/frames/{filename}"
        
        return json_data
    except Exception as e:
        logger.error(f"Failed to load from GCS {gcs_path}: {e}")
        return None


def clean_json_response(response_text: str) -> str:
    """Clean Gemini response to extract valid JSON."""
    response_text = response_text.strip()
    
    # Remove markdown code fences
    if response_text.startswith("```"):
        lines = response_text.split('\n')
        # Remove first line (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = '\n'.join(lines).strip()
    
    return response_text


def validate_template_structure(template_data: dict) -> tuple[bool, list]:
    """Validate that template has required fields. Returns (is_valid, missing_fields)."""
    required_fields = [
        "template_identity",
        "structure",
        "visual_system",
        "motion_language",
        "content_formula",
        "niche_targeting",
        "parametric_design",
        "technical_execution",
        "replication_blueprint",
        "template_metadata"
    ]
    
    missing = [field for field in required_fields if field not in template_data]
    return len(missing) == 0, missing


def save_to_template_database(template_data: dict, video_id: str, username: str):
    """Save template to searchable database format."""
    try:
        # Create user-specific directory
        user_dir = TEMPLATE_DB_DIR / username
        user_dir.mkdir(exist_ok=True)
        
        # Enhance template data with metadata
        template_data["source_video_id"] = video_id
        template_data["source_username"] = username
        template_data["extraction_timestamp"] = time.time()
        
        # Generate template ID if not present
        if "template_id" not in template_data.get("template_identity", {}) or not template_data.get("template_identity", {}).get("template_id"):
            # Create ID from video ID and primary niche
            niche = template_data.get("niche_targeting", {}).get("primary_niche", "unknown")
            template_id = f"{username}_{video_id}_{niche.lower().replace(' ', '-').replace('>', '-')}"
            if "template_identity" not in template_data:
                template_data["template_identity"] = {}
            template_data["template_identity"]["template_id"] = template_id
        
        # Save individual template file
        template_file = user_dir / f"{video_id}_template.json"
        with open(template_file, "w") as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"✓ Template saved to database: {template_file}")
        
        # Append to master template index
        index_file = TEMPLATE_DB_DIR / "template_index.jsonl"
        with open(index_file, "a") as f:
            # Create simplified index entry for fast searching
            template_id = template_data.get("template_identity", {}).get("template_id", "unknown")
            index_entry = {
                "template_id": template_id,
                "template_name": template_data.get("template_identity", {}).get("template_name"),
                "category": template_data.get("template_identity", {}).get("template_category"),
                "primary_niche": template_data.get("niche_targeting", {}).get("primary_niche"),
                "visual_fingerprint": template_data.get("template_identity", {}).get("visual_fingerprint"),
                "complexity": template_data.get("replication_blueprint", {}).get("complexity_level"),
                "versatility": template_data.get("template_metadata", {}).get("versatility"),
                "tags": template_data.get("template_metadata", {}).get("search_keywords", []),
                "niche_fit_reasoning": template_data.get("niche_targeting", {}).get("template_niche_fit_reasoning"),
                "source_video": video_id,
                "username": username,
                "file_path": str(template_file)
            }
            f.write(json.dumps(index_entry) + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save to template database: {e}")
        return False


def analyze_video_with_gemini(gcs_json_path: str, video_id: str) -> bool:
    """Analyze video using JSON file from GCS.
    
    Args:
        gcs_json_path: Path to JSON file in GCS (e.g., 'gemini_inputs/username/video.json')
        video_id: Identifier for the video (stem of filename)
    """
    try:
        # Extract username from path
        path_parts = gcs_json_path.split('/')
        username = path_parts[1] if len(path_parts) > 1 else "unknown"
        
        # Load per-video input from GCS
        video_data = load_json_from_gcs(gcs_json_path)
        
        if not video_data:
            logger.warning(f"Empty input: {gcs_json_path}")
            return False

        # For template analysis, we want to analyze ALL frames together for comprehensive view
        # Build single comprehensive content array
        content = [ANALYSIS_PROMPT]
        
        logger.info(f"Analyzing {len(video_data)} frames for {video_id}")
        
        # Add all frames and transcripts (limit to reasonable number for context window)
        max_frames = 50  # Adjust based on video length and model limits
        sample_interval = max(1, len(video_data) // max_frames)
        
        sampled_frames = video_data[::sample_interval][:max_frames]
        
        for idx, item in enumerate(sampled_frames):
            # Image part
            content.append(Part.from_uri(item["frame"], mime_type="image/jpeg"))
            # Transcript text with frame number
            frame_num = idx * sample_interval
            content.append(f"Frame {frame_num} transcript: {item['transcript']}")

        # Call Gemini with comprehensive analysis
        logger.info(f"Sending {len(sampled_frames)} frames to Gemini for template extraction...")
        
        response = model.generate_content(
            content,
            generation_config={
                "temperature": 0.2,  # Lower for more consistent JSON
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,  # Higher limit for detailed template data
            }
        )

        # Parse response
        try:
            response_text = clean_json_response(response.text)
            template_data = json.loads(response_text)
            
            # Validate structure
            is_valid, missing_fields = validate_template_structure(template_data)
            if not is_valid:
                logger.warning(f"Template data missing fields: {missing_fields}")
                # Continue anyway, partial data is better than nothing
            
            logger.info(f"✓ Successfully extracted template data for {video_id}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse JSON from Gemini response: {e}")
            logger.debug(f"Response text preview: {response.text[:500]}...")
            template_data = {
                "parse_error": str(e),
                "raw_response": response.text,
                "template_name": f"Parse Error - {video_id}",
                "template_category": "unknown"
            }

        # Save detailed output
        if len(path_parts) > 2:
            subdir = '/'.join(path_parts[1:-1])
            output_subdir = OUTPUT_DIR / subdir
        else:
            output_subdir = OUTPUT_DIR
        
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"{video_id}_analysis.json"
        
        with open(output_file, "w") as f:
            json.dump({
                "video_id": video_id,
                "username": username,
                "gcs_input_path": gcs_json_path,
                "frame_count": len(video_data),
                "frames_analyzed": len(sampled_frames),
                "template_data": template_data
            }, f, indent=2)

        logger.info(f"Analysis saved to {output_file}")
        
        # Save to template database if valid
        if "parse_error" not in template_data:
            save_to_template_database(template_data, video_id, username)
        
        return True

    except Exception as e:
        logger.error(f"Error analyzing {gcs_json_path}: {e}", exc_info=True)
        return False


def generate_summary_report():
    """Generate summary statistics from all extracted templates."""
    try:
        index_file = TEMPLATE_DB_DIR / "template_index.jsonl"
        if not index_file.exists():
            logger.info("No templates in database yet")
            return
        
        templates = []
        with open(index_file, "r") as f:
            for line in f:
                templates.append(json.loads(line))
        
        # Generate statistics
        stats = {
            "total_templates": len(templates),
            "by_category": {},
            "by_niche": {},
            "by_complexity": {},
            "by_user": {}
        }
        
        for t in templates:
            # Count by category
            cat = t.get("category", "unknown")
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            
            # Count by niche
            niche = t.get("primary_niche", "unknown")
            stats["by_niche"][niche] = stats["by_niche"].get(niche, 0) + 1
            
            # Count by complexity
            complexity = t.get("complexity", "unknown")
            stats["by_complexity"][complexity] = stats["by_complexity"].get(complexity, 0) + 1
            
            # Count by versatility
            versatility = t.get("versatility", "unknown")
            stats["by_versatility"] = stats.get("by_versatility", {})
            stats["by_versatility"][versatility] = stats["by_versatility"].get(versatility, 0) + 1
            
            # Count by user
            user = t.get("username", "unknown")
            stats["by_user"][user] = stats["by_user"].get(user, 0) + 1
        
        # Save summary
        summary_file = TEMPLATE_DB_DIR / "summary_stats.json"
        with open(summary_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"\n{'='*50}")
        logger.info("TEMPLATE DATABASE SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total templates: {stats['total_templates']}")
        logger.info(f"\nTop categories:")
        for cat, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {cat}: {count}")
        logger.info(f"\nTop niches:")
        for niche, count in sorted(stats['by_niche'].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {niche}: {count}")
        logger.info(f"\nBy complexity:")
        for complexity, count in sorted(stats['by_complexity'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {complexity}: {count}")
        logger.info(f"\nBy versatility:")
        for versatility, count in sorted(stats.get('by_versatility', {}).items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {versatility}: {count}")
        logger.info(f"{'='*50}\n")
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")


def main():
    """List all JSON files from GCS bucket and analyze them."""
    try:
        # List all JSON files in GCS bucket under gemini_inputs prefix
        blobs = storage_client.list_blobs(
            BUCKET_NAME, 
            prefix=GCS_INPUT_PREFIX,
            delimiter=None
        )
        
        video_files = [
            blob.name for blob in blobs
            if blob.name.endswith(".json") and "_frame_mapping" not in blob.name
        ]

        if not video_files:
            logger.error(f"No video input files found in GCS: gs://{BUCKET_NAME}/{GCS_INPUT_PREFIX}")
            return

        logger.info(f"Found {len(video_files)} videos in GCS")

        # SAMPLE: Limit to first N videos for testing
        # SAMPLE_SIZE = 5  # Change this to test fewer videos
        # video_files = video_files[:SAMPLE_SIZE]
        # logger.info(f"Testing with sample of {SAMPLE_SIZE} videos")

        # Filter out already analyzed videos
        videos_to_process = []
        for gcs_path in video_files:
            video_id = gcs_path.split('/')[-1].replace('.json', '')
            path_parts = gcs_path.split('/')
            
            # Determine output file path
            if len(path_parts) > 2:
                subdir = '/'.join(path_parts[1:-1])
                output_subdir = OUTPUT_DIR / subdir
            else:
                output_subdir = OUTPUT_DIR
            
            output_file = output_subdir / f"{video_id}_analysis.json"
            
            if output_file.exists():
                logger.info(f"Skipping {video_id} (already analyzed)")
            else:
                videos_to_process.append(gcs_path)
        
        if not videos_to_process:
            logger.info("All videos have already been analyzed!")
            generate_summary_report()
            return
        
        logger.info(f"Processing {len(videos_to_process)} new videos (skipped {len(video_files) - len(videos_to_process)} already analyzed)")

        successful, failed = 0, 0
        for idx, gcs_path in enumerate(videos_to_process, 1):
            video_id = gcs_path.split('/')[-1].replace('.json', '')
            logger.info(f"\n[{idx}/{len(videos_to_process)}] Processing {video_id}...")
            
            if analyze_video_with_gemini(gcs_path, video_id):
                successful += 1
            else:
                failed += 1
            
            # Rate limiting between videos
            if idx < len(videos_to_process):
                time.sleep(2)

        logger.info(f"\n{'='*50}")
        logger.info(f"BATCH COMPLETE: {successful} successful, {failed} failed")
        logger.info(f"{'='*50}\n")
        
        # Generate summary report
        generate_summary_report()
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()