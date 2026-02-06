import json
import mimetypes
import os
import random
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydub import AudioSegment

from tts_generator import TTSAudioGenerator

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE VALUES FOR YOUR USE CASE
# =============================================================================

# Input prompts and transcript
IMAGE_PROMPT_FILE = "video_gen_pipeline/prompts/image_prompt_stitching.txt"
VIDEO_PROMPT_FILE = "video_gen_pipeline/prompts/video_prompt_stitching.txt"
TIMESTAMPED_TRANSCRIPT_FILE = "video_gen_pipeline/prompts/timestamped_transcript.txt"
STITCHING_PROMPT_FILE = "video_gen_pipeline/prompts/stitching_prompt.txt"

# Template source (used to pull the first frame from the original video)
TEMPLATE_JSON_PATH = "data/output/template_database/tiktok_vids/tiktok_vids/7124337427774311722_template.json"
SOURCE_VIDEO_BASE_DIR = "data/input/tiktok_vids"

# Output
FINAL_OUTPUT_BASE_DIR = "data/output/generated_videos"

# TOGGLE HERE!
GENERATE_IMAGE = True
GENERATE_VIDEO_WITH_SEGMENT_STITCHING = True

# Video settings
VIDEO_ASPECT_RATIO = "9:16"
VIDEO_DURATION_SECONDS = 8
SEGMENT_DURATION_SECONDS = 8

# Testing
# Set to an integer to limit how many segments are generated (e.g., 2). Use None for no limit.
MAX_SEGMENTS_TO_GENERATE = 2

# Continuity and segment extension
EXTEND_SEGMENTS = True
EXTEND_REMOVE_OVERLAP = True
LAST_FRAME_IMAGE_NAME = "last_frame.png"
USE_TRANSCRIPT_TIMESTAMPS = True
USE_PREVIOUS_VIDEO_CONTEXT = True  # Pass prior segment video into Veo for smoother stitching

# Person reference image
USE_PERSON_REFERENCE_IMAGE = True
PERSON_REFERENCE_IMAGE_PATH = "video_gen_pipeline/assets/person_reference.jpg"
# UK region requires allow_adult when using reference images / people
PERSON_GENERATION = "allow_adult"

# Audio + lip sync
GENERATE_SILENT_VIDEOS = True  # Tell Veo to skip audio generation
GENERATE_FULL_AUDIO_TRACK = True  # Generate one complete audio file
APPLY_LIPSYNC = True  # Apply Wav2Lip to sync audio with video
TTS_VOICE_NAME = "en-US-Studio-0"
TTS_SPEAKING_RATE = 1.0
TTS_PITCH = 0.0

# Wav2Lip settings
WAV2LIP_CHECKPOINT_PATH = "Wav2Lip/checkpoints/wav2lip_gan.pth"
WAV2LIP_RESIZE_FACTOR = 1
WAV2LIP_FACE_DETECT_BATCH_SIZE = 16
WAV2LIP_WAV2LIP_BATCH_SIZE = 128

# Rate limiting
ENABLE_RATE_LIMITING = True
MAX_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 5
RETRY_BACKOFF_MULTIPLIER = 2
MAX_RETRY_DELAY_SECONDS = 300
JITTER_ENABLED = True
DELAY_BETWEEN_VIDEO_GENERATIONS = 2

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================

# Initialize TTS generator (created lazily if needed)
tts_generator = None


def call_api_with_retries(
    api_call_func,
    *args,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY_SECONDS,
    backoff_multiplier: float = RETRY_BACKOFF_MULTIPLIER,
    max_delay: float = MAX_RETRY_DELAY_SECONDS,
    operation_name: str = "API call",
    **kwargs
):
    """
    Call an API function with exponential backoff retry logic for rate limiting.
    
    Args:
        api_call_func: Function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retries
        initial_delay: Initial delay before first retry in seconds
        backoff_multiplier: Multiply delay by this for each retry
        max_delay: Cap maximum delay at this value
        operation_name: Name of operation for logging
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the API call
    
    Raises:
        The last exception if all retries are exhausted
    """
    if not ENABLE_RATE_LIMITING:
        return api_call_func(*args, **kwargs)
    
    last_exception = None
    current_delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return api_call_func(*args, **kwargs)
        except errors.ClientError as e:
            last_exception = e
            
            # Check if this is a rate limiting error
            if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries:
                    # Add jitter if enabled
                    jitter = random.uniform(0, current_delay * 0.1) if JITTER_ENABLED else 0
                    wait_time = current_delay + jitter
                    
                    print(f"\n⚠ Rate limited on {operation_name} (attempt {attempt + 1}/{max_retries + 1})")
                    print(f"Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                    
                    # Exponential backoff for next retry
                    current_delay = min(current_delay * backoff_multiplier, max_delay)
                    continue
                else:
                    # Out of retries
                    print(f"\n✗ Rate limiting failed after {max_retries} retries for {operation_name}")
                    raise
            else:
                # Not a rate limiting error, fail immediately
                raise
    
    # Should not reach here, but raise last exception just in case
    raise last_exception


def _get_next_run_index(base_dir: str) -> int:
    """Get the next index for a new run. Reuses the last folder if incomplete, otherwise creates a new one."""
    if not os.path.exists(base_dir):
        return 1
    
    max_index = 0
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                max_index = max(max_index, int(item))
    except (OSError, ValueError):
        pass
    
    if max_index == 0:
        return 1
    
    # Check if the last run is complete by looking for a completion marker
    last_run_dir = os.path.join(base_dir, str(max_index))
    completion_marker = os.path.join(last_run_dir, ".complete")
    
    # If the last run doesn't have a completion marker, reuse it
    if not os.path.exists(completion_marker):
        return max_index
    
    # Otherwise, create a new run
    return max_index + 1

def _extract_first_frame(video_path: str, output_path: str) -> bool:
    """Extract the first frame of a video using ffmpeg. Returns True on success."""
    if not os.path.exists(video_path):
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-frames:v", "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception:
        return False


def _prepare_video_context(video_path: str) -> Optional[str]:
    """Re-encode prior segment to a Veo-friendly MP4 (H.264, yuv420p, 720x1280)."""
    if not os.path.exists(video_path):
        return None

    base, _ = os.path.splitext(video_path)
    prepared_path = f"{base}_veo_ctx.mp4"

    if os.path.exists(prepared_path) and os.path.getsize(prepared_path) > 0:
        return prepared_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
        "-r", "30",
        "-an",
        prepared_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return prepared_path if os.path.exists(prepared_path) else None
    except Exception:
        return None


def _sorted_segments(timestamped_data: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    """Return transcript segments sorted by segment number."""
    segments = timestamped_data.get("segments", {})
    return sorted(segments.items())


def generate_full_audio_from_transcript(
    timestamped_data: Dict[str, Any],
    output_path: str,
    tts_generator: TTSAudioGenerator
) -> bool:
    """Generate a single full audio track by concatenating all segment text."""
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"⊘ Full audio already exists: {output_path} ({file_size_mb:.1f} MB)")
        return True

    ordered_segments = _sorted_segments(timestamped_data)
    if not ordered_segments:
        print("⚠ No transcript segments found for full audio generation")
        return False

    texts = [seg_data.get("text", "").strip() for _, seg_data in ordered_segments if seg_data.get("text")]
    full_text = " ".join(texts).strip()

    if not full_text:
        print("⚠ Full transcript text is empty, skipping audio generation")
        return False

    print(f"Generating full audio: {len(full_text)} characters across {len(ordered_segments)} segments")
    success = tts_generator.synthesize_text(full_text, output_path)
    return success and os.path.exists(output_path)


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    if not os.path.exists(audio_path):
        return 0.0

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def slice_audio_by_timestamps(
    full_audio_path: str,
    timestamped_data: Dict[str, Any],
    output_dir: str
) -> Dict[int, str]:
    """Slice a full audio file into timestamped segments."""
    if not os.path.exists(full_audio_path):
        print(f"✗ Full audio file not found: {full_audio_path}")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    full_audio = AudioSegment.from_file(full_audio_path)

    segment_audio_paths: Dict[int, str] = {}

    for seg_num, seg_data in _sorted_segments(timestamped_data):
        start_ms = int(seg_data.get("start_time", 0.0) * 1000)
        end_ms = int(seg_data.get("end_time", 0.0) * 1000)
        segment_audio_path = os.path.join(output_dir, f"segment_{seg_num:02d}_audio.mp3")

        if os.path.exists(segment_audio_path):
            file_size_kb = os.path.getsize(segment_audio_path) / 1024
            print(f"⊘ Audio segment already exists: {segment_audio_path} ({file_size_kb:.1f} KB)")
            segment_audio_paths[seg_num] = segment_audio_path
            continue

        segment_audio = full_audio[start_ms:end_ms]
        segment_audio.export(segment_audio_path, format="mp3")
        print(f"✓ Exported audio segment {seg_num} to {segment_audio_path}")
        segment_audio_paths[seg_num] = segment_audio_path

    return segment_audio_paths


def apply_wav2lip_to_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    checkpoint_path: str = WAV2LIP_CHECKPOINT_PATH,
    resize_factor: int = WAV2LIP_RESIZE_FACTOR,
    face_det_batch_size: int = WAV2LIP_FACE_DETECT_BATCH_SIZE,
    wav2lip_batch_size: int = WAV2LIP_WAV2LIP_BATCH_SIZE,
) -> bool:
    """Apply Wav2Lip lip sync to a video using the provided audio."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return False
    if not os.path.exists(audio_path):
        print(f"✗ Audio file not found: {audio_path}")
        return False
    if not os.path.exists(checkpoint_path):
        print(f"✗ Wav2Lip checkpoint not found: {checkpoint_path}")
        return False

    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"⊘ Lip-synced video already exists: {output_path} ({file_size_mb:.1f} MB)")
        return True

    cmd = [
        "python",
        "Wav2Lip/inference.py",
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--resize_factor", str(resize_factor),
        "--face_det_batch_size", str(face_det_batch_size),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
        "--nosmooth",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(output_path):
            return True
        print("✗ Wav2Lip completed but output file was not created")
        return False
    except subprocess.CalledProcessError as e:
        print("✗ Wav2Lip failed")
        if e.stderr:
            print(e.stderr)
        return False
    except Exception as e:
        print(f"✗ Wav2Lip error: {e}")
        return False


def _build_person_replacement_prompt(base_prompt: str) -> str:
    """Enhance the prompt with strict person-replacement guidance."""
    return (
        "Edit the first image (the source frame). Replace the original person with the person in the reference image. "
        "Match the original person's exact pose, body orientation, gaze direction, and facial expression from the source frame. "
        "Keep the background, camera angle, lens, framing, lighting, shadows, color grade, and scene details identical to the source frame. "
        "Preserve all non-person elements and text overlays exactly. "
        "Match realistic skin texture, pores, hair strands, clothing fabric, and natural edge blending. "
        "Ensure consistent perspective, scale, and contact shadows; avoid AI artifacts, distortions, extra limbs, or warped hands. "
        "Result should look like a real photograph or video frame, not AI-generated. "
        f"{base_prompt}"
    )


def generate_reference_image(
    prompt: str,
    output_path: str,
    base_frame_path: Optional[str] = None,
    person_reference_image_path: Optional[str] = None,
) -> None:
    """Generate the initial reference image using Gemini."""
    # Skip if image already exists
    if os.path.exists(output_path):
        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"⊘ Reference image already exists: {output_path} ({file_size_kb:.1f} KB)")
        return
    
    print("Generating reference image from prompt...")
    try:
        contents = []
        base_prompt = prompt

        if base_frame_path and os.path.exists(base_frame_path):
            base_prompt = _build_person_replacement_prompt(base_prompt)
            mime_type, _ = mimetypes.guess_type(base_frame_path)
            if not mime_type:
                raise ValueError(
                    f"Could not determine mime type for {base_frame_path}"
                )
            with open(base_frame_path, "rb") as f:
                base_bytes = f.read()
            base_part = types.Part.from_bytes(data=base_bytes, mime_type=mime_type)
            contents.append(base_part)
            print(f"Using base frame for replacement: {base_frame_path}")

        if (USE_PERSON_REFERENCE_IMAGE and person_reference_image_path and os.path.exists(person_reference_image_path)):
            ref_size_kb = os.path.getsize(person_reference_image_path) / 1024
            print(
                "Using person reference image for replacement: "
                f"{person_reference_image_path} ({ref_size_kb:.1f} KB)"
            )
            mime_type, _ = mimetypes.guess_type(person_reference_image_path)
            if not mime_type:
                raise ValueError(
                    f"Could not determine mime type for {person_reference_image_path}"
                )
            with open(person_reference_image_path, "rb") as f:
                person_bytes = f.read()
            person_part = types.Part.from_bytes(data=person_bytes, mime_type=mime_type)
            contents.append(person_part)

        contents.append(base_prompt)

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio="9:16",
                ),
            ),
        )
        
        # Check if response is valid
        if not response:
            raise RuntimeError("Received empty response from Gemini API")
        
        if not hasattr(response, 'parts') or response.parts is None:
            raise RuntimeError(f"Response missing 'parts' attribute. Response: {response}")
        
        # Look for image data in response
        image_saved = False
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(output_path)
                print(f"Reference image saved to {output_path}")
                image_saved = True
                break
        
        if not image_saved:
            raise RuntimeError("No image data found in response parts")
            
    except errors.ClientError as e:
        print("=== ClientError from Gemini ===")
        print("Error:", e)
        if hasattr(e, 'response_json'):
            print("Response JSON:", e.response_json)
        raise
    except Exception as e:
        print(f"Error generating image: {e}")
        raise


def generate_video_from_image(
    video_prompt: str,
    image_path: str,
    output_path: str,
    aspect_ratio: str = "9:16",
    duration_seconds: int = 4,
    previous_video: Optional[Any] = None,
    previous_video_path: Optional[str] = None,
    generate_audio: bool = False,
) -> Optional[Any]:
    """Generate video from reference image using Veo-3.1."""
    # Skip if video already exists
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"⊘ Video already exists: {output_path} ({file_size_mb:.1f} MB)")
        return None
    
    image = None
    video_context = None
    if previous_video is not None:
        print("Using previous Veo-generated video context from in-memory reference...")
        video_context = previous_video
    elif previous_video_path:
        print("⚠ Previous video path provided, but Veo extension requires a Veo-generated video object. Falling back to image input...")

    if video_context is None:
        print(f"Loading reference image from {image_path}...")
        image = types.Image.from_file(location=image_path)

    # Use only the generated reference image for video generation
    enhanced_prompt = video_prompt
    if not generate_audio:
        enhanced_prompt = f"{video_prompt} [Generate without audio/soundtrack]"
    reference_images = []

    print(f"Starting video generation (audio={'enabled' if generate_audio else 'disabled'})...")

    def _generate_video(use_video_context: bool, fallback_image: Optional[types.Image] = None):
        config_kwargs = {
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_seconds,
        }
        if use_video_context and video_context:
            config_kwargs["resolution"] = "720p"
        if reference_images:
            config_kwargs["reference_images"] = reference_images
            if PERSON_GENERATION:
                config_kwargs["person_generation"] = PERSON_GENERATION

        call_kwargs = {
            "model": "veo-3.1-generate-preview",
            "prompt": enhanced_prompt,
            "config": types.GenerateVideosConfig(**config_kwargs),
        }
        if use_video_context and video_context is not None:
            call_kwargs["video"] = video_context
        else:
            if image is not None:
                call_kwargs["image"] = image
            elif fallback_image is not None:
                call_kwargs["image"] = fallback_image

        return client.models.generate_videos(**call_kwargs)

    try:
        operation = call_api_with_retries(
            _generate_video,
            True,
            operation_name="video generation"
        )
    except errors.ClientError as e:
        print("=== ClientError from Veo ===")
        print("Status code:", e.code)
        if hasattr(e, 'response_json') and e.response_json:
            print("Response JSON:", e.response_json)

        error_text = str(e)
        response_text = ""
        if hasattr(e, "response_json") and e.response_json:
            response_text = json.dumps(e.response_json)

        context_rejected = any(
            token in error_text or token in response_text
            for token in [
                "encoding",
                "INVALID_ARGUMENT",
                "video context rejected",
                "video_context_rejected",
            ]
        )

        if video_context and context_rejected:
            print("⚠ Video context rejected by model; retrying with image input...")
            fallback_image = types.Image.from_file(location=image_path)
            operation = call_api_with_retries(
                _generate_video,
                False,
                fallback_image,
                operation_name="video generation (fallback)"
            )
        else:
            raise

    print(f"Operation started: {operation.name}")

    # Poll until video generation completes
    while not operation.done:
        print("Waiting for video generation to complete...")
        time.sleep(10)
        operation = client.operations.get(operation)

    # Check for errors
    if operation.error:
        print("Video generation FAILED:")
        print(operation.error)
        raise SystemExit(1)

    if not operation.response:
        raise RuntimeError(
            f"Video generation completed but response is None. "
            f"Operation metadata: {operation.metadata}"
        )

    if not getattr(operation.response, "generated_videos", None):
        raise RuntimeError(
            f"Video generation completed but no generated_videos found. "
            f"Operation response: {operation.response}"
        )

    # Save the generated video
    video = operation.response.generated_videos[0]
    client.files.download(file=video.video)
    video.video.save(output_path)
    
    # Add delay between video generations to avoid hitting rate limits
    if ENABLE_RATE_LIMITING and DELAY_BETWEEN_VIDEO_GENERATIONS > 0:
        print(f"Waiting {DELAY_BETWEEN_VIDEO_GENERATIONS}s before next video generation...")
        time.sleep(DELAY_BETWEEN_VIDEO_GENERATIONS)
    print(f"Generated video saved to {output_path}")
    return video.video


def _split_segment_prompts(raw_prompt: str) -> List[str]:
    """Split prompt file content into per-segment prompts."""
    # Try new delimiter first, fallback to old one
    if "\n---SEGMENT_BREAK---\n" in raw_prompt:
        prompts = [p.strip() for p in raw_prompt.split("\n---SEGMENT_BREAK---\n") if p.strip()]
    else:
        prompts = [p.strip() for p in raw_prompt.split("\n---\n") if p.strip()]
    return prompts


def _load_timestamped_transcript(path: str) -> Dict[str, Any]:
    """Load timestamped transcript file and parse segment boundaries."""
    if not os.path.exists(path):
        return {"segments": []}
    
    with open(path, "r") as f:
        content = f.read()
    
    segments = {}
    current_segment = None
    current_text = []
    
    for line in content.split("\n"):
        if line.startswith("[Segment"):
            if current_segment is not None and current_text:
                # Store previous segment
                segments[current_segment]["text"] = "\n".join(current_text).strip()
            # Extract segment number from "[Segment N: label]"
            match = re.search(r'\[Segment (\d+):', line)
            if match:
                current_segment = int(match.group(1))
                segments[current_segment] = {"start_time": 0.0, "end_time": 0.0, "text": ""}
            current_text = []
        elif line.startswith("Time:") and current_segment is not None:
            # Parse time: "Time: 0.0s - 8.0s"
            time_match = re.search(r'Time: ([\d.]+)s - ([\d.]+)s', line)
            if time_match and current_segment in segments:
                segments[current_segment]["start_time"] = float(time_match.group(1))
                segments[current_segment]["end_time"] = float(time_match.group(2))
        elif line.startswith("Transcript:") and current_segment is not None:
            continue
        elif current_segment is not None and line and not line.startswith("-") and not line.startswith("["):
            current_text.append(line)
    
    # Store final segment
    if current_segment is not None and current_text:
        segments[current_segment]["text"] = "\n".join(current_text).strip()
    
    return {"segments": segments}


def _extract_last_frame(video_path: str, output_path: str) -> bool:
    """Extract the last frame of a video using ffmpeg. Returns True on success."""
    if not os.path.exists(video_path):
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof", "-0.1",
        "-i", video_path,
        "-vframes", "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception:
        return False


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _remove_overlap(prev_prompt: str, next_prompt: str) -> str:
    """Remove sentences in next_prompt that appear in prev_prompt (simple heuristic)."""
    prev_sentences = set(_sentence_split(prev_prompt))
    next_sentences = _sentence_split(next_prompt)
    kept = [s for s in next_sentences if s not in prev_sentences]
    return " ".join(kept).strip()


def _compose_extension_prompt(action_delta: str, invariants: str, transcript_context: str = None) -> str:
    """Compose a strict continuation prompt with transcript context."""
    base = (
        "Continue seamlessly from the last frame. Maintain the same camera position, framing, "
        "lighting, subject appearance, environment, and tone."
    )
    if invariants:
        base = f"{base} {invariants}"
    
    parts = [base]
    if transcript_context:
        parts.append(f"Transcript for this segment: {transcript_context}")
    if action_delta:
        parts.append(action_delta)
    
    return " ".join(parts)


def _load_stitching_prompts(path: str) -> Dict[str, str]:
    """Load stitching prompts for within-segment and between-segment modes."""
    with open(path, "r") as f:
        raw = f.read()

    def _extract_block(label: str) -> str:
        start = raw.find(f"[{label}]")
        if start == -1:
            return ""
        start += len(f"[{label}]")
        end = raw.find("\n[", start)
        return raw[start:end].strip() if end != -1 else raw[start:].strip()

    return {
        "within_segment": _extract_block("WITHIN_SEGMENT"),
        "between_segments": _extract_block("BETWEEN_SEGMENTS"),
    }


def _compose_segment_prompt(segment_prompt: str, within_stitching: str) -> str:
    if within_stitching:
        return f"{segment_prompt}\n\nStitching guidance (within segment): {within_stitching}"
    return segment_prompt


def _compose_between_segments_prompt(prev_prompt: str, next_prompt: str, between_stitching: str) -> str:
    if between_stitching:
        return (
            f"Create a seamless transition from the previous segment to the next. "
            f"Previous segment summary: {prev_prompt}\n"
            f"Next segment summary: {next_prompt}\n"
            f"Stitching guidance (between segments): {between_stitching}"
        )
    return f"Transition from: {prev_prompt} -> {next_prompt}"


def _stitch_videos(segment_paths: List[str], output_path: str) -> bool:
    """Stitch multiple video segments into one final video using ffmpeg concat demuxer.
    
    Args:
        segment_paths: List of paths to video segments in order
        output_path: Path where the final stitched video will be saved
    
    Returns:
        True if stitching succeeded, False otherwise
    """
    if not segment_paths:
        print("No segments to stitch.")
        return False
    
    if len(segment_paths) == 1:
        # If only one segment, just copy it
        print(f"Only one segment, copying to output...")
        import shutil
        shutil.copy(segment_paths[0], output_path)
        print(f"Single segment copied to {output_path}")
        return True
    
    # Create a concat demuxer file
    concat_file_path = os.path.join(os.path.dirname(output_path), "concat_list.txt")
    try:
        with open(concat_file_path, "w") as f:
            for segment_path in segment_paths:
                # Use absolute path to avoid issues
                abs_path = os.path.abspath(segment_path)
                f.write(f"file '{abs_path}'\n")
        
        print(f"Stitching {len(segment_paths)} segments into final video...")
        cmd_copy = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",  # Copy codec without re-encoding for speed
            "-loglevel", "info",
            output_path,
        ]

        try:
            subprocess.run(cmd_copy, check=True)
        except subprocess.CalledProcessError as e:
            print("⚠ Fast concat failed, retrying with re-encode to normalize streams...")
            cmd_reencode = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-level", "4.1",
                "-r", "30",
                "-c:a", "aac",
                "-ar", "48000",
                "-ac", "2",
                "-movflags", "+faststart",
                "-loglevel", "info",
                output_path,
            ]
            try:
                subprocess.run(cmd_reencode, check=True)
            except subprocess.CalledProcessError:
                print("⚠ Re-encode concat failed, retrying without audio...")
                cmd_video_only = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_file_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-profile:v", "high",
                    "-level", "4.1",
                    "-r", "30",
                    "-an",
                    "-movflags", "+faststart",
                    "-loglevel", "info",
                    output_path,
                ]
                subprocess.run(cmd_video_only, check=True)
        
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ Video stitching completed successfully!")
            print(f"  Output: {output_path}")
            print(f"  Size: {file_size_mb:.1f} MB")
            return True
        else:
            print(f"✗ Stitching failed: Output file not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg command failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Stitching failed with error: {e}")
        return False
    finally:
        # Clean up concat file
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)


def main():
    """Main pipeline to generate videos from templates."""
    # Get the next run index
    run_index = _get_next_run_index(FINAL_OUTPUT_BASE_DIR)
    run_dir = os.path.join(FINAL_OUTPUT_BASE_DIR, str(run_index))
    os.makedirs(run_dir, exist_ok=True)
    
    # Define output paths within the run directory
    image_output_path = os.path.join(run_dir, f"generated_image.png")
    segment_output_dir = os.path.join(run_dir, "segments")
    audio_output_dir = os.path.join(run_dir, "audio")
    final_stitched_output_path = os.path.join(run_dir, "stitched_video_silent.mp4")
    full_audio_path = os.path.join(run_dir, "full_audio.mp3")
    final_video_with_audio_path = os.path.join(run_dir, "final_video_with_lipsync.mp4")
    
    print(f"Run index: {run_index}")
    print(f"Output directory: {run_dir}\n")
    
    # Read prompts from files
    with open(IMAGE_PROMPT_FILE, "r") as f:
        image_prompt = f.read().strip()
    
    with open(VIDEO_PROMPT_FILE, "r") as f:
        video_prompt = f.read().strip()

    stitching_prompts = _load_stitching_prompts(STITCHING_PROMPT_FILE)
    
    # Load timestamped transcript if available
    timestamped_data = _load_timestamped_transcript(TIMESTAMPED_TRANSCRIPT_FILE)

    if MAX_SEGMENTS_TO_GENERATE is not None:
        limit = max(0, int(MAX_SEGMENTS_TO_GENERATE))
        print(f"⚙ Limiting segments to first {limit} for testing")
        if timestamped_data.get("segments"):
            limited_keys = sorted(timestamped_data["segments"].keys())[:limit]
            timestamped_data["segments"] = {
                k: timestamped_data["segments"][k] for k in limited_keys
            }
    
    # Validate person reference image
    person_reference_path = None
    if USE_PERSON_REFERENCE_IMAGE:
        if not os.path.exists(PERSON_REFERENCE_IMAGE_PATH):
            raise FileNotFoundError(
                f"Person reference image not found: {PERSON_REFERENCE_IMAGE_PATH}"
            )
        person_reference_path = PERSON_REFERENCE_IMAGE_PATH

    # Load template JSON to locate source video for first-frame extraction
    source_video_path = None
    if TEMPLATE_JSON_PATH and os.path.exists(TEMPLATE_JSON_PATH):
        with open(TEMPLATE_JSON_PATH, "r") as f:
            template = json.load(f)
        source_video_id = template.get("source_video_id") or template.get("video_id")
        source_username = template.get("source_username") or template.get("username")
        if source_video_id and source_username:
            candidate = os.path.join(SOURCE_VIDEO_BASE_DIR, source_username, f"{source_video_id}.mp4")
            if os.path.exists(candidate):
                source_video_path = candidate
                print(f"Source video found for template: {source_video_path}")
            else:
                print(f"⚠ Source video not found at {candidate}")
                # Fallback: search for the video ID anywhere under SOURCE_VIDEO_BASE_DIR
                try:
                    import glob
                    matches = glob.glob(
                        os.path.join(SOURCE_VIDEO_BASE_DIR, "**", f"{source_video_id}.mp4"),
                        recursive=True,
                    )
                    if matches:
                        source_video_path = matches[0]
                        print(f"✓ Source video found via fallback search: {source_video_path}")
                except Exception as e:
                    print(f"⚠ Fallback search failed: {e}")
        else:
            print("⚠ Template missing source_video_id or source_username; cannot locate source video.")
    else:
        print(f"⚠ Template JSON not found at {TEMPLATE_JSON_PATH}; skipping source frame extraction.")

    # Generate full audio track and slice into segments if enabled
    segment_audio_paths = {}
    if GENERATE_FULL_AUDIO_TRACK and timestamped_data.get("segments"):
        print(f"\n{'='*60}")
        print("Generating full audio track from transcript")
        print(f"{'='*60}")

        if not tts_generator:
            tts_generator = TTSAudioGenerator(
                voice_name=TTS_VOICE_NAME,
                speaking_rate=TTS_SPEAKING_RATE,
                pitch=TTS_PITCH,
            )

        generate_full_audio_from_transcript(
            timestamped_data,
            full_audio_path,
            tts_generator
        )

        os.makedirs(audio_output_dir, exist_ok=True)
        segment_audio_paths = slice_audio_by_timestamps(
            full_audio_path,
            timestamped_data,
            audio_output_dir
        )
    else:
        print("⚠ Skipping audio generation")
    
    if GENERATE_IMAGE:
        base_frame_path = None
        if source_video_path:
            base_frame_path = os.path.join(run_dir, "source_first_frame.png")
            if not os.path.exists(base_frame_path):
                if _extract_first_frame(source_video_path, base_frame_path):
                    print(f"✓ Extracted first frame to {base_frame_path}")
                else:
                    print("⚠ Failed to extract first frame; falling back to prompt-only image generation.")
                    base_frame_path = None

        generate_reference_image(
            image_prompt,
            image_output_path,
            base_frame_path=base_frame_path,
            person_reference_image_path=person_reference_path,
        )

    if GENERATE_VIDEO_WITH_SEGMENT_STITCHING:
        os.makedirs(segment_output_dir, exist_ok=True)

        segment_prompts = _split_segment_prompts(video_prompt)
        if not segment_prompts:
            raise ValueError("No segment prompts found in video prompt file.")

        if MAX_SEGMENTS_TO_GENERATE is not None:
            limit = max(0, int(MAX_SEGMENTS_TO_GENERATE))
            segment_prompts = segment_prompts[:limit]
            if not segment_prompts:
                raise ValueError("Segment limit resulted in zero prompts. Increase MAX_SEGMENTS_TO_GENERATE.")

        # Use all segments from the prompt file (dynamic based on transcript length)
        # Don't artificially limit to TARGET_TOTAL_SECONDS
        target_segments = len(segment_prompts)
        total_expected_seconds = target_segments * SEGMENT_DURATION_SECONDS
        print(f"\n{'='*60}")
        print(f"Generating {target_segments} SILENT video segments ({total_expected_seconds}s total)")
        print(f"{'='*60}")

        # Find the last completed segment (resume from there if incomplete run)
        start_segment = 1
        for check_i in range(target_segments, 0, -1):
            seg_check_path = os.path.join(segment_output_dir, f"segment_{check_i:02d}_silent.mp4")
            if os.path.exists(seg_check_path):
                start_segment = check_i + 1
                print(f"Found existing segment {check_i}, resuming from segment {start_segment}")
                break

        generated_silent_segments = []
        generated_segment_video_objs = []
        # Load previously generated segments if resuming
        for i in range(1, start_segment):
            seg_path = os.path.join(segment_output_dir, f"segment_{i:02d}_silent.mp4")
            if os.path.exists(seg_path):
                generated_silent_segments.append(seg_path)
            generated_segment_video_objs.append(None)
        
        for i, seg_prompt in enumerate(segment_prompts, start=1):
            # Skip if segment already exists
            seg_output = os.path.join(segment_output_dir, f"segment_{i:02d}_silent.mp4")
            if os.path.exists(seg_output):
                print(f"\nSegment {i}/{target_segments} already exists, skipping...")
                generated_silent_segments.append(seg_output)
                generated_segment_video_objs.append(None)
                continue
            
            if i < start_segment:
                continue
            
            # Get transcript context for this segment if available
            transcript_context = None
            if USE_TRANSCRIPT_TIMESTAMPS and i in timestamped_data.get("segments", {}):
                transcript_context = timestamped_data["segments"][i].get("text", "")

            if i == 1 or not EXTEND_SEGMENTS:
                seg_prompt_full = _compose_segment_prompt(
                    seg_prompt,
                    stitching_prompts.get("within_segment", ""),
                )
                ref_image = image_output_path
                prev_video = None
            else:
                prev_prompt = segment_prompts[i - 2]
                delta = seg_prompt
                if EXTEND_REMOVE_OVERLAP:
                    delta = _remove_overlap(prev_prompt, seg_prompt)
                invariants = stitching_prompts.get("between_segments", "")
                seg_prompt_full = _compose_extension_prompt(delta, invariants, transcript_context)

                prev_video_obj = generated_segment_video_objs[-1] if USE_PREVIOUS_VIDEO_CONTEXT else None
                if prev_video_obj:
                    ref_image = image_output_path
                else:
                    last_frame_path = os.path.join(
                        segment_output_dir,
                        f"segment_{i-1:02d}_{LAST_FRAME_IMAGE_NAME}",
                    )
                    if _extract_last_frame(generated_silent_segments[-1], last_frame_path):
                        ref_image = last_frame_path
                    else:
                        ref_image = image_output_path

            print(f"\nGenerating segment {i}/{target_segments}...")
            video_obj = generate_video_from_image(
                video_prompt=seg_prompt_full,
                image_path=ref_image,
                output_path=seg_output,
                aspect_ratio=VIDEO_ASPECT_RATIO,
                duration_seconds=SEGMENT_DURATION_SECONDS,
                previous_video=prev_video_obj if i > 1 else None,
                generate_audio=False,
            )
            generated_silent_segments.append(seg_output)
            generated_segment_video_objs.append(video_obj)

        if APPLY_LIPSYNC and segment_audio_paths:
            print(f"\n{'='*60}")
            print(f"Applying Wav2Lip lip sync to {len(generated_silent_segments)} segments")
            print(f"{'='*60}")

            segments_with_lipsync = []
            for seg_num, silent_seg_path in enumerate(generated_silent_segments, start=1):
                if seg_num in segment_audio_paths:
                    synced_output = os.path.join(
                        segment_output_dir,
                        f"segment_{seg_num:02d}_lipsynced.mp4"
                    )

                    print(f"\n[{seg_num}/{len(generated_silent_segments)}] Lip syncing segment {seg_num}...")
                    success = apply_wav2lip_to_video(
                        silent_seg_path,
                        segment_audio_paths[seg_num],
                        synced_output
                    )

                    if success:
                        segments_with_lipsync.append(synced_output)
                    else:
                        print(f"⚠ Lip sync failed for segment {seg_num}, using silent video")
                        segments_with_lipsync.append(silent_seg_path)
                else:
                    print(f"\n[{seg_num}/{len(generated_silent_segments)}] No audio for segment {seg_num}")
                    segments_with_lipsync.append(silent_seg_path)
        else:
            print("⚠ Skipping lip sync")
            segments_with_lipsync = generated_silent_segments

        # Generate transition prompts (between segments) to guide stitching
        transitions = []
        for i in range(len(segment_prompts) - 1):
            transitions.append(
                _compose_between_segments_prompt(
                    segment_prompts[i],
                    segment_prompts[i + 1],
                    stitching_prompts.get("between_segments", ""),
                )
            )

        # Stitch all segments into final video
        print(f"\n{'='*60}")
        print(f"Stitching {len(segments_with_lipsync)} segments into final video")
        print(f"{'='*60}")

        success = _stitch_videos(segments_with_lipsync, final_video_with_audio_path)
        
        if success:
            # Save manifest with completion info
            manifest_path = os.path.join(segment_output_dir, "stitch_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(
                    {
                        "silent_segment_outputs": generated_silent_segments,
                        "lipsynced_segment_outputs": segments_with_lipsync,
                        "audio_segments": list(segment_audio_paths.values()),
                        "full_audio": full_audio_path if os.path.exists(full_audio_path) else None,
                        "transition_prompts": transitions,
                        "final_output": final_video_with_audio_path,
                        "stitching_status": "completed",
                    },
                    f,
                    indent=2,
                )
            print(f"Stitch manifest saved to {manifest_path}")
        else:
            # Save manifest with failure info
            manifest_path = os.path.join(segment_output_dir, "stitch_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(
                    {
                        "silent_segment_outputs": generated_silent_segments,
                        "lipsynced_segment_outputs": segments_with_lipsync,
                        "audio_segments": list(segment_audio_paths.values()),
                        "full_audio": full_audio_path if os.path.exists(full_audio_path) else None,
                        "transition_prompts": transitions,
                        "final_output": final_video_with_audio_path,
                        "stitching_status": "failed",
                    },
                    f,
                    indent=2,
                )
            print(f"Stitch manifest saved to {manifest_path} (stitching failed)")

    # Create completion marker to indicate this run finished successfully
    completion_marker = os.path.join(run_dir, ".complete")
    with open(completion_marker, "w") as f:
        f.write(f"Run {run_index} completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
