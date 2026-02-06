import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None


load_dotenv()

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE VALUES FOR YOUR USE CASE
# =============================================================================

# Input
TRANSCRIPT_FILE = "video_gen_pipeline/prompts/talking_head_transcript.txt"
PERSON_REFERENCE_IMAGE_PATH = "video_gen_pipeline/assets/person_reference_talking_head_1024.jpg"

# Output
FINAL_OUTPUT_BASE_DIR = "data/output/generated_videos/talking_head"
OUTPUT_BASENAME = "talking_head"

# ElevenLabs
ELEVENLABS_VOICE = "TX3LPaxmHKxFdv7VOQHJ"  # Liam - Energetic, Social Media Creator
ELEVENLABS_MODEL = "eleven_turbo_v2_5"

# D-ID
DID_ENHANCE = True
DID_MAX_WAIT_SECONDS = 300

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================


def _get_next_run_index(base_dir: str) -> int:
    """Get the next index for a new run. Reuses the last folder if incomplete."""
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

    last_run_dir = os.path.join(base_dir, str(max_index))
    completion_marker = os.path.join(last_run_dir, ".complete")

    if not os.path.exists(completion_marker):
        return max_index

    return max_index + 1


def _read_transcript_text(path: str) -> str:
    """Read a plain transcript file and return its content."""
    if not os.path.exists(path):
        return ""

    with open(path, "r") as f:
        content = f.read().strip()

    return content


class TalkingHeadPipeline:
    """Pipeline for generating a talking head video with ElevenLabs + D-ID."""

    def __init__(self, elevenlabs_api_key: str, did_api_key: str, output_dir: str) -> None:
        self.elevenlabs_api_key = elevenlabs_api_key
        self.did_api_key = did_api_key
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_audio_elevenlabs(
        self,
        text: str,
        output_path: str,
        voice: str = ELEVENLABS_VOICE,
        model: str = ELEVENLABS_MODEL,
    ) -> bool:
        """Generate audio using ElevenLabs."""
        if ElevenLabs is None:
            print("✗ ElevenLabs SDK not installed. Run: pip install elevenlabs")
            return False

        if os.path.exists(output_path):
            file_size_kb = os.path.getsize(output_path) / 1024
            if file_size_kb > 0:
                print(f"⊘ Audio already exists: {output_path} ({file_size_kb:.1f} KB)")
                return True
            else:
                print(f"  Found empty audio file, regenerating...")
                os.remove(output_path)

        try:
            client = ElevenLabs(api_key=self.elevenlabs_api_key)
            voice_id = self._resolve_elevenlabs_voice_id(client, voice)
            audio_iter = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model,
            )
            with open(output_path, "wb") as f:
                for chunk in audio_iter:
                    f.write(chunk)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"✗ Audio generation failed: {e}")
            return False

    def _resolve_elevenlabs_voice_id(self, client: "ElevenLabs", voice: str) -> str:
        """Return a voice_id, resolving from name when possible."""
        try:
            voices = client.voices.get_all().voices
            for item in voices:
                if item.name and item.name.lower() == voice.lower():
                    return item.voice_id
        except Exception:
            pass
        return voice

    def generate_talking_head_did(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        enhance: bool = DID_ENHANCE,
    ) -> bool:
        """Generate talking head video using D-ID."""
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return False
        if not os.path.exists(audio_path):
            print(f"✗ Audio not found: {audio_path}")
            return False

        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            if file_size_mb > 0:
                print(f"⊘ Video already exists: {output_path} ({file_size_mb:.1f} MB)")
                return True
            else:
                print(f"  Found empty video file, regenerating...")
                os.remove(output_path)

        try:
            # Validate image before upload
            print("Validating image...")
            validated_image_path = self._validate_image(image_path)
            
            print("Creating D-ID talk...")
            talk_id = self._create_did_talk_direct(validated_image_path, audio_path, enhance)
            print(f"Talk ID: {talk_id}")
            
            print("Waiting for video generation...")
            video_url = self._poll_did_status(talk_id)
            if not video_url:
                return False
            
            print("Downloading video...")
            return self._download_video(video_url, output_path)
        except Exception as e:
            print(f"✗ Talking head generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _validate_image(self, image_path: str) -> str:
        """Validate and optimize image for D-ID."""
        try:
            from PIL import Image
        except ImportError:
            print("  Pillow not installed, skipping validation")
            return image_path
        
        img = Image.open(image_path)
        print(f"  Original: {img.size}, {img.mode}")
        
        needs_conversion = False
        
        # Convert to RGB
        if img.mode != 'RGB':
            print(f"  Converting {img.mode} → RGB")
            img = img.convert('RGB')
            needs_conversion = True
        
        # Resize if too large
        max_dim = 1024
        if img.size[0] > max_dim or img.size[1] > max_dim:
            print(f"  Resizing to fit {max_dim}px")
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            needs_conversion = True
        
        # Save optimized version if needed
        if needs_conversion:
            base, _ = os.path.splitext(image_path)
            optimized_path = f"{base}_optimized.jpg"
            img.save(optimized_path, 'JPEG', quality=95, optimize=True)
            file_size = os.path.getsize(optimized_path) / 1024
            print(f"✓ Optimized: {img.size}, {file_size:.1f} KB")
            return optimized_path
        
        return image_path

    def _upload_image_to_did(self, image_path: str) -> str:
        """Upload image to D-ID storage and return the URL."""
        print(f"  Uploading image to D-ID...")

        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(
                "https://api.d-id.com/images",
                headers=self._did_auth_header(),
                files=files,
                timeout=60
            )

        if not response.ok:
            print(f"  Image upload failed: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            response.raise_for_status()

        result = response.json()
        image_url = result.get("url")
        print(f"  Image uploaded: {image_url}")
        return image_url

    def _upload_audio_to_did(self, audio_path: str) -> str:
        """Upload audio to D-ID storage and return the URL."""
        print(f"  Uploading audio to D-ID...")

        with open(audio_path, "rb") as f:
            files = {"audio": (os.path.basename(audio_path), f, "audio/mpeg")}
            response = requests.post(
                "https://api.d-id.com/audios",
                headers=self._did_auth_header(),
                files=files,
                timeout=60
            )

        if not response.ok:
            print(f"  Audio upload failed: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            response.raise_for_status()

        result = response.json()
        audio_url = result.get("url")
        print(f"  Audio uploaded: {audio_url}")
        return audio_url

    def _create_did_talk_direct(self, image_path: str, audio_path: str, enhance: bool) -> str:
        """Create D-ID talk by uploading files first (avoids payload size limits)."""

        # Upload files to D-ID storage first
        image_url = self._upload_image_to_did(image_path)
        audio_url = self._upload_audio_to_did(audio_path)

        # Build payload with uploaded file URLs
        payload = {
            "source_url": image_url,
            "script": {
                "type": "audio",
                "audio_url": audio_url,
            },
            "config": {
                "fluent": True,
                "pad_audio": 0.0,
                "result_format": "mp4",
            },
        }

        if enhance:
            payload["config"]["auto_match"] = True

        # Create talk
        print("  Creating talk...")
        response = requests.post(
            "https://api.d-id.com/talks",
            headers={**self._did_auth_header(), "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        if not response.ok:
            print(f"✗ Request failed: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            response.raise_for_status()

        return response.json()["id"]

    def _did_auth_header(self) -> dict:
        """Create D-ID authorization header.
        
        D-ID API keys are pre-encoded base64 strings in format:
        base64_encoded_username:base64_encoded_password
        
        Just add 'Basic ' prefix.
        """
        token = (self.did_api_key or "").strip()
        
        if not token:
            raise ValueError("DID_API_KEY is empty")
        
        # If already has Basic prefix, use as-is
        if token.lower().startswith("basic "):
            return {"Authorization": token}
        
        # Add Basic prefix to the token
        return {"Authorization": f"Basic {token}"}

    def _poll_did_status(self, talk_id: str, max_wait: int = DID_MAX_WAIT_SECONDS) -> Optional[str]:
        """Poll D-ID for video completion."""
        start_time = time.time()
        last_status = None

        while time.time() - start_time < max_wait:
            response = requests.get(
                f"https://api.d-id.com/talks/{talk_id}",
                headers=self._did_auth_header(),
            )
            response.raise_for_status()
            data = response.json()
            status = data.get("status")

            if status != last_status:
                elapsed = time.time() - start_time
                print(f"  Status: {status} ({elapsed:.1f}s)")
                last_status = status

            if status == "done":
                return data.get("result_url")

            if status == "error":
                error_msg = data.get("error", {}).get("description", "Unknown")
                print(f"✗ D-ID error: {error_msg}")
                return None

            time.sleep(3)

        print(f"✗ Timeout after {max_wait}s")
        return None

    def _download_video(self, video_url: str, output_path: str) -> bool:
        """Download video from URL."""
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ Downloaded: {file_size_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False


def main() -> None:
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    did_key = os.getenv("DID_API_KEY")

    if not elevenlabs_key:
        print("✗ ELEVENLABS_API_KEY not found")
        return
    if not did_key:
        print("✗ DID_API_KEY not found")
        return

    transcript_text = _read_transcript_text(TRANSCRIPT_FILE)
    if not transcript_text:
        print(f"✗ Transcript empty: {TRANSCRIPT_FILE}")
        return

    if not os.path.exists(PERSON_REFERENCE_IMAGE_PATH):
        print(f"✗ Image not found: {PERSON_REFERENCE_IMAGE_PATH}")
        return

    run_index = _get_next_run_index(FINAL_OUTPUT_BASE_DIR)
    run_dir = os.path.join(FINAL_OUTPUT_BASE_DIR, str(run_index))
    os.makedirs(run_dir, exist_ok=True)

    audio_path = os.path.join(run_dir, f"{OUTPUT_BASENAME}_audio.mp3")
    video_path = os.path.join(run_dir, f"{OUTPUT_BASENAME}.mp4")

    print(f"\n{'='*60}")
    print(f"Talking Head Video Generation - Run {run_index}")
    print(f"{'='*60}\n")

    pipeline = TalkingHeadPipeline(
        elevenlabs_api_key=elevenlabs_key,
        did_api_key=did_key,
        output_dir=run_dir,
    )

    print("STEP 1: Audio Generation")
    print("-" * 60)
    audio_ok = pipeline.generate_audio_elevenlabs(
        text=transcript_text,
        output_path=audio_path,
    )
    if not audio_ok:
        print("\n✗ Failed at audio generation")
        return

    print("\nSTEP 2: Video Generation")
    print("-" * 60)
    video_ok = pipeline.generate_talking_head_did(
        image_path=PERSON_REFERENCE_IMAGE_PATH,
        audio_path=audio_path,
        output_path=video_path,
        enhance=DID_ENHANCE,
    )
    if not video_ok:
        print("\n✗ Failed at video generation")
        return

    completion_marker = os.path.join(run_dir, ".complete")
    with open(completion_marker, "w") as f:
        f.write(f"Run {run_index} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n{'='*60}")
    print("✓ SUCCESS!")
    print(f"{'='*60}")
    print(f"Output: {video_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()