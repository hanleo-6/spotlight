"""Text-to-Speech audio generation with consistent voice across segments."""

import os
from typing import Dict, Any, List, Optional
from google.cloud import texttospeech
from dotenv import load_dotenv


class TTSAudioGenerator:
    """Generate audio using Google Cloud Text-to-Speech with consistent voice."""
    
    def __init__(
        self,
        voice_name: str = "en-US-Neural2-C",
        language_code: str = "en-US",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ):
        """
        Initialize TTS generator with consistent voice settings.
        
        Args:
            voice_name: Google Cloud voice name (e.g., "en-US-Neural2-C", "en-US-Neural2-A")
            language_code: Language code (e.g., "en-US", "en-GB")
            speaking_rate: Speech rate multiplier (0.25 to 4.0, default 1.0)
            pitch: Pitch adjustment in semitones (-20 to 20, default 0.0)
        """
        load_dotenv()
        
        # Initialize Google Cloud TTS client using API key or service account
        api_key = os.getenv("GOOGLE_TTS_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if api_key:
            # Use API key authentication
            self.client = texttospeech.TextToSpeechClient(
                client_options={"api_key": api_key}
            )
        else:
            # Fall back to service account credentials from GOOGLE_APPLICATION_CREDENTIALS
            self.client = texttospeech.TextToSpeechClient()
        
        self.voice_name = voice_name
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        self.pitch = pitch
    
    def generate_audio(
        self,
        text: str,
        output_path: str,
        audio_encoding: str = "MP3",
    ) -> bool:
        """
        Generate audio for a single segment.
        
        Args:
            text: Text to convert to speech
            output_path: Path where audio file will be saved
            audio_encoding: Audio format ("MP3", "LINEAR16", "OGG_OPUS")
        
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            print(f"Warning: Empty text for {output_path}, skipping.")
            return False
        
        try:
            # Configure voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=self.voice_name,
            )
            
            # Configure audio encoding
            audio_config = texttospeech.AudioConfig(
                audio_encoding=getattr(texttospeech.AudioEncoding, audio_encoding),
                speaking_rate=self.speaking_rate,
                pitch=self.pitch,
            )
            
            # Make API request
            input_text = texttospeech.SynthesisInput(text=text)
            response = self.client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config,
            )
            
            # Save audio file
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
            
            file_size_kb = os.path.getsize(output_path) / 1024
            print(f"✓ Audio generated: {output_path} ({file_size_kb:.1f} KB)")
            return True
            
        except Exception as e:
            print(f"✗ Error generating audio for {output_path}: {e}")
            return False
    
    def generate_segment_audio(
        self,
        segments: Dict[int, Dict[str, Any]],
        output_dir: str,
        audio_encoding: str = "MP3",
    ) -> Dict[int, str]:
        """
        Generate audio for all transcript segments.
        
        Args:
            segments: Dictionary of segment data with text (e.g., from timestamped_transcript)
            output_dir: Directory where audio files will be saved
            audio_encoding: Audio format
        
        Returns:
            Dictionary mapping segment numbers to audio file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        audio_paths = {}
        for seg_num in sorted(segments.keys()):
            seg_data = segments[seg_num]
            text = seg_data.get("text", "")
            
            audio_path = os.path.join(output_dir, f"segment_{seg_num:02d}_audio.mp3")
            
            # Skip if audio file already exists
            if os.path.exists(audio_path):
                file_size_kb = os.path.getsize(audio_path) / 1024
                print(f"⊘ Audio already exists: {audio_path} ({file_size_kb:.1f} KB)")
                audio_paths[seg_num] = audio_path
            elif self.generate_audio(text, audio_path, audio_encoding):
                audio_paths[seg_num] = audio_path
        
        return audio_paths


def overlay_audio_on_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_delay_seconds: float = 0.0,
) -> bool:
    """
    Overlay audio onto video using ffmpeg.
    
    Args:
        video_path: Path to video file (may or may not have audio)
        audio_path: Path to audio file to overlay
        output_path: Path where output video will be saved
        audio_delay_seconds: Delay before audio starts (in seconds)
    
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return False
    
    if not os.path.exists(audio_path):
        print(f"✗ Audio file not found: {audio_path}")
        return False
    
    try:
        delay_filter = ""
        if audio_delay_seconds > 0:
            # Add delay to audio using adelay filter
            delay_ms = int(audio_delay_seconds * 1000)
            delay_filter = f",adelay={delay_ms}|{delay_ms}"
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",  # Copy video codec without re-encoding
            "-c:a", "aac",   # Re-encode audio to AAC
            "-map", "0:v:0",  # Use video from first input
            "-map", "1:a:0",  # Use audio from second input
            "-shortest",  # Finish when shortest stream ends
            output_path,
        ]
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Audio overlaid on video: {output_path} ({file_size_mb:.1f} MB)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error overlaying audio: {e}")
        return False


def stitch_audio_segments(
    audio_paths: List[str],
    output_path: str,
) -> bool:
    """
    Stitch multiple audio segments into one file using ffmpeg concat demuxer.
    
    Args:
        audio_paths: List of paths to audio segments in order
        output_path: Path where final stitched audio will be saved
    
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    
    if not audio_paths:
        print("✗ No audio segments to stitch")
        return False
    
    if len(audio_paths) == 1:
        # If only one segment, just copy it
        import shutil
        try:
            shutil.copy(audio_paths[0], output_path)
            print(f"✓ Single audio segment copied to {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error copying audio: {e}")
            return False
    
    # Create concat demuxer file
    concat_file_path = os.path.join(os.path.dirname(output_path), "concat_audio_list.txt")
    
    try:
        with open(concat_file_path, "w") as f:
            for audio_path in audio_paths:
                abs_path = os.path.abspath(audio_path)
                f.write(f"file '{abs_path}'\n")
        
        print(f"Stitching {len(audio_paths)} audio segments...")
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",  # Copy without re-encoding for speed
            output_path,
        ]
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ Audio stitching completed: {output_path} ({file_size_mb:.1f} MB)")
            return True
        else:
            print(f"✗ Audio stitching failed: Output file not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg command failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Audio stitching failed: {e}")
        return False
    finally:
        # Clean up concat file
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)
