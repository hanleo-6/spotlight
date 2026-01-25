"""
Module 2: Template Extraction
Extracts video template profiles including scenes, text overlays, audio patterns, and visual features.
"""
import cv2
import json
import re
import subprocess
import tempfile
from pathlib import Path
from collections import Counter
import numpy as np
import whisper
import easyocr
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


class TemplateExtractor:
    """Extract video template profiles from video files."""
    
    def __init__(self, workspace_root: Path = None):
        """Initialize template extractor."""
        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parent.parent
        
        self.workspace_root = workspace_root
        self.reader = easyocr.Reader(['en'])
        self.whisper_model = whisper.load_model("base")
    
    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================
    
    def _to_seconds(self, obj):
        """Convert FrameTimecode/number/tuple to seconds robustly."""
        try:
            if hasattr(obj, 'get_seconds'):
                return float(obj.get_seconds())
        except Exception:
            pass
        return float(obj[-1]) if isinstance(obj, tuple) and obj else float(obj) if isinstance(obj, (int, float)) else 0.0
    
    def _get_vm_duration_seconds(self, video_manager):
        """Get VideoManager duration in seconds."""
        try:
            return self._to_seconds(video_manager.get_duration())
        except Exception:
            return 0.0
    
    def _scene_bounds_seconds(self, scene):
        """Return (start_seconds, end_seconds) for a scene tuple."""
        try:
            if isinstance(scene, tuple) and len(scene) >= 2:
                return (self._to_seconds(scene[0]), self._to_seconds(scene[1]))
        except Exception:
            pass
        return (0.0, 0.0)
    
    def _extract_audio_segment(self, video_path, start_sec, end_sec):
        """Extract audio segment from video, return path to temp WAV file."""
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ss", str(start_sec), "-to", str(end_sec),
            "-ac", "1", "-ar", "16000", tmp.name
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return Path(tmp.name)
    
    # ============================================================================
    # SCENE DETECTION
    # ============================================================================
    
    def detect_scenes(self, video_path: Path, threshold: float = 30.0):
        """Detect scene cuts using PySceneDetect."""
        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        scene_list = scene_manager.get_scene_list()
        duration = self._get_vm_duration_seconds(video_manager)
        video_manager.release()
        
        scenes = [self._scene_bounds_seconds(s) for s in scene_list]
        
        # Calculate scene metrics
        if scenes:
            durations = [end - start for start, end in scenes]
            avg_scene_length = sum(durations) / len(durations)
            median_scene_length = sorted(durations)[len(durations) // 2]
        else:
            avg_scene_length = duration
            median_scene_length = duration
        
        return {
            "total_scenes": len(scenes),
            "scene_list": scenes,
            "avg_scene_length": round(avg_scene_length, 2),
            "median_scene_length": round(median_scene_length, 2),
        }
    
    # ============================================================================
    # TEXT OVERLAY DETECTION
    # ============================================================================
    
    def detect_text_overlays(self, video_path: Path, sample_rate: float = 1.0):
        """Detect on-screen text overlays using OCR."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        interval = int(fps * sample_rate)
        text_detections = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % interval == 0:
                timestamp = frame_idx / fps
                result = self.reader.readtext(frame, paragraph=False)
                
                for (bbox, text, conf) in result:
                    if conf > 0.5:
                        x_coords = [pt[0] for pt in bbox]
                        y_coords = [pt[1] for pt in bbox]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)
                        
                        text_detections.append({
                            "timestamp": round(timestamp, 2),
                            "text": text,
                            "confidence": round(conf, 2),
                            "position": {"x": int(center_x), "y": int(center_y)}
                        })
            
            frame_idx += 1
        
        cap.release()
        
        # Analyze text patterns
        timestamps_with_text = set(d["timestamp"] for d in text_detections)
        coverage = len(timestamps_with_text) / (duration / sample_rate) if duration > 0 else 0
        
        all_texts = [d["text"] for d in text_detections]
        common_phrases = Counter(all_texts).most_common(5)
        
        # Determine overlay zones (top, middle, bottom thirds)
        height = 720  # Assume standard height
        zones = {"top": 0, "middle": 0, "bottom": 0}
        for d in text_detections:
            y = d["position"]["y"]
            if y < height / 3:
                zones["top"] += 1
            elif y < 2 * height / 3:
                zones["middle"] += 1
            else:
                zones["bottom"] += 1
        
        total_detections = sum(zones.values())
        common_zones = [k for k, v in sorted(zones.items(), key=lambda x: -x[1]) if v > 0]
        
        return {
            "total_detections": len(text_detections),
            "text_coverage_percent": round(coverage * 100, 1),
            "common_phrases": [{"text": text, "count": count} for text, count in common_phrases],
            "common_zones": common_zones,
            "overlay_timing_pattern": "frequent" if coverage > 0.5 else "sparse",
            "detections": text_detections[:50]  # Limit to first 50 for storage
        }
    
    # ============================================================================
    # AUDIO TRANSCRIPTION
    # ============================================================================
    
    def transcribe_audio(self, video_path: Path):
        """Transcribe audio using Whisper."""
        result = self.whisper_model.transcribe(str(video_path))
        
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            })
        
        full_text = result.get("text", "").strip()
        word_count = len(full_text.split())
        speaking_time = sum(seg["end"] - seg["start"] for seg in segments)
        
        return {
            "full_transcription": full_text,
            "word_count": word_count,
            "speaking_time_seconds": round(speaking_time, 2),
            "segments": segments
        }
    
    # ============================================================================
    # VISUAL ANALYSIS
    # ============================================================================
    
    def analyze_visual_features(self, video_path: Path, sample_frames: int = 10):
        """Analyze visual features like color palette and brightness."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {"error": "Could not read video"}
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        colors = []
        brightness_values = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Dominant colors
            pixels = frame.reshape(-1, 3)
            avg_color = pixels.mean(axis=0)
            colors.append(avg_color)
            
            # Brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            brightness_values.append(brightness)
        
        cap.release()
        
        if colors:
            avg_rgb = np.mean(colors, axis=0)
            avg_brightness = np.mean(brightness_values)
        else:
            avg_rgb = [0, 0, 0]
            avg_brightness = 0
        
        return {
            "dominant_color_rgb": [int(c) for c in avg_rgb],
            "avg_brightness": round(avg_brightness, 2),
            "brightness_variance": round(np.std(brightness_values), 2) if brightness_values else 0
        }
    
    # ============================================================================
    # MAIN EXTRACTION PIPELINE
    # ============================================================================
    
    def extract_template(self, video_path: Path, video_id: str = None, niche: str = "unknown"):
        """
        Extract complete template profile for a video.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for video
            niche: Classified niche category
        
        Returns:
            dict: Template profile
        """
        if video_id is None:
            video_id = video_path.stem
        
        print(f"Extracting template for: {video_path.name}")
        
        # Get video duration
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        template = {
            "video_id": video_id,
            "filename": video_path.name,
            "niche": niche,
            "duration": round(duration, 2),
            "scenes": self.detect_scenes(video_path),
            "text_overlays": self.detect_text_overlays(video_path),
            "audio": self.transcribe_audio(video_path),
            "visual": self.analyze_visual_features(video_path)
        }
        
        print(f"  ✓ Scenes: {template['scenes']['total_scenes']}")
        print(f"  ✓ Text detections: {template['text_overlays']['total_detections']}")
        print(f"  ✓ Word count: {template['audio']['word_count']}")
        
        return template
    
    def extract_templates_batch(self, video_dir: Path, classifications_csv: Path, output_dir: Path):
        """
        Extract templates for all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
            classifications_csv: CSV file with video classifications
            output_dir: Directory to save template JSON files
        """
        import csv
        
        # Load classifications
        classifications = {}
        if classifications_csv.exists():
            with classifications_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    classifications[row["video_id"]] = row["niche"]
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos
        video_files = sorted(video_dir.glob("**/*.mp4"))
        print(f"\nExtracting templates for {len(video_files)} videos...\n")
        
        for video_path in video_files:
            video_id = video_path.stem
            niche = classifications.get(video_id, "unknown")
            
            template = self.extract_template(video_path, video_id, niche)
            
            # Save template
            output_file = output_dir / f"{video_id}_template.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(template, f, indent=2)
            
            print(f"  Saved: {output_file.name}\n")
        
        print(f"Template extraction complete. Saved to: {output_dir}")


def main():
    """Standalone execution for best_media template extraction."""
    workspace_root = Path(__file__).resolve().parent.parent
    
    extractor = TemplateExtractor(workspace_root)
    extractor.extract_templates_batch(
        video_dir=workspace_root / "best_media",
        classifications_csv=workspace_root / "data" / "output" / "video_classifications.csv",
        output_dir=workspace_root / "data" / "output" / "templates"
    )


if __name__ == "__main__":
    main()
