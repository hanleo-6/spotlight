import json
import os
import subprocess
from typing import Dict, Any, List, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv

class TranscriptSegmenter:
    """Segment transcript by time and word count for 8-second chunks."""
    
    def __init__(self, transcript_text: str, segment_duration_seconds: int = 8, actual_video_duration_seconds: float = None):
        self.transcript = transcript_text
        self.segment_duration = segment_duration_seconds
        self.words = transcript_text.split()
        self.total_words = len(self.words)
        
        # If actual video duration is provided, calculate the actual speech rate
        if actual_video_duration_seconds is not None:
            self.estimated_total_seconds = actual_video_duration_seconds
            self.words_per_second = self.total_words / self.estimated_total_seconds if self.estimated_total_seconds > 0 else 2.5
        else:
            # Fallback to default: ~150 words per minute = 2.5 words per second
            self.words_per_second = 2.5
            self.estimated_total_seconds = self.total_words / self.words_per_second
        
    def segment_transcript(self) -> List[Dict[str, Any]]:
        """Split transcript into segments with timestamps and text."""
        segments = []
        words_per_segment = int(self.segment_duration * self.words_per_second)
        
        for i in range(0, self.total_words, words_per_segment):
            segment_words = self.words[i:i + words_per_segment]
            segment_text = " ".join(segment_words)
            
            # Calculate timestamps
            start_seconds = (i / self.words_per_second)
            end_seconds = min((i + len(segment_words)) / self.words_per_second, self.estimated_total_seconds)
            
            segments.append({
                "segment_num": len(segments) + 1,
                "start_time": start_seconds,
                "end_time": end_seconds,
                "duration": end_seconds - start_seconds,
                "text": segment_text,
                "word_count": len(segment_words),
            })
        
        return segments


class TemplatePromptCompiler:
    def __init__(self):
        self.negation_block = (
            "Do not use cinematic camera movement, dramatic lighting, shallow depth of field, "
            "stylized color grading, exaggerated facial expressions, floating UI elements, "
            "overly smooth motion, animated transitions, kinetic typography, or artificial polish."
        )

    def compile(self, template: Dict[str, Any]) -> str:
        sections = [
            self._compile_intent(template),
            self._compile_physical_constraints(template),
            self._compile_human_behavior(template),
            self._compile_story_structure(template),
            self._compile_visual_layout(template),
            self._compile_motion(template),
            self._compile_typography(template),
            self._compile_color_and_lighting(template),
            self._compile_format_mix(template),
            self.negation_block,
            self._compile_outro(template),
        ]

        return " ".join(s for s in sections if s)

    def compile_segments(self, template: Dict[str, Any], transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compile segment-level prompts with transcript context and timestamps.
        Creates one compiled segment per transcript segment (dynamic based on transcript length)."""
        story = template.get("story_structure", {}) or template.get("structure", {})
        content_flow = story.get("content_flow", [])
        
        # Build segment-specific prompts with transcript context
        # One segment per 8-second chunk of transcript
        visual_constraints = self._compile_segment_constraints(template)
        prompts = []
        
        # Map content flow to segments dynamically
        num_transcript_segments = len(transcript_segments)
        
        # If we have content_flow, distribute it across transcript segments
        if content_flow:
            # Calculate how many transcript segments get each content flow item
            segments_per_flow = max(1, num_transcript_segments // len(content_flow))
        
        for idx, transcript_seg in enumerate(transcript_segments, start=1):
            # Determine label based on content_flow or sequence
            if content_flow:
                flow_idx = min((idx - 1) // segments_per_flow, len(content_flow) - 1)
                label = content_flow[flow_idx]
            else:
                label = f"Segment {idx}"
            
            continuity_cue = ""
            if idx > 1:
                continuity_cue = f"Continuing from previous segment. "
            
            seg_prompt = {
                "segment_num": idx,
                "label": label,
                "start_time": transcript_seg["start_time"],
                "end_time": transcript_seg["end_time"],
                "transcript_text": transcript_seg["text"],
                "content_details": f"Focus on {label} aspect of the narrative.",
                "continuity_cue": continuity_cue,
                "visual_constraints": visual_constraints,
                "compiled_prompt": (
                    f"[8-Second Segment {idx}: {label}] ({transcript_seg['start_time']:.1f}s - {transcript_seg['end_time']:.1f}s)\n"
                    f"{continuity_cue}"
                    f"Transcript: {transcript_seg['text']}\n"
                    f"Content: Focus on {label} aspect of the narrative.\n"
                    f"Visual Style: {visual_constraints}"
                ).strip()
            }
            prompts.append(seg_prompt)

        return prompts

    def _single_segment_with_transcript(self, template: Dict[str, Any], transcript_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a single segment with full transcript."""
        base = self.compile(template)
        full_transcript = "\n".join([s["text"] for s in transcript_segments])
        return {
            "segment_num": 1,
            "label": "full_video",
            "start_time": 0.0,
            "end_time": transcript_segments[-1]["end_time"] if transcript_segments else 0.0,
            "transcript_text": full_transcript,
            "content_details": "Full video narrative",
            "visual_constraints": self._compile_segment_constraints(template),
            "compiled_prompt": base
        }

    def _compile_segment_constraints(self, t: Dict[str, Any]) -> str:
        """Compile shared visual and technical constraints for all segments."""
        tech = t.get("technical_execution", {}) or t.get("technical_settings", {})
        motion = t.get("motion_language", {}) or t.get("motion_design", {})
        visual = t.get("visual_system", {}) or t.get("visual_design", {})
        layout = visual.get("layout", {}) if isinstance(visual, dict) else {}
        colors = visual.get("color_system", {}) if isinstance(visual, dict) else {}
        typo = visual.get("typography", {}) if isinstance(visual, dict) else {}

        camera_quality = tech.get('camera_quality', 'smartphone')
        lighting_style = tech.get('lighting_style', 'natural')
        subject_positioning = layout.get('subject_positioning', 'center frame')
        primary_appearance = colors.get('primary', {}).get('appearance', 'vibrant') if isinstance(colors.get('primary'), dict) else 'vibrant'
        headline_style = typo.get('headline_style', 'bold sans-serif')

        return (
            f"Shot on {camera_quality} with {lighting_style} lighting. Subject positioned {subject_positioning}. "
            f"Primary color: {primary_appearance}. Typography: {headline_style}. {self.negation_block}"
        )

    def _compile_intent(self, t: Dict[str, Any]) -> str:
        identity = t.get("template_identity", {})
        personality = identity.get('personality') or t.get('personality', 'engaging')
        template_name = identity.get('template_name') or t.get('template_name', 'video')
        category = identity.get('niche_category') or t.get('category', 'general')
        
        story = t.get('story_structure', {}) or t.get('structure', {})
        storytelling_type = story.get('storytelling_type', 'narrative')
        
        return (
            f"Create a {personality} {template_name.lower()} "
            f"video in the {category} category using a "
            f"{storytelling_type} structure."
        )

    def _compile_physical_constraints(self, t: Dict[str, Any]) -> str:
        tech = t.get("technical_execution", {}) or t.get("technical_settings", {})
        motion = t.get("motion_language", {}) or t.get("motion_design", {})

        camera_quality = tech.get('camera_quality', 'smartphone')
        camera_behavior = motion.get('camera_behavior', 'static')
        lighting_style = tech.get('lighting_style', 'natural')
        depth_of_field = tech.get('depth_of_field', 'normal')
        color_grading = tech.get('color_grading', 'neutral')

        return (
            f"The video is shot on a {camera_quality} camera mounted on a tripod. "
            f"The camera remains {camera_behavior}. "
            f"Lighting is {lighting_style}, with a {depth_of_field} depth of field "
            f"and a slightly {color_grading} tone caused by ambient lighting."
        )

    def _compile_human_behavior(self, t: Dict[str, Any]) -> str:
        story = t.get("story_structure", {}) or t.get("structure", {})
        pacing = story.get("pacing_rhythm", "moderate").replace("-", " ") if story.get("pacing_rhythm") else story.get("pacing", "moderate")
        return (
            f"The speaker delivers lines at a {pacing} pace with natural pauses, "
            f"minor posture shifts, and slightly imperfect eye contact."
        )

    def _compile_story_structure(self, t: Dict[str, Any]) -> str:
        s = t.get("story_structure", {}) or t.get("structure", {})
        content_flow = s.get("content_flow", [])
        flow = " ".join(content_flow) if content_flow else "the main content"
        intro = s.get('intro_characteristics', '') or s.get('intro', 'an attention-grabbing opener')
        hook = s.get('hook_strategy', '') or s.get('hook', 'a compelling hook')
        outro = s.get('outro_characteristics', '') or s.get('outro', 'a clear conclusion')
        
        return (
            f"Begin with {intro} Follow with {hook} "
            f"The narrative progresses as follows: {flow} "
            f"The sequence concludes with {outro}"
        )

    def _compile_visual_layout(self, t: Dict[str, Any]) -> str:
        visual = t.get("visual_system", {}) or t.get("visual_design", {})
        layout = visual.get("layout", {}) if isinstance(visual, dict) else {}
        layer_hierarchy = layout.get("layer_hierarchy", [])
        layers = ", ".join(layer_hierarchy) if layer_hierarchy else "subject, text, background"
        subject_positioning = layout.get('subject_positioning', 'center frame')
        grid_system = layout.get('grid_system', 'rule of thirds')

        return (
            f"The subject is positioned {subject_positioning} "
            f"using the {grid_system}. "
            f"The visual layers appear in this order: {layers}."
        )

    def _compile_motion(self, t: Dict[str, Any]) -> str:
        motion = t.get("motion_language", {}) or t.get("motion_design", {})
        transition_list = motion.get("transitions", [])
        transitions = ", ".join(transition_list) if transition_list else motion.get('transition_style', 'cuts')
        animation_speed = motion.get('animation_speed', 'smooth')
        text_entry = motion.get('text_entry', 'fade')

        return (
            f"Scenes are connected using {transitions}. "
            f"Any movement is {animation_speed}, "
            f"with text entering and exiting via {text_entry}."
        )

    def _compile_typography(self, t: Dict[str, Any]) -> str:
        visual = t.get("visual_system", {}) or t.get("visual_design", {})
        typo = visual.get("typography", {}) if isinstance(visual, dict) else {}
        headline_style = typo.get('headline_style', 'bold sans-serif')
        body_style = typo.get('body_style', 'regular sans-serif')
        alignment = typo.get('alignment', 'center')
        uppercase_pattern = typo.get('uppercase_pattern', 'title case')

        return (
            f"Text overlays use {headline_style} for headlines and "
            f"{body_style} for body text, aligned {alignment}. "
            f"Text appears in {uppercase_pattern} and fades without motion."
        )

    def _compile_color_and_lighting(self, t: Dict[str, Any]) -> str:
        visual = t.get("visual_system", {}) or t.get("visual_design", {})
        colors = visual.get("color_system", {}) if isinstance(visual, dict) else {}
        primary_appearance = colors.get('primary', {}).get('appearance', 'vibrant') if isinstance(colors.get('primary'), dict) else 'vibrant'
        secondary_appearance = colors.get('secondary', {}).get('appearance', 'muted') if isinstance(colors.get('secondary'), dict) else 'muted'
        background_treatment = colors.get('background_treatment', 'simple')
        contrast_strategy = colors.get('contrast_strategy', 'high')

        return (
            f"The primary accent color is {primary_appearance}, "
            f"used sparingly against a {secondary_appearance} background. "
            f"The background treatment is {background_treatment} with "
            f"{contrast_strategy} contrast."
        )

    def _compile_format_mix(self, t: Dict[str, Any]) -> str:
        fmt = t.get("content_formula", {}) or t.get("content_format", {})
        format_mix_ratio = fmt.get('format_mix_ratio', 'mixed content')
        information_density = fmt.get('information_density', 'moderate')
        text_visual_balance = fmt.get('text_visual_balance', 'balanced')

        return (
            f"The video uses a mixed format consisting of {format_mix_ratio}. "
            f"Information density is {information_density} with a "
            f"{text_visual_balance} balance between text and visuals."
        )

    def _compile_outro(self, t: Dict[str, Any]) -> str:
        return "End with a simple logo and tagline on a clean background."


def load_transcript(template: Dict[str, Any]) -> Tuple[str, float]:
    """Load the transcript file corresponding to the template and return (transcript, video_duration)."""
    if "transcript_path" in template:
        transcript_path = template["transcript_path"]
    elif "source_video_id" in template and "source_username" in template:
        source_username = template["source_username"]
        video_id = template["source_video_id"]
        base_transcript_dir = "data/output/transcripts"
        search_pattern = os.path.join(base_transcript_dir, source_username, "*", f"{video_id}_transcript.txt")
        
        import glob
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            transcript_path = matching_files[0]
        else:
            print(f"Warning: No transcript found for video ID {video_id} in {source_username}")
            return None, None
    else:
        print("Warning: No transcript path or source metadata found in template. Proceeding without transcript.")
        return None, None
    
    # Load transcript
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
    else:
        print(f"Warning: Transcript file not found at {transcript_path}")
        return None, None
    
    # Try to find video duration from .info.json file
    video_duration = None
    if "source_video_id" in template and "source_username" in template:
        source_username = template["source_username"]
        video_id = template["source_video_id"]
        
        # Use subprocess to find the .info.json file
        import subprocess
        try:
            result = subprocess.run(
                ["find", "data/input", "-name", f"{video_id}.info.json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            matching_info_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            
            if matching_info_files:
                try:
                    with open(matching_info_files[0], "r", encoding="utf-8") as f:
                        info_data = json.load(f)
                        if "duration" in info_data:
                            video_duration = info_data["duration"]
                            print(f"✓ Found video duration: {video_duration} seconds")
                except Exception as e:
                    print(f"Warning: Could not parse video duration from {matching_info_files[0]}: {e}")
        except Exception as e:
            print(f"Warning: Could not find video info file: {e}")
    
    return transcript_text, video_duration


def refine_with_claude(base_prompt: str, prompt_type: str, transcript_context: str = None) -> str:
    """Refine the compiled prompt using Claude Sonnet 3.5."""
    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")
    
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    client = Anthropic(api_key=api_key)
    
    if prompt_type == "image":
        refinement_instruction = """You are a prompt engineer specializing in image generation. 
Refine the following base prompt to create a detailed, specific image generation prompt for Gemini 3 Pro Image Preview.
Focus on visual composition, lighting, and scene setup for a single frame that will serve as a reference image for video generation.
Keep it concise but highly descriptive. Output only the refined prompt, nothing else."""
    else:  # video
        refinement_instruction = """You are a prompt engineer specializing in video generation.
Refine the following base prompt to create a detailed, specific video generation prompt for Veo 3.1.
Focus on camera movement, action sequences, and temporal progression. Describe what happens over time in the video.
Keep it concise but dynamic and descriptive of all events. Output only the refined prompt, nothing else."""
    
    prompt_content = f"{refinement_instruction}\n\nBase prompt:\n{base_prompt}"
    
    if transcript_context:
        print("  - Transcript context included")
        prompt_content += f"\n\nTranscript for this segment:\n{transcript_context}"
    else:
        print("  - No transcript context")

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt_content}]
    )
    
    return message.content[0].text.strip()


def main():
    """Process template JSON to generate timestamped prompts integrated with transcript."""
    TEMPLATE_JSON_PATH = "data/output/template_database/tiktok_vids/tiktok_vids/7124337427774311722_template.json"
    IMAGE_PROMPT_OUTPUT = "video_gen_pipeline/prompts/image_prompt_stitching.txt"
    VIDEO_PROMPT_OUTPUT = "video_gen_pipeline/prompts/video_prompt_stitching.txt"
    TIMESTAMPED_TRANSCRIPT_OUTPUT = "video_gen_pipeline/prompts/timestamped_transcript.txt"
    STITCHING_PROMPT_OUTPUT = "video_gen_pipeline/prompts/stitching_prompt.txt"
    
    # Load template
    print(f"Loading template from {TEMPLATE_JSON_PATH}...")
    with open(TEMPLATE_JSON_PATH, "r") as f:
        template = json.load(f)
    
    # Load transcript and video duration
    print("Loading transcript and video duration...")
    transcript_text, video_duration = load_transcript(template)
    if not transcript_text:
        raise ValueError("No transcript found. Cannot proceed with transcript-integrated pipeline.")
    
    print(f"✓ Transcript loaded ({len(transcript_text)} characters, ~{len(transcript_text.split())} words)")
    
    # Segment transcript
    print("\nSegmenting transcript by 8-second chunks...")
    segmenter = TranscriptSegmenter(transcript_text, segment_duration_seconds=8, actual_video_duration_seconds=video_duration)
    transcript_segments = segmenter.segment_transcript()
    total_duration = transcript_segments[-1]["end_time"] if transcript_segments else 0
    print(f"✓ Created {len(transcript_segments)} transcript segments ({total_duration:.1f}s total)")
    if video_duration:
        print(f"  (Actual video duration: {video_duration}s, Calculated speech rate: {segmenter.words_per_second:.2f} words/second)")
    for ts in transcript_segments:
        print(f"  [{ts['segment_num']}] {ts['start_time']:.1f}s - {ts['end_time']:.1f}s: {ts['text'][:60]}...")
    
    # Compile base prompt
    print("\nCompiling base prompt...")
    compiler = TemplatePromptCompiler()
    base_prompt = compiler.compile(template)
    print(f"✓ Base prompt compiled")

    # Compile segment prompts with transcript integration
    print("\nCompiling segment prompts with transcript context...")
    segment_prompts = compiler.compile_segments(template, transcript_segments)
    print(f"✓ Generated {len(segment_prompts)} segment prompts (dynamic based on transcript)")
    for i, sp in enumerate(segment_prompts, 1):
        print(f"  [{i}] {sp['label']} ({sp['start_time']:.1f}s - {sp['end_time']:.1f}s)")
    
    # Refine image prompt
    print("\nRefining image prompt with Claude...")
    image_prompt = refine_with_claude(base_prompt, "image", transcript_text[:500])
    print(f"✓ Image prompt refined")
    
    # Refine video segment prompts
    print("\nRefining video segment prompts with Claude...")
    refined_segments = []
    for i, seg_prompt in enumerate(segment_prompts, start=1):
        print(f"\n  [Segment {i}/{len(segment_prompts)}] {seg_prompt['label']} ({seg_prompt['start_time']:.1f}s)")
        refined = refine_with_claude(
            seg_prompt['compiled_prompt'],
            "video",
            seg_prompt['transcript_text']
        )
        refined_segments.append({
            **seg_prompt,
            "refined_prompt": refined
        })
        print(f"    ✓ Refined: {refined[:80]}...")

    # Generate video prompt with timestamps
    video_prompt_lines = []
    for seg in refined_segments:
        video_prompt_lines.append(
            f"[Segment {seg['segment_num']}: {seg['label']} ({seg['start_time']:.1f}s - {seg['end_time']:.1f}s)]\n"
            f"{seg['refined_prompt']}"
        )
    video_prompt = "\n---SEGMENT_BREAK---\n".join(video_prompt_lines)
    
    # Generate timestamped transcript
    timestamped_transcript_lines = ["TIMESTAMPED TRANSCRIPT"]
    timestamped_transcript_lines.append("=" * 80)
    for seg in refined_segments:
        timestamped_transcript_lines.append(
            f"\n[Segment {seg['segment_num']}: {seg['label']}]"
        )
        timestamped_transcript_lines.append(f"Time: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s (duration: {seg['end_time'] - seg['start_time']:.1f}s)")
        timestamped_transcript_lines.append(f"\nTranscript:\n{seg['transcript_text']}")
        timestamped_transcript_lines.append("-" * 80)
    
    # Save outputs
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")
    
    with open(IMAGE_PROMPT_OUTPUT, "w") as f:
        f.write(image_prompt)
    print(f"✓ Image prompt saved to {IMAGE_PROMPT_OUTPUT}")
    
    with open(VIDEO_PROMPT_OUTPUT, "w") as f:
        f.write(video_prompt)
    print(f"✓ Video prompt saved to {VIDEO_PROMPT_OUTPUT}")
    
    with open(TIMESTAMPED_TRANSCRIPT_OUTPUT, "w") as f:
        f.write("\n".join(timestamped_transcript_lines))
    print(f"✓ Timestamped transcript saved to {TIMESTAMPED_TRANSCRIPT_OUTPUT}")

    with open(STITCHING_PROMPT_OUTPUT, "w") as f:
        f.write(
            "[WITHIN_SEGMENT]\n"
            "Ensure each segment contributes a clear mini-arc (intro → demo → CTA). "
            "If a beat (e.g., the demo) needs more time, allow it to continue across adjacent segments, "
            "but preserve the mini-arc across the segment group so the viewer experiences a complete arc. "
            "Across the full video, maintain a coherent macro-arc made of sections built from these segments. "
            "Maintain strict visual continuity within the segment: same people, wardrobe, props, devices, "
            "set dressing, camera position, lighting, color tone, and typography. Avoid introducing new objects "
            "or changing fonts, on-screen styling, or layout mid-segment.\n\n"
            "[BETWEEN_SEGMENTS]\n"
            "Create seamless handoffs between segments by preserving the prior clip's visual identity. "
            "Keep the same people (faces, hair, wardrobe), the same objects and devices (e.g., laptops, phones), "
            "the same environment, camera framing, lighting, color balance, and motion style. "
            "Typography must remain identical in font, weight, size, color, and placement. "
            "Only advance the action or dialogue; do not introduce new subjects or change the look and feel.\n"
        )
    print(f"✓ Stitching prompt saved to {STITCHING_PROMPT_OUTPUT}")
    
    print(f"\n{'='*60}")
    print("✓ Prompt generation with transcript integration completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
