import json
import os
from typing import Dict, Any
from anthropic import Anthropic
from dotenv import load_dotenv

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

    def _compile_intent(self, t: Dict[str, Any]) -> str:
        personality = t.get('personality', 'engaging')
        template_name = t.get('template_name', 'video')
        category = t.get('category', 'general')
        storytelling_type = t.get('story_structure', {}).get('storytelling_type', 'narrative')
        return (
            f"Create a {personality} {template_name.lower()} "
            f"video in the {category} category using a "
            f"{storytelling_type} structure."
        )

    def _compile_physical_constraints(self, t: Dict[str, Any]) -> str:
        tech = t.get("technical_settings", {})
        motion = t.get("motion_design", {})

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
        pacing = t.get("story_structure", {}).get("pacing", "moderate")
        return (
            f"The speaker delivers lines at a {pacing} pace with natural pauses, "
            f"minor posture shifts, and slightly imperfect eye contact."
        )

    def _compile_story_structure(self, t: Dict[str, Any]) -> str:
        s = t.get("story_structure", {})
        content_flow = s.get("content_flow", [])
        flow = " ".join(content_flow) if content_flow else "the main content"
        intro = s.get('intro', 'an attention-grabbing opener')
        hook = s.get('hook', 'a compelling hook')
        outro = s.get('outro', 'a clear conclusion')
        return (
            f"Begin with {intro} Follow with {hook} "
            f"The narrative progresses as follows: {flow} "
            f"The sequence concludes with {outro}"
        )

    def _compile_visual_layout(self, t: Dict[str, Any]) -> str:
        layout = t.get("visual_design", {}).get("layout", {})
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
        motion = t.get("motion_design", {})
        transition_list = motion.get("transitions", [])
        transitions = ", ".join(transition_list) if transition_list else "cuts"
        animation_speed = motion.get('animation_speed', 'smooth')
        text_entry = motion.get('text_entry', 'fade')

        return (
            f"Scenes are connected using {transitions}. "
            f"Any movement is {animation_speed}, "
            f"with text entering and exiting via {text_entry}."
        )

    def _compile_typography(self, t: Dict[str, Any]) -> str:
        typo = t.get("visual_design", {}).get("typography", {})
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
        colors = t.get("visual_design", {}).get("color_system", {})
        primary_appearance = colors.get('primary', {}).get('appearance', 'vibrant')
        secondary_appearance = colors.get('secondary', {}).get('appearance', 'muted')
        background_treatment = colors.get('background_treatment', 'simple')
        contrast_strategy = colors.get('contrast_strategy', 'high')

        return (
            f"The primary accent color is {primary_appearance}, "
            f"used sparingly against a {secondary_appearance} background. "
            f"The background treatment is {background_treatment} with "
            f"{contrast_strategy} contrast."
        )

    def _compile_format_mix(self, t: Dict[str, Any]) -> str:
        fmt = t.get("content_format", {})
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


def load_transcript(template: Dict[str, Any]) -> str:
    """Load the transcript file corresponding to the template."""
    # Try to get transcript path from template metadata
    if "transcript_path" in template:
        transcript_path = template["transcript_path"]
    elif "source_video_id" in template and "source_username" in template:
        # Derive transcript path from template metadata
        # Template structure: source_username and source_video_id
        # Need to find the actual creator folder by searching the transcript directory
        source_username = template["source_username"]
        video_id = template["source_video_id"]
        
        # Base directory for transcripts
        base_transcript_dir = "data/output/transcripts"
        search_pattern = os.path.join(base_transcript_dir, source_username, "*", f"{video_id}_transcript.txt")
        
        # Use glob to find the transcript file
        import glob
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            transcript_path = matching_files[0]
        else:
            print(f"Warning: No transcript found for video ID {video_id} in {source_username}")
            print(f"Searched in: {search_pattern}")
            return None
    else:
        print("Warning: No transcript path or source metadata found in template. Proceeding without transcript.")
        return None
    
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        print(f"Warning: Transcript file not found at {transcript_path}")
        return None


def refine_with_claude(base_prompt: str, prompt_type: str, transcript: str = None) -> str:
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
    
    # Build the prompt content
    prompt_content = f"{refinement_instruction}\n\nBase prompt:\n{base_prompt}"
    
    if transcript:
        print("Transcript used")
        prompt_content += f"\n\nVideo transcript (use this to understand the content and narrative):\n{transcript}"
    else:
        print("Transcript not found")

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt_content
            }
        ]
    )
    
    return message.content[0].text.strip()


def main():
    """Process template JSON to generate refined image and video prompts."""
    # Configuration
    # TEMPLATE_JSON_PATH = "data/output/template_database/tiktok_vids/tiktok_vids/7584401832210877713_template.json"
    TEMPLATE_JSON_PATH = "data/output/template_database/tiktok_vids/tiktok_vids/7596628926374448439_template.json"
    IMAGE_PROMPT_OUTPUT = "video_gen_pipeline/image_prompt.txt"
    VIDEO_PROMPT_OUTPUT = "video_gen_pipeline/video_prompt.txt"
    
    # Load template
    print(f"Loading template from {TEMPLATE_JSON_PATH}...")
    with open(TEMPLATE_JSON_PATH, "r") as f:
        template = json.load(f)
    
    # Load transcript
    print("Loading corresponding transcript...")
    transcript = load_transcript(template)
    if transcript:
        print(f"Transcript loaded ({len(transcript)} characters)")
    
    # Compile base prompt
    print("Compiling template to base prompt...")
    compiler = TemplatePromptCompiler()
    base_prompt = compiler.compile(template)
    print(f"\nBase prompt:\n{base_prompt}\n")
    
    # Refine for image generation
    print("Refining image prompt with Claude...")
    image_prompt = refine_with_claude(base_prompt, "image", transcript)
    print(f"\nRefined image prompt:\n{image_prompt}\n")
    
    # Refine for video generation
    print("Refining video prompt with Claude...")
    video_prompt = refine_with_claude(base_prompt, "video", transcript)
    print(f"\nRefined video prompt:\n{video_prompt}\n")
    
    # Save to files
    with open(IMAGE_PROMPT_OUTPUT, "w") as f:
        f.write(image_prompt)
    print(f"Image prompt saved to {IMAGE_PROMPT_OUTPUT}")
    
    with open(VIDEO_PROMPT_OUTPUT, "w") as f:
        f.write(video_prompt)
    print(f"Video prompt saved to {VIDEO_PROMPT_OUTPUT}")
    
    print("\nPrompt generation completed successfully!")

if __name__ == "__main__":
    main()