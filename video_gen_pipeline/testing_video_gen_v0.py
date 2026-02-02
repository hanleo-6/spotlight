import json
import time
from typing import Dict, Any, Optional
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv
import os


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
        return (
            f"Create a {t['personality']} {t['template_name'].lower()} "
            f"video in the {t['category']} category using a "
            f"{t['story_structure']['storytelling_type']} structure."
        )

    def _compile_physical_constraints(self, t: Dict[str, Any]) -> str:
        tech = t["technical_settings"]
        motion = t["motion_design"]

        return (
            f"The video is shot on a {tech['camera_quality']} camera mounted on a tripod. "
            f"The camera remains {motion['camera_behavior']}. "
            f"Lighting is {tech['lighting_style']}, with a {tech['depth_of_field']} depth of field "
            f"and a slightly {tech['color_grading']} tone caused by ambient lighting."
        )

    def _compile_human_behavior(self, t: Dict[str, Any]) -> str:
        pacing = t["story_structure"]["pacing"]
        return (
            f"The speaker delivers lines at a {pacing} pace with natural pauses, "
            f"minor posture shifts, and slightly imperfect eye contact."
        )

    def _compile_story_structure(self, t: Dict[str, Any]) -> str:
        s = t["story_structure"]
        flow = " ".join(s["content_flow"])
        return (
            f"Begin with {s['intro']} Follow with {s['hook']} "
            f"The narrative progresses as follows: {flow} "
            f"The sequence concludes with {s['outro']}"
        )

    def _compile_visual_layout(self, t: Dict[str, Any]) -> str:
        layout = t["visual_design"]["layout"]
        layers = ", ".join(layout["layer_hierarchy"])

        return (
            f"The subject is positioned {layout['subject_positioning']} "
            f"using the {layout['grid_system']}. "
            f"The visual layers appear in this order: {layers}."
        )

    def _compile_motion(self, t: Dict[str, Any]) -> str:
        motion = t["motion_design"]
        transitions = ", ".join(motion["transitions"])

        return (
            f"Scenes are connected using {transitions}. "
            f"Any movement is {motion['animation_speed']}, "
            f"with text entering and exiting via {motion['text_entry']}."
        )

    def _compile_typography(self, t: Dict[str, Any]) -> str:
        typo = t["visual_design"]["typography"]

        return (
            f"Text overlays use {typo['headline_style']} for headlines and "
            f"{typo['body_style']} for body text, aligned {typo['alignment']}. "
            f"Text appears in {typo['uppercase_pattern']} and fades without motion."
        )

    def _compile_color_and_lighting(self, t: Dict[str, Any]) -> str:
        colors = t["visual_design"]["color_system"]

        return (
            f"The primary accent color is {colors['primary']['appearance']}, "
            f"used sparingly against a {colors['secondary']['appearance']} background. "
            f"The background treatment is {colors['background_treatment']} with "
            f"{colors['contrast_strategy']} contrast."
        )

    def _compile_format_mix(self, t: Dict[str, Any]) -> str:
        fmt = t["content_format"]

        return (
            f"The video uses a mixed format consisting of {fmt['format_mix_ratio']}. "
            f"Information density is {fmt['information_density']} with a "
            f"{fmt['text_visual_balance']} balance between text and visuals."
        )

    def _compile_outro(self, t: Dict[str, Any]) -> str:
        return "End with a simple logo and tagline on a clean background."


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)


# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE VALUES FOR YOUR USE CASE
# ============================================================================

# Generate reference image (Step 1)
GENERATE_IMAGE = False  # Set to True to generate a new reference image
IMAGE_PROMPT = (
    "A close-up, slightly angled shot of a laptop screen against a bright blue sky with wispy clouds. "
    "The screen displays a dark-themed website with the \"WebDash\" logo and \"Product\" and \"Resources\" menu items at the top. "
    "A person's index finger is touching the screen near the logo. "
    "The browser tab says \"WebDash - Get a website in seconds\"."
)
IMAGE_OUTPUT_PATH = "data/output/generated_videos/generated_image.png"

# Generate video from reference image (Step 2)
GENERATE_VIDEO = True  # Set to False to skip video generation
VIDEO_PROMPT = (
    "The camera slowly zooms into the screen as the finger taps on the \"WebDash\" text. "
    "The webpage then scrolls down smoothly to reveal a headline about generating ecommerce stores, "
    "with the word \"THIS\" appearing in bold yellow text over the center of the frame."
)
VIDEO_OUTPUT_PATH = "data/output/generated_videos/test_video3.mp4"
VIDEO_ASPECT_RATIO = "9:16"
VIDEO_DURATION_SECONDS = 4

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


def generate_reference_image(prompt: str, output_path: str) -> None:
    """Generate the initial reference image using Gemini."""
    print(f"Generating reference image from prompt...")
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[prompt],
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio="9:16",
            ),
        ),
    )

    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            image.save(output_path)
            print(f"Reference image saved to {output_path}")


def generate_video_from_image(
    video_prompt: str,
    image_path: str,
    output_path: str,
    aspect_ratio: str = "9:16",
    duration_seconds: int = 4,
) -> None:
    """Generate video from reference image using Veo-3.1."""
    print(f"Loading reference image from {image_path}...")
    image = types.Image.from_file(location=image_path)

    print(f"Starting video generation...")
    try:
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=video_prompt,
            image=image,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
            ),
        )
    except errors.ClientError as e:
        print("=== ClientError from Veo ===")
        print("Status code:", e)
        print("Response JSON:", e.response_json)
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
    print(f"Generated video saved to {output_path}")


def main():
    """Main pipeline to generate videos from templates."""
    if GENERATE_IMAGE:
        generate_reference_image(IMAGE_PROMPT, IMAGE_OUTPUT_PATH)

    if GENERATE_VIDEO:
        generate_video_from_image(
            video_prompt=VIDEO_PROMPT,
            image_path=IMAGE_OUTPUT_PATH,
            output_path=VIDEO_OUTPUT_PATH,
            aspect_ratio=VIDEO_ASPECT_RATIO,
            duration_seconds=VIDEO_DURATION_SECONDS,
        )

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
