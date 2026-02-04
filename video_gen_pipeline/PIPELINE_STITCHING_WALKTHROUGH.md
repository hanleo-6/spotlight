# Stitching Pipeline Walkthrough

This document explains the end-to-end logic in [video_gen_pipeline/video_gen_pipeline_stitching.py](video_gen_pipeline_stitching.py), focusing on the timestamped-transcript workflow.

## 1) Configuration and inputs
- **Prompt inputs**
  - Image prompt: [video_gen_pipeline/prompts/image_prompt_stitching.txt](prompts/image_prompt_stitching.txt)
  - Video prompt: [video_gen_pipeline/prompts/video_prompt_stitching.txt](prompts/video_prompt_stitching.txt)
  - Timestamped transcript: [video_gen_pipeline/prompts/timestamped_transcript.txt](prompts/timestamped_transcript.txt)
  - Stitching guidance: [video_gen_pipeline/prompts/stitching_prompt.txt](prompts/stitching_prompt.txt)
- **Output root**
  - All outputs go under data/output/generated_videos/<run_index>/
- **Key toggles**
  - `GENERATE_IMAGE`, `GENERATE_VIDEO`, `STITCH_SEGMENTS`
  - `EXTEND_SEGMENTS`, `USE_PREVIOUS_VIDEO_CONTEXT`, `USE_TRANSCRIPT_TIMESTAMPS`
  - `MAX_SEGMENTS_TO_GENERATE` for testing

## 2) Run folder selection
- `_get_next_run_index()` chooses the next run directory in data/output/generated_videos.
- If the last run has no .complete marker, it reuses that folder to resume.

## 3) Read prompts
- The image and video prompt files are read into memory.
- Stitching guidance is loaded and split into two blocks:
  - WITHIN_SEGMENT (applied to each segment prompt)
  - BETWEEN_SEGMENTS (used for continuity between adjacent segments)

## 4) Load timestamped transcript
- `_load_timestamped_transcript()` parses [video_gen_pipeline/prompts/timestamped_transcript.txt](prompts/timestamped_transcript.txt):
  - Extracts each [Segment N: ...] block
  - Captures time ranges and the transcript text
  - Produces a dictionary keyed by segment number
- If `MAX_SEGMENTS_TO_GENERATE` is set, the transcript map is truncated to the first N segments.

## 5) Optional TTS generation (per segment)
- If audio generation is enabled and `USE_PREVIOUS_VIDEO_CONTEXT` is false:
  - `tts_generator.generate_segment_audio()` creates one audio file per transcript segment
  - These are stored under the run’s audio/ folder

## 6) Reference image generation
- If `GENERATE_IMAGE` is true:
  - `generate_reference_image()` calls Gemini image preview
  - Saves a single reference image to generated_image.png

## 7) Optional single-video generation
- If `GENERATE_VIDEO` is true:
  - `generate_video_from_image()` creates a single, non-segmented video
  - This is separate from the stitched pipeline and uses the full video prompt

## 8) Segment prompt preparation
- If `STITCH_SEGMENTS` is true:
  - `_split_segment_prompts()` splits the video prompt into per-segment prompts using ---SEGMENT_BREAK---
  - If `MAX_SEGMENTS_TO_GENERATE` is set, the prompt list is truncated to the first N segments

## 9) Resume logic for segments
- The pipeline scans the run’s segments/ folder for existing segment files.
- If it finds the last completed segment, it resumes from the next segment.

## 10) Per-segment generation loop
For each segment in order:
1. **Transcript lookup** (timestamped only)
   - If `USE_TRANSCRIPT_TIMESTAMPS` is true, the segment’s transcript text is pulled from the parsed map.
2. **Prompt composition**
   - First segment (or when extension is off):
     - `_compose_segment_prompt()` adds WITHIN_SEGMENT stitching guidance.
   - Subsequent segments (when extension is on):
     - `_remove_overlap()` optionally removes duplicate sentences.
     - `_compose_extension_prompt()` adds BETWEEN_SEGMENTS guidance and the segment transcript.
3. **Reference inputs**
   - If extending, the last frame from the previous segment is extracted and used as a reference image.
   - If `USE_PREVIOUS_VIDEO_CONTEXT` is true, the previous video is also passed into Veo.
4. **Generate segment video**
   - `generate_video_from_image()` calls Veo 3.1 with the segment prompt.
5. **Optional audio overlay**
   - If audio was generated, it is overlaid onto the segment video.

## 11) Transition prompt generation
- For logging/debugging, `_compose_between_segments_prompt()` builds a short transition prompt between each adjacent pair of segments.

## 12) Stitch final video
- `_stitch_videos()` uses ffmpeg concat to join all segment mp4 files.
- If audio overlays exist, those versions are stitched; otherwise the raw segments are stitched.

## 13) Manifest and completion marker
- A stitch_manifest.json is written to the segments/ folder with:
  - Segment outputs
  - Audio outputs (if any)
  - Transition prompts
  - Final stitched output path
  - Success or failure status
- A .complete marker is written in the run directory when the pipeline finishes.

## 14) Where transcript data is used
- Only timestamped transcript segments are used.
- Transcript text is injected into segment prompts via `_compose_extension_prompt()` when segments extend each other.
- No full-video transcript is passed into the model.
