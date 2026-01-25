# Video Analysis Pipeline

A modular, end-to-end pipeline for analyzing video content, extracting template patterns, and generating insights by niche.

## Overview

This pipeline processes video datasets to:
1. **Classify** videos by niche/category
2. **Extract** video template profiles (scenes, text overlays, audio, visuals)
3. **Analyze** patterns across niches and generate recommendations

## Pipeline Structure

### Modules (Ordered)

```
analysis_pipeline/
├── 01_classification.py          # Module 1: Video Classification
├── 02_template_extraction.py     # Module 2: Template Extraction
├── 03_template_analysis.py       # Module 3: Template Analysis
├── pipeline_driver.py             # Main pipeline driver
├── niche_taxonomy.py              # Niche classification taxonomy
└── README.md                      # This file
```

### Module 1: Classification (`01_classification.py`)
Classifies videos by niche using content analysis and taxonomy.

**Input:** 
- Video directory (e.g., `best_media/`)
- Features directory with JSON metadata (e.g., `features/`)

**Output:**
- `video_classifications.csv` - Video IDs mapped to niches

**Run standalone:**
```bash
python 01_classification.py
```

### Module 2: Template Extraction (`02_template_extraction.py`)
Extracts comprehensive video template profiles including:
- Scene detection and pacing analysis
- Text overlay detection (OCR)
- Audio transcription (Whisper)
- Visual feature analysis (color, brightness)

**Input:**
- Video directory
- Classifications from Module 1

**Output:**
- `templates/<video_id>_template.json` - Individual template profiles

**Run standalone:**
```bash
python 02_template_extraction.py
```

### Module 3: Template Analysis (`03_template_analysis.py`)
Analyzes template patterns to:
- Generate niche-level statistics
- Identify template clusters
- Create visualizations
- Generate recommendations

**Input:**
- Template profiles from Module 2

**Output:**
- `template_insights.json` - Aggregated insights by niche
- `niche_comparison.png` - Visualization of patterns
- Console output with recommendations

**Run standalone:**
```bash
python 03_template_analysis.py
```

## Usage

### Quick Start: Run Full Pipeline

The simplest way to run the complete pipeline:

```bash
python pipeline_driver.py --video-dir best_media
```

This will:
1. Classify all videos in `best_media/`
2. Extract template profiles for each video
3. Analyze patterns and generate insights
4. Save all results to `data/output/`

### Custom Configuration

Specify custom directories:

```bash
python pipeline_driver.py \
  --video-dir my_videos \
  --features-dir my_features \
  --output-dir results
```

### Skip Steps

If you've already run some steps:

```bash
# Skip classification (if already done)
python pipeline_driver.py --video-dir best_media --skip-classification

# Run only analysis
python pipeline_driver.py --video-dir best_media --skip-classification --skip-extraction
```

### Run Individual Modules

Each module can be run independently for testing or debugging:

```python
# Classification only
from classification import VideoClassifier
classifier = VideoClassifier()
classifier.classify_videos(video_dir=Path("best_media"), 
                          features_dir=Path("features"),
                          output_csv=Path("data/output/classifications.csv"))

# Extraction only
from template_extraction import TemplateExtractor
extractor = TemplateExtractor()
extractor.extract_templates_batch(video_dir=Path("best_media"),
                                  classifications_csv=Path("data/output/classifications.csv"),
                                  output_dir=Path("data/output/templates"))

# Analysis only
from template_analysis import TemplateAnalyser
analyser = TemplateAnalyser(templates_dir=Path("data/output/templates"))
analyser.load_templates()
analyser.analyze_by_niche()
analyser.generate_recommendation('tech_saas')
```

## Output Files

After running the pipeline, you'll find:

```
data/output/
├── video_classifications.csv      # Video -> Niche mappings
├── templates/                      # Individual template profiles
│   ├── video1_template.json
│   ├── video2_template.json
│   └── ...
├── template_insights.json         # Aggregated insights by niche
└── niche_comparison.png           # Visual comparison charts
```

## Dependencies

Required Python packages:
```
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
easyocr
whisper
scenedetect
```

Install with:
```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-learn easyocr openai-whisper scenedetect
```

## Architecture

The pipeline follows a **modular, sequential architecture**:

```
[Videos] → [Classification] → [Template Extraction] → [Template Analysis] → [Insights]
```

Each module:
- Has a clear, single responsibility
- Can run independently
- Produces well-defined outputs
- Consumes outputs from previous stages

## Extending the Pipeline

### Add a New Niche Category

Edit `niche_taxonomy.py` to add keywords for your niche:

```python
class NicheTaxonomy:
    def __init__(self):
        self.niche_keywords = {
            "your_new_niche": ["keyword1", "keyword2", "keyword3"],
            # ...
        }
```

### Add New Features to Templates

Extend `TemplateExtractor` in `02_template_extraction.py`:

```python
def extract_custom_feature(self, video_path: Path):
    # Your feature extraction logic
    return feature_data

def extract_template(self, video_path: Path, ...):
    template = {
        # ... existing features
        "custom_feature": self.extract_custom_feature(video_path)
    }
    return template
```

### Add New Visualizations

Extend `TemplateAnalyser` in `03_template_analysis.py`:

```python
def plot_custom_analysis(self, save_path: Path):
    # Your visualization logic
    plt.savefig(save_path)
```

## Best Practices

1. **Always run the full pipeline** on new datasets to ensure consistency
2. **Check output files** after each module to validate results
3. **Use skip flags** when iterating on later modules to save time
4. **Monitor memory usage** for large video collections (extraction is memory-intensive)
5. **Store templates separately** from raw videos to enable re-analysis without re-extraction

## Troubleshooting

### Module import errors
Make sure you're running from the workspace root or the analysis_pipeline directory.

### Missing video classifications
Run Module 1 first or remove `--skip-classification` flag.

### No templates found
Ensure Module 2 completed successfully. Check `data/output/templates/` directory.

### CUDA/GPU errors with Whisper
Set environment variable: `export WHISPER_DEVICE=cpu`

### OCR not detecting text
Try adjusting confidence threshold in `detect_text_overlays()` method.

## Contributing

When adding features:
1. Keep modules focused and independent
2. Use consistent naming conventions
3. Add error handling and logging
4. Update this README with new capabilities
5. Provide examples in docstrings

## License

Internal use only - Hameo Spotlight Project
