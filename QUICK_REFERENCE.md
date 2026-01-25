# Quick Reference - Video Analysis Pipeline

## 🚀 Quick Start

```bash
# Easiest way - run everything
./run_pipeline.sh

# Or use the driver directly
cd analysis_pipeline
python pipeline_driver.py --video-dir ../best_media
```

## 📁 Folder Structure

```
analysis_pipeline/  → All analysis code (USE THIS)
tiktok/             → TikTok scraping only
best_media/         → Video files
data/output/        → Results appear here
```

## 🎯 Main Entry Points

| Script | Purpose | Location |
|--------|---------|----------|
| `run_pipeline.sh` | Quick launcher | Root folder |
| `pipeline_driver.py` | Main driver | analysis_pipeline/ |
| `apify_scraper.py` | Scrape TikTok | tiktok/ |
| `tiktok_downloader.py` | Download videos | tiktok/ |

## 📋 Pipeline Steps

```
01_classification.py      → Classify videos by niche
02_template_extraction.py → Extract video features
03_template_analysis.py   → Analyze & visualize
```

## 💡 Common Commands

### Run full pipeline
```bash
./run_pipeline.sh
```

### Run on custom folder
```bash
python analysis_pipeline/pipeline_driver.py --video-dir my_videos
```

### Skip already-done steps
```bash
python analysis_pipeline/pipeline_driver.py \
  --video-dir best_media \
  --skip-classification
```

### Run single module
```bash
cd analysis_pipeline
python 01_classification.py     # Just classify
python 02_template_extraction.py # Just extract
python 03_template_analysis.py   # Just analyze
```

### TikTok scraping
```bash
cd tiktok
python apify_scraper.py        # Scrape metadata
python tiktok_downloader.py    # Download videos
```

## 📊 Output Files

```
data/output/
├── video_classifications.csv  → Video → Niche mappings
├── templates/                  → Individual video profiles
│   └── <video_id>_template.json
├── template_insights.json     → Aggregated insights
└── niche_comparison.png       → Visualization
```

## 🔍 Find Documentation

- **Pipeline:** `analysis_pipeline/README.md`
- **Scraping:** `tiktok/README.md`
- **Summary:** `REORGANIZATION_SUMMARY.md`
- **Structure:** `PROJECT_STRUCTURE.md`

## ⚙️ Python Package Usage

```python
# Import modules
from analysis_pipeline.classification import VideoClassifier
from analysis_pipeline.template_extraction import TemplateExtractor
from analysis_pipeline.template_analysis import TemplateAnalyser

# Use in your code
classifier = VideoClassifier()
classifier.classify_videos(video_dir=..., features_dir=..., output_csv=...)
```

## 🛠️ Troubleshooting

**Import errors:**
```bash
cd analysis_pipeline
python pipeline_driver.py --video-dir ../best_media
```

**No templates found:**
- Make sure classification ran first
- Check `data/output/video_classifications.csv` exists

**Module not found:**
- Run from correct directory
- Check Python path includes analysis_pipeline/

## 📝 Quick Tips

✅ Use `./run_pipeline.sh` for simplest experience
✅ Modules are numbered - run in order
✅ Each module can run independently for testing
✅ Pipeline works on ANY video dataset, not just TikTok
✅ Use `--skip-*` flags to rerun only what you need

## 🎓 Learn More

Run with `--help` for all options:
```bash
python analysis_pipeline/pipeline_driver.py --help
```

Read full docs:
```bash
cat analysis_pipeline/README.md
```
