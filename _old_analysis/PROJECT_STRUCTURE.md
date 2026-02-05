┌─────────────────────────────────────────────────────────────────┐
│                      VIDEO ANALYSIS PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

INPUT: Video Files (any source)
   │
   ├─── best_media/
   ├─── tiktok_vids/
   └─── my_custom_videos/
   │
   ▼
┌──────────────────────┐
│  Module 01           │  📋 01_classification.py
│  CLASSIFICATION      │  
│                      │  • Analyze video metadata
│  Video → Niche       │  • Apply niche taxonomy
│                      │  • Output: classifications.csv
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│  Module 02           │  🎬 02_template_extraction.py
│  EXTRACTION          │
│                      │  • Scene detection
│  Video → Template    │  • OCR text overlays
│                      │  • Transcribe audio (Whisper)
│                      │  • Analyze visual features
│                      │  • Output: template JSONs
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│  Module 03           │  📊 03_template_analysis.py
│  ANALYSIS            │
│                      │  • Aggregate by niche
│  Templates →         │  • Identify clusters
│  Insights            │  • Generate visualizations
│                      │  • Create recommendations
│                      │  • Output: insights + charts
└──────────────────────┘
   │
   ▼
OUTPUT: Actionable Insights
   │
   ├─── video_classifications.csv
   ├─── templates/<video_id>.json
   ├─── template_insights.json
   └─── niche_comparison.png
```

---

## Data Flow: TikTok to Insights

```
1. SCRAPE (tiktok/)
   ┌─────────────────┐
   │ Apify API       │ → tiktok_data_1.json
   │ apify_scraper   │
   └─────────────────┘
          ↓
   ┌─────────────────┐
   │ yt-dlp          │ → tiktok_vids/<uploader>/<id>.mp4
   │ tiktok_download │
   └─────────────────┘

2. ANALYZE (analysis_pipeline/)
   ┌─────────────────┐
   │ Classification  │ → video_classifications.csv
   └─────────────────┘
          ↓
   ┌─────────────────┐
   │ Extraction      │ → templates/<id>_template.json
   └─────────────────┘
          ↓
   ┌─────────────────┐
   │ Analysis        │ → insights.json + visualizations
   └─────────────────┘
```

---

## Usage Examples

### Run Full Pipeline
```bash
# Simple (uses defaults)
./run_pipeline.sh

# Or directly
python analysis_pipeline/pipeline_driver.py --video-dir best_media
```

### Custom Dataset
```bash
python analysis_pipeline/pipeline_driver.py \
  --video-dir my_videos \
  --features-dir my_features \
  --output-dir results
```

### Skip Steps (Rerun Only Analysis)
```bash
python analysis_pipeline/pipeline_driver.py \
  --video-dir best_media \
  --skip-classification \
  --skip-extraction
```

### TikTok Workflow
```bash
# 1. Scrape
cd tiktok && python apify_scraper.py

# 2. Download
python tiktok_downloader.py

# 3. Analyze
cd .. && ./run_pipeline.sh --video-dir tiktok_vids
```

---

## Module Details

| Module | Purpose | Input | Output | Can Run Standalone? |
|--------|---------|-------|--------|-------------------|
| **01_classification** | Classify videos by niche | Videos + Features | classifications.csv | ✅ Yes |
| **02_extraction** | Extract template profiles | Videos + Classifications | template JSONs | ✅ Yes |
| **03_analysis** | Analyze patterns | Template JSONs | Insights + Charts | ✅ Yes |
| **pipeline_driver** | Orchestrate all modules | Videos | All outputs | ✅ Yes (recommended) |

---

## Migration Checklist

- [x] Create new `analysis_pipeline/` folder (underscore, no space)
- [x] Create modular pipeline (01, 02, 03 modules)
- [x] Create unified driver (pipeline_driver.py)
- [x] Move analysis tools from `tiktok/` to `analysis_pipeline/`
- [x] Keep only scraping tools in `tiktok/`
- [x] Create comprehensive documentation
- [x] Create quick launcher script (run_pipeline.sh)
- [ ] Test pipeline on best_media dataset
- [ ] Remove old `analysis pipeline/` folder (optional)

---

## Next Actions

1. **Test the new pipeline:**
   ```bash
   ./run_pipeline.sh
   ```

2. **Verify outputs** in `data/output/`

3. **Update any external scripts** that referenced old file paths

4. **Remove old folder** when ready:
   ```bash
   rm -rf "analysis pipeline/"  # Old folder
   ```

5. **Start using the modular pipeline** for all video analysis tasks!
