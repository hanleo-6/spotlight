# Pipeline Optimization Recommendations

## Current Architecture Analysis

The pipeline consists of 3 sequential stages:
1. **Classification** - Text-based niche classification (CPU-bound)
2. **Template Extraction** - Video processing: scene detection, OCR, transcription, visual analysis (I/O + CPU intensive)
3. **Template Analysis** - Aggregation and clustering (CPU-bound)

### Current Bottlenecks

1. **Sequential Processing**: Each video is processed completely before moving to the next
2. **Expensive Operations Per Video** (Template Extraction):
   - Scene detection (video decoding)
   - OCR on sampled frames (loaded per sample)
   - Whisper transcription (entire audio transcoding)
   - Visual feature extraction (frame sampling)
3. **GPU Contention**: OCR (EasyOCR) and Whisper both compete for GPU resources
4. **Model Loading**: `easyocr.Reader` and `whisper_model` loaded per TemplateExtractor instance
5. **No Caching**: Results not cached between runs

---

## Optimization Strategies

### 1. **Parallel Batch Processing (Highest Impact)**

**Current**: Sequential video processing  
**Improved**: Process multiple videos concurrently using multiprocessing/concurrent.futures

```python
# Benefits:
# - Use all CPU cores for classification
# - Mask I/O waits during template extraction
# - Estimated speedup: 3-8x (depends on core count and I/O pattern)
```

**Implementation Details**:
- Use `concurrent.futures.ThreadPoolExecutor` for I/O-heavy classification
- Use `multiprocessing.Pool` for CPU-heavy template extraction
- Set worker count = CPU cores (4-8 typical)
- Chunk videos into batches to reduce memory overhead

### 2. **Optimize Template Extraction (High Impact)**

**Current Issues**:
- Entire audio transcribed even if only partial needed
- All frames sampled for visual analysis
- Scene detection reads full video twice (VideoManager + cv2.VideoCapture)
- OCR run on all sampled frames sequentially

**Improvements**:
```python
# a) Reduce frame sampling
- Current: Linspace across total frames
- Better: Sample first 10s, middle 10s, last 10s (most representative)
- Reduction: ~70% fewer frames

# b) Batch OCR processing
- Current: Sequential frame OCR
- Better: Queue frames and run batched inference
- Reduction: ~40% faster OCR (batch GPU inference)

# c) Reuse video handle
- Current: Open video 2-3 times (scene detection, text, visual)
- Better: Single pass through video extracting all features
- Reduction: ~50% I/O time

# d) Conditional transcription
- Current: Always transcribe full audio
- Better: Detect if video has significant audio first
- Reduction: Skip transcription for muted/music-only videos
```

### 3. **Model Loading Optimization (Medium Impact)**

**Current**: Models loaded per TemplateExtractor instance

**Better Approach**:
```python
# Use module-level singletons
_EASYOCR_READER = None
_WHISPER_MODEL = None

def get_ocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(['en'])
    return _EASYOCR_READER

# Share across worker processes via multiprocessing.Manager or
# Load once per worker in process pool
```

**Reduction**: Eliminate model loading overhead (2-5 seconds per worker)

### 4. **GPU Scheduling & Resource Management (Medium Impact)**

**Current**: No coordination between GPU-using operations

**Better**:
- Run OCR and Whisper serially on GPU (not concurrently)
- Use CPU for scene detection while GPU processes OCR/Whisper
- Consider quantized/lighter models (e.g., Whisper `tiny` or `base` instead of `small`)

### 5. **Add Caching Layer (Medium Impact)**

**Current**: No results cached

**Better**:
```python
# Cache extracted templates to avoid re-processing
# Cache classifications if features haven't changed
# Use file modification time to invalidate cache

Estimated recovery: 10-100% depending on re-run frequency
```

### 6. **Async Pipeline Orchestration (Low-Medium Impact)**

**Current**: Strict serial stages (Classification → Extraction → Analysis)

**Better**: Start analysis on completed templates while extraction still running
```python
# Timeline with concurrent processing:
# ├─ Stage 1 [████████]
# ├─ Stage 2 [████████████████] (Process batches in parallel)
# ├─ Stage 3 [    ████████] (Start once first batch done)
# └─ Result   [═══════════════]
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours) → **3-5x speedup**
1. Add batch processing to Classification (ThreadPoolExecutor)
2. Optimize Template Extraction: single video pass + batch OCR
3. Add module-level model singletons

### Phase 2: Medium Effort (2-4 hours) → **Additional 1.5-2x speedup**
1. Multiprocessing for Template Extraction
2. GPU/CPU scheduling for OCR + Whisper
3. Smart frame sampling strategy

### Phase 3: Polish (1-2 hours) → **Additional 1.2-1.5x speedup**
1. Caching layer
2. Async pipeline orchestration
3. Results preprocessing/vectorization for Analysis stage

---

## Expected Results

| Configuration | Estimated Speed | Relative Gain |
|---|---|---|
| Current (Sequential) | 1x | Baseline |
| Phase 1 (Batching + Optimization) | 3-5x | 3-5x faster |
| Phase 1 + 2 (Parallel + GPU Scheduling) | 5-10x | 5-10x faster |
| All Phases (Full Optimization) | 6-15x | 6-15x faster |

**Assumptions**: 
- 4-8 CPU cores available
- 50 videos (typical batch)
- Each video: ~30-60 seconds processing time

---

## Specific Code Changes Required

### Classification Parallelization
- Add `ThreadPoolExecutor` with worker pool
- Process videos in parallel (I/O-bound)

### Template Extraction Improvements
- Refactor to single-pass video processing
- Batch OCR frame processing
- Create worker pool with shared GPU/model resources
- Add early-exit for videos without audio

### Model Management
- Create `models.py` with singleton initialization
- Ensure thread/process safety for model sharing

### Caching System
- Add JSON cache for extracted templates
- Implement cache invalidation logic

---

## Recommended Next Step

Start with Phase 1 implementation focusing on:
1. **Template Extraction Optimization** (single-pass + batch OCR) - highest ROI
2. **Classification Parallelization** - quick and easy
3. **Model Singletons** - immediate improvement with minimal changes

This should deliver 3-5x speedup with minimal code changes and risk.
