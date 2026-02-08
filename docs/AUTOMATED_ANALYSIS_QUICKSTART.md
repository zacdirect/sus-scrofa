# Automated Analysis - Quick Reference

## What It Does

Combines AI Detection and OpenCV Manipulation Detection into one unified tab showing all automatic forensics analysis.

## Files to Know

```
templates/analyses/report/
├── show.html                      # Main report (modified: added unified tab)
└── _automated_analysis.html       # NEW: Unified template (384 lines)

docs/
├── AUTOMATED_ANALYSIS.md          # Full documentation (685 lines)
└── AUTOMATED_ANALYSIS_IMPLEMENTATION.md  # Implementation details (398 lines)

plugins/analyzer/
├── ai_detection.py                # AI Detection plugin (order=60)
└── opencv_manipulation.py         # Manipulation plugin (order=65)
```

## How to Use

### Start Services

```bash
# OpenCV service (if not running)
make opencv-start

# Check status
make opencv-status
curl http://localhost:8080/health
```

### View Results

1. Upload image to Ghiro
2. Wait for analysis
3. Click analyzed image
4. Go to **"Automated Analysis"** tab
5. Review combined results

## What You'll See

### 1. Summary Card
- AI Detection status (AI Generated / Authentic)
- Manipulation status (Suspicious / No Issues)
- Confidence percentages

### 2. AI Detection Section
- **Detection Layers Table**: Shows which method detected AI
  - Metadata (EXIF/XMP markers)
  - Filename (pattern matching)
  - ML Model (SPAI deep learning)
- **AI Probability Bar**: 0-100% visual indicator
- **Evidence**: Detailed findings

### 3. Manipulation Section
- **Overall Confidence Bar**: Combined score
- **Three Method Cards**:
  - Gaussian Blur (cloning detection)
  - Noise Consistency (splicing detection)
  - JPEG Artifacts (compression analysis)
- **Technical Metrics**: Per-method details

## Detection Methods

### AI Detection (3 Layers)

| Method | Speed | Accuracy | Triggers On |
|--------|-------|----------|-------------|
| Metadata | <10ms | High | EXIF/XMP markers (Midjourney, DALL-E, etc.) |
| Filename | <1ms | Medium | Patterns like "gemini_generated", "ai_generated" |
| SPAI Model | 2-60s | Very High | Deep learning analysis (fallback) |

### Manipulation Detection (3 Methods)

| Method | Detects | Metrics |
|--------|---------|---------|
| Gaussian Blur | Cloned regions | Anomaly count, coverage % |
| Noise Analysis | Spliced regions | Quadrant variances, consistency |
| JPEG Artifacts | Compression inconsistencies | DCT variation |

## Common Scenarios

### Scenario 1: AI with Metadata
```
✓ Metadata detector finds "Midjourney" → HIGH confidence → STOP
Result: "AI Generated" (metadata layer only)
Time: <10ms
```

### Scenario 2: AI without Metadata
```
✓ Metadata: Nothing found
✓ Filename: Nothing found
✓ SPAI Model: 93.2% AI → CERTAIN confidence
Result: "AI Generated" (ML layer)
Time: 2-60s depending on GPU
```

### Scenario 3: Manipulated Photo
```
✓ AI Detection: Shows "Authentic"
✓ Gaussian Blur: 65% confidence (cloning detected)
✓ Noise Analysis: 72% confidence (inconsistent)
✓ JPEG: 58% confidence (artifacts)
Result: Overall 65% suspicious
```

### Scenario 4: Authentic Photo
```
✓ AI Detection: Metadata shows "Canon EOS"
✓ All manipulation methods: Consistent
Result: Clean/Authentic with high confidence
```

## Interpreting Results

### AI Detection Labels

| Label | Meaning | Action |
|-------|---------|--------|
| AI Generated (HIGH/CERTAIN) | Strong evidence | Likely AI |
| Authentic (HIGH) | No AI markers | Likely real |
| Unknown (MEDIUM/LOW) | Borderline | Manual review |

### Manipulation Labels

| Label | Meaning | Action |
|-------|---------|--------|
| Suspicious (>50%) | Anomalies detected | Investigate |
| No Issues (<50%) | Consistent | Likely authentic |

## Troubleshooting

### "Analysis Not Available"

**AI Detection:**
```bash
make ai-setup  # Install dependencies
ls -lh ai_detection/models/spai_model.pth  # Check weights (892MB)
```

**Manipulation Detection:**
```bash
make opencv-status  # Check container
make opencv-restart  # Restart if needed
make opencv-logs    # View errors
```

### Service Not Running

```bash
# Check if container exists
podman ps -a | grep opencv

# Start if stopped
make opencv-start

# Rebuild if corrupted
make opencv-stop
make opencv-build
make opencv-start
```

### High CPU Usage

**AI Detection:**
- GPU recommended (10-100x faster)
- CPU mode: 30-60s per image
- Consider GPU upgrade for production

**Manipulation Detection:**
- Normal: 1-3s per image
- High load: Run multiple containers
- Scale with podman/docker swarm

## Performance Tips

1. **Enable GPU** for AI detection (massive speedup)
2. **Run OpenCV container** on separate host for scaling
3. **Cache results** to avoid re-analysis
4. **Batch processing** for multiple images

## Data Structure

### AI Detection Output
```python
analysis.report.ai_detection = {
    "enabled": True,
    "likely_ai": True/False,
    "ai_probability": 0-100,
    "confidence": "certain"/"high"/"medium"/"low",
    "detection_layers": [
        {"method": "...", "verdict": "...", "confidence": "..."}
    ],
    "evidence": "..."
}
```

### Manipulation Output
```python
analysis.report.opencv_manipulation = {
    "enabled": True,
    "is_suspicious": True/False,
    "overall_confidence": 0-100,
    "manipulation_detection": {...},
    "noise_analysis": {...},
    "jpeg_artifacts": {...}
}
```

## Make Targets

```bash
# OpenCV Service
make opencv-build    # Build container
make opencv-start    # Start service
make opencv-stop     # Stop service
make opencv-restart  # Restart service
make opencv-status   # Check status
make opencv-logs     # View logs

# AI Detection
make ai-setup        # Install dependencies
```

## Port Reference

- **8080**: OpenCV service (HTTP REST API)
  - GET /health: Health check
  - POST /analyze: Image analysis

## Further Reading

- `docs/AUTOMATED_ANALYSIS.md` - Complete documentation
- `docs/AUTOMATED_ANALYSIS_IMPLEMENTATION.md` - Technical details
- `MODERNIZATION.md` - Django 4.2 upgrade notes

## Support

1. Check service logs
2. Verify setup (health endpoints)
3. Review documentation
4. Check GitHub issues
