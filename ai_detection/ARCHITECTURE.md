# AI Detection — Inner Detector Architecture

> **Where this fits**: The engine orchestrator (`lib/analyzer/orchestrator.py`)
> runs all plugins in two tiers.  The `ai_detection` plugin
> (`plugins/ai_ml/ai_detection.py`) is one AI/ML-tier plugin.  Internally
> it delegates to the detector classes described here.
>
> For the full engine pipeline, see [docs/ENGINE_ARCHITECTURE.md](../docs/ENGINE_ARCHITECTURE.md).

---

## Overview

The `ai_detection` module contains specialised ML detectors that look
for signs of AI-generated images.  The plugin instantiates a
`MultiLayerDetector` (the inner orchestrator) which runs detectors in
order and can stop early when confidence is high enough.

```
plugins/ai_ml/ai_detection.py          ← plugin entry point
  └─ ai_detection/detectors/orchestrator.py
       ├─ MetadataDetector   (order=10, fast, deterministic)
       ├─ SDXLDetector       (order=60, HuggingFace Swin Transformer)
       └─ SPAIDetector       (order=100, spectral analysis ML)
```

## Detectors

### MetadataDetector
- Checks EXIF/XMP for AI generator tags (Midjourney, DALL-E, Stable Diffusion, etc.)
- Checks for C2PA content credentials
- Analyses software signatures and dimensions
- Fast and deterministic — 100% accurate when AI tags are present

### SDXLDetector
- Swin Transformer model: `Organika/sdxl-detector`
- Trained on Wikimedia-SDXL image pairs
- 98.1% published accuracy
- Runs as a subprocess to isolate model dependencies

### SPAIDetector
- SPAI (Spectral AI-Generated Image Detector) — CVPR 2025
- Frequency domain analysis
- Works on any image resolution
- Runs as a subprocess

## Data Flow

Each detector returns a result dict with `confidence`, `score`, and
`detected_types`.  The `MultiLayerDetector` collects these as
`detection_layers` and writes them under `results["ai_detection"]`:

```python
results["ai_detection"] = {
    "enabled": True,
    "detection_layers": [
        {"detector": "MetadataDetector", "confidence": 95, "score": 90, ...},
        {"detector": "SDXLDetector", "confidence": 88, "score": 82, ...},
    ],
    "methods_run": ["MetadataDetector", "SDXLDetector"]
}
```

The plugin does **not** produce scores or verdicts.  The engine's
compliance auditor (`lib/analyzer/auditor.py`) reads these raw layers
and factors them into the overall authenticity score.

## Early Stopping

The inner orchestrator stops running further detectors when a result
comes back with `CERTAIN` confidence (≥95%).  This is a performance
optimisation only — it doesn't affect correctness because the engine
auditor always runs on whatever data is available.

## Adding a New Detector

1. Create a class in `ai_detection/detectors/` inheriting from `BaseDetector`.
2. Implement `detect(image_path, context=None)` returning a result dict.
3. Register it in `MultiLayerDetector` at the appropriate order.

See `ai_detection/detectors/base.py` for the interface.
