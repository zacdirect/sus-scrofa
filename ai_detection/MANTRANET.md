# ManTraNet Forgery Localization

**Status**: ✅ Fully integrated and working

ManTraNet (Manipulation Tracing Network) is a deep learning model that detects and localizes image manipulations at the pixel level. Unlike binary classifiers that only determine if an image is manipulated, ManTraNet produces a spatial heatmap showing **where** manipulation occurred.

## Features

- **Pixel-level forgery detection**: Identifies clone tool, splicing, copy-move, and other manipulations
- **Heatmap visualization**: Red/yellow regions indicate manipulated areas
- **Audit integration**: Automatically generates findings for the authenticity scoring system
- **UI integration**: New "Forgery Localization" tab in analysis reports

## Installation

```bash
make mantranet-setup
```

This will:
1. Create model directory at `models/weights/mantranet/`
2. Download ManTraNet architecture code (modelCore.py with custom layers)
3. Install TensorFlow 2.x and Keras into main venv
4. Download the pretrained model (~170MB) to `models/weights/mantranet/ManTraNet_Ptrain4.h5`

**Note**: The setup automatically downloads the model architecture with custom Bayar convolution layers from the official repository.

Verify installation:
```bash
make mantranet-verify
```

## Architecture
 with custom Bayar convolution layers
- **Framework**: Keras with TensorFlow 2.x backend
- **Framework**: TensorFlow 1.x frozen graph, runs via TensorFlow 2.x compatibility
- **Input**: RGB images (any size)
- **Output**: Single-channel manipulation mask (0=pristine, 1=forged)
- **Paper**: [ManTra-Net: Manipulation Tracing Network (CVPR 2019)](https://arxiv.org/abs/1812.08045)

## Audit Findings

ManTraNet generates tiered audit findings based on manipulated area percentage:

| Percentage | Level | Impact | Description |
|------------|-------|--------|-------------|
| >20% | HIGH | -50 points | Extensive manipulation detected |
| 5-20% | MEDIUM | -15 points | Moderate editing in multiple regions |
| 1-5% | LOW | -5 points | Minor manipulation detected |
| <0.5% | MEDIUM (positive) | +15 points | Image appears pristine |

### Base Scoring Model
- Starting score: **50** (uncertain)
- Positive findings **add** points
- Negative findings **subtract** points
- Final score clamped to **0-100**

## Usage

ManTraNet runs automatically during image analysis when enabled. Results appear in:

1. **Automated Analysis tab**: Overall manipulation summary
2. **Forgery Localization tab**: Side-by-side original + heatmap visualization

### Via UI
Upload an image → Process → Navigate to "Forgery Localization" tab

### Via API
Results stored in `analysis.report.mantranet`:
```python
{
    "manipulated_percentage": 8.5,
    "region_count": 3,
    "max_confidence": 0.85,
    "mask_id": "64abc123...",  # GridFS ID
    "inference_time_s": 15.2
}
```

## Files Created

- `ai_detection/detectors/mantranet_detector.py` - Detector plugin
- `ai_detection/mantranet_infer.py` - Inference script (TF 1.14)
- `templates/analyses/report/forgery.html` - UI template
- `analyses/views.py` - Added `show_forgery()` view
- `analyses/urls.py` - Added forgery URL route

## Performance

- **Inference time**: ~5-15s per image (CPU), ~1-3s (GPU)
- **Model size**: ~100MB (frozen graph)
- **Memory**: ~2GB RAM during inference
- **Execution order**: Runs after SDXL/SPAI (order=80)

## Troubleshooting

**Model not found:**
```bash
make mantranet-setup
```

**TensorFlow errors:**
ManTraNet uses a TensorFlow 1.x frozen graph loaded via TF 2.x compatibility. Check:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```
Should be TensorFlow 2.10+

**Mask not displaying:**
- Check that mask was saved to GridFS: `analysis.report.mantranet.mask_id`
- Verify image service is serving GridFS files
- Check browser console for image loading errors

## References

- **Paper**: https://arxiv.org/abs/1812.08045
- **Model Source**: https://github.com/ISICV/ManTraNet (model only, no clone needed)
- **Colab Demo**: https://colab.research.google.com/drive/1ai4kVlI6w9rREqqYnTfpk3gM3YX9k-Ek

**Note**: Our implementation uses only the pretrained frozen graph file (.pb). The inference code is self-contained in `mantranet_infer.py`, so we don't need to clone or depend on the original repository.

## Cleanup

Remove ManTraNet:
```bash
make mantranet-clean
```

This removes the `models/weights/mantranet/` directory. TensorFlow remains installed in the main venv (may be used by other modules).
