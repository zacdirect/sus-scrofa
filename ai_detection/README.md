# AI Detection System

Multi-layer AI-generated image detection integrated into the engine's compliance auditor.

## Architecture

```
┌──────────────────────┐
│ Orchestrator         │ ← Runs ML detectors efficiently
│ (MultiLayerDetector) │
└──────┬───────────────┘
       │
       ├─▶ MetadataDetector (fast) → detection_layers[]
       ├─▶ SDXLDetector (ML model) → detection_layers[]
       ├─▶ SPAIDetector (ML model) → detection_layers[]
       │
       └─▶ Results → ai_detection: {detection_layers: [...]}
                                                 │
                                                 ▼
                    ┌────────────────────────────────────────┐
                    │ Engine: lib/analyzer/auditor.py        │
                    │ - Reads detection_layers[]             │
                    │ - Creates audit findings               │
                    │ - Calculates authenticity score 0-100  │
                    └────────────────────────────────────────┘
```

### Components

**1. Orchestrator (`MultiLayerDetector`)**
- Runs detectors in efficient order (fast → slow)
- Pure operational logic - no scoring decisions
- Returns raw `detection_layers` array

**2. Detectors (Specialized Analyzers)**
Each detector focuses on what it knows:

- `MetadataDetector` - EXIF/XMP analysis for AI signatures
- `SDXLDetector` - Organika/sdxl-detector ML model (Swin Transformer)
- `SPAIDetector` - SPAI spectral analysis (ViT + frequency analysis)

Returns: `{method, verdict, confidence, score, evidence}`

**3. Engine Auditor (`lib/analyzer/auditor.py`)**
- NOT part of ai_detection package
- Reads `ai_detection.detection_layers[]` from results
- Creates audit findings: positive/negative, LOW/MEDIUM/HIGH
- Calculates final authenticity score (0-100)

**Key Design**: Detectors report findings, engine auditor scores them.

## Structure

```
ai_detection/
├── README.md              # This file
├── Makefile              # Setup and installation
├── requirements.txt      # PyTorch + dependencies
├── spai_infer.py         # Standalone SPAI inference script
├── detectors/            # Detection framework
│   ├── __init__.py
│   ├── base.py           # BaseDetector interface
│   ├── metadata.py       # MetadataDetector (EXIF/XMP)
│   ├── sdxl_detector.py  # Organika/sdxl-detector wrapper
│   ├── spai_detector.py  # SPAI wrapper
│   └── orchestrator.py   # MultiLayerDetector
├── spai/                 # SPAI model implementation
│   ├── config.py         # Model configuration
│   ├── inference.py      # Inference interface
│   ├── models/           # Architecture (ViT + FRE)
│   └── data/             # Data loaders
└── .venv/                # Isolated Python environment (gitignored)
```

## Quick Start

```python
from ai_detection.detectors.orchestrator import MultiLayerDetector

# Initialize
detector = MultiLayerDetector()

# Run detection
result = detector.detect('image.jpg')

# Result contains raw detection layers
for layer in result['detection_layers']:
    print(f"{layer['method']}: {layer['verdict']} ({layer['confidence']})")
    print(f"  Evidence: {layer['evidence']}")

# Engine auditor (lib/analyzer/auditor.py) will score these findings
```
- Runs detectors in order (lowest order first)
- Early stopping: Stops at CERTAIN/HIGH confidence
- Weighted combination: Combines results when no single method is decisive
- Confidence weights: CERTAIN=1.0, HIGH=0.8, MEDIUM=0.5, LOW=0.3

### MetadataDetector (order=0)
**Fast metadata-based detection (100% accurate when present)**

**AI Signatures**: Midjourney, DALL-E, Stable Diffusion, Firefly, Imagen, Leonardo.ai, Ideogram, Flux, and 10+ more

**Technology**: GExiv2 (comprehensive XMP/EXIF) + PIL (fallback)

### SPAIDetector (order=100)
**ML-based detection via spectral analysis**

**Model**: ViT-B/16 + Frequency Restoration Estimator (CVPR 2025)
**Weights**: 892MB (spai.pth)
**Resolution**: Arbitrary (224x224 patches, any image size)

## Setup

### Quick Setup (from project root)
```bash
# Setup AI detection (creates venv, installs dependencies, downloads weights)
make ai-setup

# Verify installation
make ai-verify
```

### Manual Setup (from ai_detection directory)
```bash
cd ai_detection
make setup        # Creates venv and installs dependencies
make weights      # Downloads model weights
make verify       # Verifies installation
```

## Usage from SusScrofa Plugin

The plugin automatically uses the embedded SPAI code:

```python
from ai_detection.spai.inference import SPAIDetector

# Initialize detector (loads model once)
detector = SPAIDetector(
    weights_path="ai_detection/weights/spai.pth",
    device="cuda"  # or "cpu"
)

# Analyze an image
result = detector.predict(image_path)
# Returns: {'score': 0.95, 'confidence': 0.89, 'is_ai_generated': True}
```

## Requirements

### Hardware
- **CPU mode**: Any modern CPU (slower)
- **GPU mode**: CUDA-capable GPU with 8GB+ VRAM (recommended)
- **Disk**: ~500MB for dependencies, ~100MB for weights

### Software
- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

## Updating AI Detection Method

Since AI detection is rapidly evolving, this module is designed for easy replacement:

1. **Add new method**: Create `ai_detection/new_method/` directory
2. **Implement interface**: Follow the same structure as SPAI
3. **Update plugin**: Modify `plugins/analyzer/ai_detection.py` to use new method
4. **Update Makefile**: Add setup targets for new method

Example structure for new method:
```python
# ai_detection/new_method/inference.py
class NewMethodDetector:
    def __init__(self, weights_path, device="cpu"):
        # Load model
        pass
    
    def predict(self, image_path):
        # Return: {'score': float, 'confidence': float, 'is_ai_generated': bool}
        pass
```

## Performance Considerations

### Model Loading
- **First call**: 3-5 seconds (one-time model loading)
- **Subsequent calls**: <1 second per image
- **Best practice**: Initialize detector once, reuse for all images in a case

### Batch Processing
For analyzing multiple images, use batch mode:
```python
detector = SPAIDetector(weights_path="weights/spai.pth")
results = detector.predict_batch([img1, img2, img3])
```

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
cd ai_detection
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Out of Memory
- Reduce batch size
- Use CPU mode: `device="cpu"`
- Clear GPU cache between batches

### Model Loading Issues
```bash
# Verify weights file
cd ai_detection
make verify
```

## References

- **SPAI Paper**: https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html
- **SPAI Repository**: https://github.com/mever-team/spai
- **Model Weights**: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view

## License

The embedded SPAI code is licensed under Apache 2.0 License.
Model weights are subject to the original SPAI project's license terms.
