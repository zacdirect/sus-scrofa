# AI Detection System

Multi-layer AI-generated image detection using a **gatekeeper architecture**.

## Architecture: Three Components

```
┌──────────────┐
│ Orchestrator │ ← Runs detectors efficiently
└──────┬───────┘
       │
       ├─▶ Detector 1 (fast) → findings
       ├─▶ Auditor: should_stop_early()? ← GATEKEEPER
       │   ├─ YES: stop
       │   └─ NO: continue
       ├─▶ Detector 2 (slower) → findings
       └─▶ Auditor: detect() → FINAL VERDICT
```

### 1. 🎯 Orchestrator (`MultiLayerDetector`)
- Runs detectors in efficient order (fast → slow)
- Consults auditor after each detector
- Pure operational logic - no decisions

### 2. 🔍 Detectors (Specialized Analyzers)
Detectors focus on what they know - they don't decide "fake or real":

- **AI-Focused**: `SPAIDetector` - reports AI generation evidence
- **Manipulation-Focused**: Future ELA/forensic detectors - report editing evidence
- **Multi-Aspect**: `MetadataDetector` - may find AI tags OR manipulation signs

Each returns: confidence + score + detected_types (what they found)

**Key**: Detectors are specialized - some only see AI, some only see edits, some see both

### 3. ⚖️ Auditor (`ComplianceAuditor`)
**THE GATEKEEPER** - Not a detector!
- Reviews results after each detector
- Decides when to stop early (saves compute)
- Performs final compliance audit
- **Consolidates varied findings into three buckets**:
  1. Authenticity Score (0-100): fake ← → real
  2. AI Generation Probability: synthetic content evidence
  3. Manipulation Probability: traditional editing evidence

**Why consolidation matters**: Different detectors report different things (AI, manipulation, both). Auditor unifies into consistent three-bucket output.

## Key Design Principle

> **The auditor is NOT a detector.** It's a separate gatekeeper component that reviews detectors and makes all decisions.

```python
# Clear separation
orchestrator.detectors = [MetadataDetector(), SPAIDetector()]
orchestrator.auditor = ComplianceAuditor()  # Not in detectors list!
```

## Structure

```
ai_detection/
├── README.md              # This file
├── Makefile              # Setup and installation
├── requirements.txt      # PyTorch + dependencies
├── spai_infer.py         # Standalone inference script
├── detectors/            # Detection framework
│   ├── __init__.py
│   ├── base.py           # BaseDetector interface
│   ├── metadata.py       # MetadataDetector (EXIF/XMP)
│   ├── spai_detector.py  # SPAIDetector (ML wrapper)
│   ├── compliance_audit.py  # ComplianceAuditor (THE GATEKEEPER)
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

# Initialize (creates detectors + auditor)
detector = MultiLayerDetector()

# Run detection
result = detector.detect('image.jpg')

# Get final verdict (from auditor)
print(f"Authenticity: {result['authenticity_score']}/100")
print(f"Verdict: {'FAKE' if result['overall_verdict'] else 'REAL'}")
```

## Detection Framework

### Orchestrator
Coordinates multiple detection methods:
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
