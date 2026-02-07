# Ghiro AI Detection Module

This directory contains embedded AI detection methods for analyzing potentially AI-generated images. The implementations are designed to be modular and easily replaceable as the field evolves.

## Structure

```
ai_detection/
├── README.md              # This file
├── Makefile              # Setup and installation for AI detection
├── __init__.py           # Module initialization
├── requirements.txt      # AI detection dependencies
├── spai/                 # SPAI (Spectral AI-Generated Image Detector)
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── inference.py      # Main inference interface
│   ├── models/           # Model architecture code
│   └── data/             # Data loading utilities
└── weights/              # Model weights (gitignored, download separately)
    └── spai.pth          # ~100MB, see setup instructions
```

## Current Method: SPAI (CVPR 2025)

**Spectral AI-Generated Image Detector** - State-of-the-art detector using frequency domain analysis.

- **Paper**: "Any-Resolution AI-Generated Image Detection by Spectral Learning" (CVPR 2025)
- **Repository**: https://github.com/mever-team/spai
- **Approach**: Self-supervised spectral learning with masked frequency restoration
- **Advantages**:
  - Works on any resolution images (no resizing needed)
  - Spectral analysis (frequency domain features)
  - State-of-the-art accuracy across multiple generators

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

## Usage from Ghiro Plugin

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
