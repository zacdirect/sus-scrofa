# Automated Analysis - Unified Detection System

## Overview

The Automated Analysis feature consolidates multiple forensics detection methods into a single, comprehensive view. This unified interface combines AI generation detection and image manipulation detection to provide researchers with a holistic assessment of image authenticity.

## Features

### 1. AI Generation Detection (Multi-Layer)

Detects AI-generated images using multiple detection layers with early stopping optimization:

#### Detection Layers (in order):

1. **Metadata Detector** (Order: 0)
   - Fastest detection method
   - Checks EXIF/XMP metadata for AI generator signatures
   - Detects: Midjourney, DALL-E, Stable Diffusion, Adobe Firefly, etc.
   - Example markers: "Software: Midjourney", "ai-generated: true"

2. **Filename Pattern Detector** (Order: 0)
   - Checks filename for AI generation indicators
   - Patterns: gemini_generated, imagen_generated, ai_generated, dall_e, midjourney, etc.
   - 20+ patterns covering major AI generators
   - Returns HIGH confidence when matched

3. **SPAI ML Model** (Order: 100)
   - CVPR 2025 state-of-the-art model
   - 892MB trained weights
   - Fallback when metadata/filename checks fail
   - Deep learning analysis of image artifacts

#### Framework Benefits:
- **Early Stopping**: Stops at first high-confidence detection
- **Performance**: Metadata checks complete in milliseconds
- **Accuracy**: ML model provides backup for stripped metadata
- **Transparency**: Shows which layer detected AI generation

### 2. Image Manipulation Detection (OpenCV)

Containerized microservice using OpenCV for manipulation analysis:

#### Detection Methods:

1. **Gaussian Blur Analysis**
   - Detects cloned or copied regions
   - Compares original vs. blurred difference
   - Identifies manipulation confidence and coverage

2. **Noise Consistency Analysis**
   - Analyzes Laplacian variance across quadrants
   - Detects inconsistent noise patterns
   - Highlights spliced or composite regions

3. **JPEG Artifact Analysis**
   - Examines DCT coefficient variations
   - Detects compression inconsistencies
   - Identifies regions with different compression history

#### Service Architecture:
- **Container**: Podman/Docker (demisto/dockerfiles based)
- **Base**: Python 3.12-slim
- **Libraries**: opencv-contrib-python 4.13.0.92, imutils, numpy
- **API**: Flask REST on port 8080
- **Endpoints**: `/health`, `/analyze`

## Template Structure

### Main Template: `_automated_analysis.html`

The unified template is organized into three main sections:

#### 1. Overall Summary Card
- Combined status of both detection types
- Quick visual indicators (labels)
- High-level confidence scores

#### 2. AI Generation Detection Section
- Detection framework information
- Multi-layer results table showing:
  - Method used (Metadata, Filename, ML Model)
  - Verdict (AI/Real/Unknown)
  - Confidence level (CERTAIN/HIGH/MEDIUM/LOW)
  - Evidence details per layer
- AI probability progress bar (0-100%)
- Detailed assessment and interpretation
- List of available detection methods

#### 3. Manipulation Detection Section
- Overall confidence progress bar
- Three sub-cards for each method:
  - **Gaussian Blur Analysis**: Anomaly count, coverage percentage
  - **Noise Consistency**: Quadrant variances, coefficient variation
  - **JPEG Artifacts**: Compression variation analysis
- Each method shows:
  - Status badge (Detected/Clean, Inconsistent/Consistent)
  - Confidence score
  - Technical metrics
  - Evidence interpretation

### Integration with `show.html`

The unified "Automated Analysis" tab replaces individual plugin tabs:

**Before:**
- Separate "AI Detection" tab
- (No separate OpenCV tab existed yet)

**After:**
- Single "Automated Analysis" tab
- Shows when either `analysis.report.ai_detection` or `analysis.report.opencv_manipulation` exists
- Consolidated view of all automatic detection methods

## Setup Instructions

### Prerequisites

1. **Django 4.2.17+** (already installed)
2. **Python 3.13+** (already installed)
3. **Podman or Docker** (for OpenCV service)

### AI Detection Setup

```bash
# Install AI detection dependencies
make ai-setup

# This installs:
# - PyTorch with CUDA support
# - timm (PyTorch Image Models)
# - pillow
# - numpy
# Downloads SPAI model weights (~892MB)
```

**GPU Recommended**: AI detection runs 10-100x faster with CUDA GPU

### Manipulation Detection Setup

```bash
# Build OpenCV container
make opencv-build

# Start OpenCV service
make opencv-start

# Verify service is running
make opencv-status
curl http://localhost:8080/health
```

**Container Management:**
```bash
make opencv-stop      # Stop service
make opencv-restart   # Restart service
make opencv-logs      # View logs
```

## Data Structure

### AI Detection Report Format

```python
analysis.report.ai_detection = {
    "enabled": True,
    "likely_ai": False,  # Overall verdict
    "ai_probability": 23.5,  # 0-100 score
    "confidence": "high",  # certain/high/medium/low/none
    "interpretation": "High confidence: Image appears authentic",
    "evidence": "No AI generation markers found...",
    "detection_framework": "Multi-Layer (Early Stopping)",
    "detection_layers": [
        {
            "method": "Metadata Detection",
            "verdict": "Real",
            "confidence": "HIGH",
            "score": 0.0,
            "evidence": "No AI generation markers in EXIF/XMP"
        }
    ],
    "available_methods": ["metadata", "filename", "ml_model"]
}
```

### Manipulation Detection Report Format

```python
analysis.report.opencv_manipulation = {
    "enabled": True,
    "is_suspicious": True,
    "overall_confidence": 49.7,  # 0-100
    "interpretation": "Low confidence: Some suspicious patterns found",
    "service": "OpenCV Container Service (v4.13.0.92)",
    "methods": ["Gaussian Blur", "Noise Analysis", "JPEG Artifacts"],
    
    "manipulation_detection": {
        "method": "Gaussian Blur Difference",
        "is_manipulated": True,
        "confidence": 58.2,
        "num_anomalies": 2322,
        "anomaly_percentage": 1.64,
        "evidence": "Found 2,322 potential manipulation regions..."
    },
    
    "noise_analysis": {
        "method": "Laplacian Variance",
        "is_noise_inconsistent": False,
        "noise_consistency": 68.3,
        "overall_noise": 456.78,
        "coefficient_variation": 0.317,
        "quadrant_variances": [442.1, 458.3, 461.2, 465.8]
    },
    
    "jpeg_artifacts": {
        "method": "DCT Coefficient Analysis",
        "has_inconsistent_artifacts": True,
        "confidence": 56.3,
        "compression_variation": 1.127,
        "evidence": "Detected compression inconsistencies..."
    }
}
```

## Usage Example

### Upload and Analyze Image

1. Upload image through SusScrofa web interface
2. Analysis runs automatically (both plugins enabled)
3. View report and click "Automated Analysis" tab
4. Review comprehensive results:
   - AI generation verdict and evidence
   - Manipulation detection findings
   - Technical details per method

### Interpreting Results

#### AI Detection:
- **AI-Generated (High Confidence)**: Strong evidence of AI generation
  - Metadata markers found, or
  - Filename patterns matched, or
  - ML model score > 85%
- **Authentic (High Confidence)**: No AI markers detected
- **Unknown**: Insufficient evidence (stripped metadata, borderline ML score)

#### Manipulation Detection:
- **Suspicious**: One or more methods detected anomalies
  - Overall confidence > 50%
  - Multiple detection methods agree
- **No Issues**: All methods show consistency
  - Overall confidence < 50%
  - No significant anomalies found

### Common Scenarios

#### Scenario 1: AI-Generated with Metadata
- **Metadata Detector**: Detects "Midjourney" in Software field
- **Result**: HIGH confidence, stops at layer 0
- **Display**: Shows metadata evidence, no ML model needed

#### Scenario 2: AI-Generated, Stripped Metadata
- **Metadata Detector**: No markers found
- **Filename Detector**: No patterns matched
- **SPAI Model**: Analyzes image, detects AI artifacts (93.2%)
- **Result**: CERTAIN confidence from ML model
- **Display**: Shows all three layers, ML evidence highlighted

#### Scenario 3: Manipulated Photo
- **Gaussian Blur**: Detects cloned region (65% confidence)
- **Noise Analysis**: Inconsistent noise (72% confidence)
- **JPEG Artifacts**: Compression inconsistencies (58% confidence)
- **Result**: Overall 65% suspicious
- **Display**: Shows all three methods with detailed metrics

#### Scenario 4: Authentic Photo
- **AI Detection**: Metadata shows "Canon EOS" camera
- **Manipulation**: All methods show consistency
- **Result**: Authentic with high confidence
- **Display**: Green labels, low confidence scores

## Performance Considerations

### AI Detection Performance

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Metadata | <10ms | High (when present) | First check |
| Filename | <1ms | Medium | Quick indicator |
| SPAI Model | 2-5s (GPU) | Very High | Fallback |
| SPAI Model | 30-60s (CPU) | Very High | Fallback (slow) |

**Optimization**: Early stopping prevents unnecessary ML model invocation

### Manipulation Detection Performance

- **Container Startup**: 5-10 seconds
- **Analysis per Image**: 1-3 seconds
- **Memory**: ~200-300MB per container
- **CPU**: 1-2 cores recommended

**Scaling**: Run multiple containers for parallel processing

## Troubleshooting

### AI Detection Not Available

**Symptoms**: "Analysis Not Available" message in AI section

**Solutions**:
```bash
# Check if dependencies installed
python -c "import torch; print(torch.__version__)"

# Reinstall if needed
make ai-setup

# Check model weights
ls -lh ai_detection/models/spai_model.pth
# Should be ~892MB
```

### Manipulation Detection Not Available

**Symptoms**: "Analysis Not Available" message in manipulation section

**Solutions**:
```bash
# Check service status
make opencv-status

# Check service health
curl http://localhost:8080/health

# Restart service
make opencv-restart

# Check logs
make opencv-logs
```

### Plugin Not Running

**Check Plugin Registration**:
```python
# In Django shell
python manage.py shell

from plugins.analyzer.ai_detection import AIDetection
from plugins.analyzer.opencv_manipulation import OpenCVManipulation

# Check if loaded
print(AIDetection.order)  # Should be 60
print(OpenCVManipulation.order)  # Should be 65
```

### High False Positive Rate

**AI Detection**:
- Lower SPAI threshold in settings (default: 0.5)
- Rely more on metadata/filename detection
- Consider retraining model with domain-specific data

**Manipulation Detection**:
- Adjust threshold in opencv_service/service.py
- Current default: 1.5 std dev for anomalies
- Increase threshold to reduce false positives

## Technical Details

### SPAI Model Architecture

- **Base**: Vision Transformer (ViT)
- **Patches**: 16x16 patches
- **Features**: Frequency domain analysis
- **Training**: 2M+ images (AI + real)
- **Output**: Single score (0-1, higher = more AI)

### OpenCV Methods Explained

#### Gaussian Blur Difference
1. Apply Gaussian blur (21x21 kernel)
2. Compute absolute difference from original
3. Threshold difference (> 30)
4. Count anomalous pixels
5. Calculate confidence based on coverage

#### Laplacian Noise Analysis
1. Convert to grayscale
2. Divide into 4 quadrants
3. Calculate Laplacian variance per quadrant
4. Compute coefficient of variation
5. High CV = inconsistent noise = potential manipulation

#### JPEG Artifact Analysis
1. Convert to YCrCb color space
2. Apply DCT to 8x8 blocks
3. Calculate mean absolute values per channel
4. Compute variation across channels
5. High variation = compression inconsistencies

## API Integration

### Standalone OpenCV Service

The manipulation detection runs as a separate microservice:

```python
import requests

# Analyze image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8080/analyze',
        files={'image': f},
        timeout=30
    )

results = response.json()
print(f"Suspicious: {results['results']['is_suspicious']}")
print(f"Confidence: {results['results']['overall_confidence']}")
```

### SusScrofa Plugin Interface

Plugins communicate through standard SusScrofa interface:

```python
from plugins.analyzer.opencv_manipulation import OpenCVManipulation

plugin = OpenCVManipulation()
results = plugin.run(task)

# Results stored in analysis.report.opencv_manipulation
```

## Future Enhancements

### Planned Features

1. **Additional Detection Methods**
   - Shadow inconsistency analysis
   - Perspective distortion detection
   - Lighting direction analysis
   - Chromatic aberration checks

2. **Machine Learning Enhancements**
   - Fine-tune SPAI on domain-specific images
   - Add GAN fingerprint detection
   - Implement ensemble models

3. **UI Improvements**
   - Visual heatmaps for manipulation regions
   - Interactive comparison views
   - Detailed technical reports (PDF export)

4. **Performance Optimizations**
   - Batch processing support
   - GPU acceleration for OpenCV
   - Caching frequent analyses

## References

### Papers

1. **SPAI**: "Towards Universal Fake Image Detectors that Generalize Across Generative Models" (CVPR 2025)
2. **ELA**: Error Level Analysis for JPEG manipulation
3. **Gaussian Blur**: Clone detection via blur difference
4. **Laplacian**: Noise inconsistency detection

### External Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [demisto/dockerfiles](https://github.com/demisto/dockerfiles)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)

## Support

For issues or questions:
1. Check logs: `make opencv-logs` or Django logs
2. Verify setup: Run health checks
3. Review this documentation
4. Check MODERNIZATION.md for upgrade notes

## License

All detection plugins follow SusScrofa's license (see LICENSE.txt)
