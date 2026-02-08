# AI Detection Model Evaluation Framework

This directory provides a standardized framework for evaluating AI detection models and integrating them into the Sus Scrofa analysis pipeline.

## Directory Structure

```
ai_detection/model_evaluation/
├── README.md                    # This file
├── Makefile                     # Automation for testing and evaluation
├── baseline/                    # Baseline test results
│   ├── spai_baseline.json
│   └── metadata_baseline.json
├── candidates/                  # New models being evaluated
│   ├── model_name/
│   │   ├── detector.py          # Detector wrapper
│   │   ├── requirements.txt     # Model dependencies
│   │   ├── setup.sh             # Setup script
│   │   └── test_results.json   # Evaluation results
│   └── ...
├── test_data/                   # Test dataset (NOT in git)
│   ├── .gitignore              # Excludes all test images
│   ├── README.md               # Setup instructions
│   ├── fake/                   # AI-generated images
│   ├── edited/                 # Human-edited images
│   └── real/                   # Authentic camera photos
├── framework/
│   ├── __init__.py
│   ├── base_tester.py           # Base class for model testing
│   ├── metrics.py               # Evaluation metrics
│   ├── comparison.py            # Compare multiple models
│   └── report_generator.py     # Generate comparison reports
└── tests/
    ├── test_detector_api.py     # Unit tests for detector interface
    └── test_integration.py      # Integration tests
```

## Quick Start

### 1. Setup Test Data

```bash
cd ai_detection/model_evaluation

# Copy your test images into the appropriate directories
cp /path/to/ai-generated/*.jpg test_data/fake/
cp /path/to/ai-edited/*.jpg test_data/edited/
cp /path/to/real-photos/*.jpg test_data/real/

# Check your test data
make check-test-data
```

**Note**: All files in `test_data/` are excluded from git via `.gitignore`. Your test images remain private. See `test_data/README.md` for detailed setup instructions.

### 2. Run Baseline Tests

```bash
# Test current detectors
make baseline

# This creates baseline/*.json files
```

### 3. Evaluate a New Model

```bash
# Create new candidate directory
make new-candidate NAME=my_model

# Add your detector implementation
# Edit candidates/my_model/detector.py

# Test the candidate
make test-candidate NAME=my_model

# Compare with baseline
make compare NAME=my_model
```

### 4. Integrate Best Model

```bash
# Copy to main detector
make integrate NAME=my_model
```

## Test Dataset Requirements

Your `test_data/` directory should contain:

### fake/ - AI-Generated Images
- **Minimum**: 10 images from various AI generators
- **Recommended**: 50+ images
- **Sources**: Midjourney, DALL-E, Stable Diffusion, etc.
- **Diversity**: Different styles, resolutions, post-processing levels

### edited/ - Human-Edited Images
- **Minimum**: 5 images with known edits
- **Recommended**: 20+ images
- **Edits**: Photoshop adjustments, filters, crops, etc.
- **Purpose**: Test false positive rate

### real/ - Authentic Photos
- **Minimum**: 5 unedited camera photos
- **Recommended**: 20+ images
- **Sources**: Direct from camera (with EXIF)
- **Purpose**: Baseline for authentic content

## Evaluation Criteria

### Primary Metrics

1. **Accuracy**: Correct classifications / Total images
2. **Precision**: True Positives / (True Positives + False Positives)
3. **Recall**: True Positives / (True Positives + False Negatives)
4. **F1 Score**: Harmonic mean of precision and recall

### Category-Specific Metrics

- **Fake Detection Rate**: Accuracy on `fake/` images
- **False Positive Rate**: Misclassified `real/` images
- **Edit Sensitivity**: How many `edited/` flagged as manipulated

### Performance Metrics

- **Inference Time**: Average time per image
- **Memory Usage**: Peak memory during inference
- **Setup Complexity**: Installation difficulty (1-5)
- **Dependency Count**: Number of external dependencies

### Integration Metrics

- **API Compliance**: Follows BaseDetector interface
- **Error Handling**: Graceful failures
- **Logging Quality**: Useful debug information
- **Documentation**: Clear usage instructions

## Model Selection Criteria

A good AI detection model should:

### ✅ Minimum Requirements
- [ ] ≥70% accuracy on test dataset
- [ ] ≤10% false positive rate on real images
- [ ] <5 seconds inference time per image
- [ ] Follows BaseDetector API
- [ ] Clear licensing (commercial use allowed)
- [ ] Active maintenance (updated within 1 year)

### ⭐ Nice to Have
- [ ] ≥85% accuracy
- [ ] Handles multiple AI generators
- [ ] GPU acceleration support
- [ ] Explainable results (attention maps, etc.)
- [ ] Pre-trained weights available
- [ ] Paper published in peer-reviewed venue

### 🚫 Dealbreakers
- ❌ Requires paid API keys
- ❌ Sends data to external servers
- ❌ License incompatible with Apache 2.0
- ❌ Requires >16GB RAM
- ❌ Takes >120 seconds per image

## Adding a New Detector

### Step 1: Create Candidate Directory

```bash
make new-candidate NAME=my_detector
cd candidates/my_detector
```

### Step 2: Implement Detector

Edit `detector.py` following the BaseDetector interface:

```python
from ai_detection.detectors.base import BaseDetector, DetectionResult, DetectionMethod, ConfidenceLevel

class MyDetector(BaseDetector):
    """Brief description of the detection method."""
    
    def __init__(self):
        super().__init__()
        # Initialize your model
        self._model = None
    
    def get_order(self) -> int:
        """Execution order (higher = later)."""
        return 50  # After metadata, before ML models
    
    def check_deps(self) -> bool:
        """Check if dependencies are available."""
        try:
            import required_library
            return True
        except ImportError:
            return False
    
    def detect(self, image_path: str) -> DetectionResult:
        """
        Analyze image for AI generation.
        
        Args:
            image_path: Path to image file
            
        Returns:
            DetectionResult with verdict and confidence
        """
        try:
            # Your detection logic here
            score = self._model.predict(image_path)
            
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,  # or METADATA, FORENSIC, etc.
                is_ai_generated=score > 0.5,
                confidence=ConfidenceLevel.HIGH if abs(score - 0.5) > 0.3 else ConfidenceLevel.MEDIUM,
                score=float(score),
                evidence=f"Model probability: {score:.2%}",
                metadata={'model': 'my_model_v1'}
            )
        except Exception as e:
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence=f"Error: {str(e)}"
            )
```

### Step 3: Add Dependencies

Create `requirements.txt`:
```
numpy>=1.20.0
pillow>=9.0.0
torch>=2.0.0  # if needed
```

### Step 4: Create Setup Script

Create `setup.sh`:
```bash
#!/bin/bash
set -e

echo "Setting up My Detector..."

# Download weights
wget https://example.com/model.pth -O weights/my_model.pth

# Install dependencies
pip install -r requirements.txt

echo "✓ Setup complete"
```

### Step 5: Test

```bash
# Run automated tests
make test-candidate NAME=my_detector

# Review results
cat candidates/my_detector/test_results.json
```

## Wrapping External Models

### Pattern 1: Python Package

If the model is a Python package:

```python
class MyDetector(BaseDetector):
    def __init__(self, weights_path: str = None):
        super().__init__()
        import external_package
        
        if weights_path is None:
            weights_path = Path(__file__).parent / "weights" / "model.pth"
        
        self._model = external_package.load_model(weights_path)
    
    def detect(self, image_path: str) -> DetectionResult:
        from PIL import Image
        
        img = Image.open(image_path)
        score = self._model.predict(img)
        
        return DetectionResult(...)
```

### Pattern 2: Subprocess (Isolated Environment)

If the model needs its own environment (like SPAI):

```python
class MyDetector(BaseDetector):
    def detect(self, image_path: str) -> DetectionResult:
        import subprocess
        import json
        
        venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
        script = Path(__file__).parent / "infer.py"
        
        result = subprocess.run(
            [str(venv_python), str(script), image_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return DetectionResult(...)  # error case
        
        data = json.loads(result.stdout)
        return DetectionResult(
            method=DetectionMethod.ML_MODEL,
            is_ai_generated=data['is_ai'],
            score=data['score'],
            ...
        )
```

### Pattern 3: REST API

If the model runs as a service:

```python
class MyDetector(BaseDetector):
    def __init__(self, api_url: str = "http://localhost:5000"):
        super().__init__()
        self._api_url = api_url
    
    def check_deps(self) -> bool:
        import requests
        try:
            r = requests.get(f"{self._api_url}/health", timeout=2)
            return r.status_code == 200
        except:
            return False
    
    def detect(self, image_path: str) -> DetectionResult:
        import requests
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            r = requests.post(
                f"{self._api_url}/predict",
                files=files,
                timeout=30
            )
        
        data = r.json()
        return DetectionResult(...)
```

## Comparing Models

### Generate Comparison Report

```bash
# Compare all candidates with baseline
make compare-all

# Output: reports/comparison_YYYY-MM-DD.md
```

### Manual Comparison

```python
from framework.comparison import ModelComparison

comparison = ModelComparison(
    baseline_path="baseline/spai_baseline.json",
    candidate_paths=[
        "candidates/model_a/test_results.json",
        "candidates/model_b/test_results.json"
    ]
)

report = comparison.generate_report()
print(report)
```

## Best Practices

### Testing
1. **Reproducibility**: Use fixed random seeds
2. **Isolation**: Test each model independently
3. **Logging**: Capture all warnings and errors
4. **Version Control**: Track model versions in results

### Documentation
1. **Model Source**: Link to paper/repo
2. **Training Data**: What was it trained on?
3. **Known Limitations**: What does it struggle with?
4. **Citation**: How to cite if used in research

### Performance
1. **Caching**: Cache model loading between tests
2. **Batch Processing**: Process multiple images when possible
3. **GPU Support**: Add GPU flag for faster inference
4. **Timeouts**: Set reasonable timeouts (30s default)

### Security
1. **Input Validation**: Check file types
2. **Resource Limits**: Prevent memory exhaustion
3. **Sandboxing**: Use subprocess for untrusted models
4. **Dependency Audit**: Check for vulnerabilities

## Troubleshooting

### Model Not Loading
- Check `check_deps()` returns True
- Verify weights file exists and is not corrupted
- Check Python version compatibility
- Review error logs in `test_results.json`

### Poor Performance
- Verify preprocessing matches training
- Check if model trained on similar data
- Review threshold calibration
- Test on official evaluation dataset

### Integration Issues
- Verify BaseDetector interface compliance
- Check return type annotations
- Ensure graceful error handling
- Test with malformed inputs

## Model Library

### Recommended Models to Evaluate

1. **NPR (Neural Pathways Recognition)**
   - Paper: "Detecting AI-Generated Images via Neural Pathways" (CVPR 2024)
   - Repo: https://github.com/chuangchuangtan/NPR-DeepfakeDetection
   - Strengths: High accuracy, fast inference
   - Weaknesses: Requires specific preprocessing

2. **CNNDetection**
   - Paper: "CNN-generated images are surprisingly easy to spot...for now"
   - Repo: https://github.com/PeterWang512/CNNDetection
   - Strengths: Well-established, simple
   - Weaknesses: May not generalize to newer models

3. **UniversalFakeDetect**
   - Paper: "Towards Universal Fake Image Detectors"
   - Repo: https://github.com/Yuheng-Li/UniversalFakeDetect
   - Strengths: Trained on diverse generators
   - Weaknesses: Larger model size

4. **DIRE (Diffusion Reconstruction Error)**
   - Paper: "DIRE for Diffusion-Generated Image Detection"
   - Repo: https://github.com/ZhendongWang6/DIRE
   - Strengths: Specifically targets diffusion models
   - Weaknesses: Limited to diffusion-based generators

### Current Baseline
- **SPAI**: 50% accuracy on test set (needs improvement)
- **Metadata Detector**: 0% on stripped images (supplementary only)

## Files NOT in Git

The following are excluded via `.gitignore`:

```
ai_detection/model_evaluation/test_data/          # Your private test images
ai_detection/model_evaluation/candidates/*/weights/  # Downloaded model weights
ai_detection/model_evaluation/candidates/*/.venv/   # Virtual environments
ai_detection/model_evaluation/**/test_results.json  # Generated results
ai_detection/model_evaluation/reports/*.html        # Generated reports
```

All test images should be kept locally or in a separate private repository.

## Contributing

When adding a new candidate model:

1. Create PR with detector implementation only
2. Include requirements.txt and setup.sh
3. Document test results in PR description
4. Add model to "Model Library" section
5. Update comparison charts

Do NOT commit:
- Test images
- Model weights files
- Test result JSON files
- Virtual environments

## License

This evaluation framework is part of SusScrofa/Ghiro and follows the same Apache 2.0 license. Individual models may have their own licenses - check before integration.
