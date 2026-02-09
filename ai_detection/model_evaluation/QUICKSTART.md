# Quick Start Guide: AI Detection Model Evaluation

## Initial Setup (One Time)

### 1. Copy Your Test Data

```bash
cd ai_detection/model_evaluation

# Copy your test images into the appropriate directories
cp /path/to/ai-generated/*.jpg test_data/fake/
cp /path/to/ai-edited/*.jpg test_data/edited/
cp /path/to/real-photos/*.jpg test_data/real/

# Verify structure
ls test_data/*/
# Should show images in: fake/ edited/ real/
```

**Note**: All files in `test_data/` are excluded from git. Your test images remain private.

The test_data directory structure:
```
test_data/
├── fake/     # AI-generated images (min 10 recommended)
├── edited/   # AI-edited images (min 5 recommended)
└── real/     # Real photographs (min 5 recommended)
```

### 2. Create Baseline

```bash
# Test current detectors to establish baseline
make baseline

# This will test:
# - SPAI model
# - Metadata detector
# Results saved to: baseline/summary.json
```

## Evaluating a New Model

### Step 1: Create Candidate

```bash
# Create new model directory
make new-candidate NAME=npr_detector

# This creates:
# candidates/npr_detector/
# ├── detector.py       # Template to implement
# ├── requirements.txt  # Dependencies
# ├── setup.sh         # Setup script
# ├── weights/         # For model weights
# └── README.md        # Documentation
```

### Step 2: Implement Detector

Edit `candidates/npr_detector/detector.py`:

```python
class NPRDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        # Load your model
        
    def detect(self, image_path: str, context=None) -> DetectionResult:
        # Your detection logic
        score = your_model.predict(image_path)
        
        return DetectionResult(
            method=DetectionMethod.ML_MODEL,
            is_ai_generated=score > 0.5,
            confidence=ConfidenceLevel.HIGH,
            score=float(score),
            evidence=f"Model score: {score:.4f}"
        )
```

### Step 3: Setup Model

```bash
cd candidates/npr_detector

# Add dependencies to requirements.txt
# Download weights (manual or via setup.sh)

# Run setup script
./setup.sh

# Or manually:
# pip install -r requirements.txt
# wget -O weights/model.pth <download_url>
```

### Step 4: Test

```bash
# Return to model_evaluation directory
cd ../..

# Run evaluation
make test-candidate NAME=npr_detector

# Results saved to: candidates/npr_detector/test_results.json
```

### Step 5: Compare

```bash
# Compare with baseline
make compare NAME=npr_detector

# Generates report in: reports/compare_npr_detector_YYYYMMDD_HHMMSS.md
```

### Step 6: Review Results

```bash
# View comparison report
cat reports/compare_npr_detector_*.md

# View detailed stats
make stats
```

### Step 7: Integrate (If Better)

```bash
# If model performs better, integrate it
make integrate NAME=npr_detector

# This copies to: ../detectors/npr_detector.py
# Then manually add to orchestrator
```

## Comparing Multiple Models

```bash
# Test multiple candidates
make test-candidate NAME=model_a
make test-candidate NAME=model_b
make test-candidate NAME=model_c

# Compare all with baseline
make compare-all

# View comprehensive comparison
cat reports/comparison_all_*.md
```

## Understanding Results

### Key Metrics

- **Accuracy**: Overall correctness (should be ≥70%)
- **Precision**: Accuracy when predicting AI (lower false positives)
- **Recall**: Ability to find all AI images (lower false negatives)
- **F1 Score**: Balanced measure (harmonic mean of precision/recall)
- **False Positive Rate**: Real images incorrectly flagged (should be <10%)

### Category Results

- **fake/**: AI-generated images (high recall expected)
- **edited/**: Human-edited images (tests false positive rate)
- **real/**: Authentic photos (should NOT be flagged)

### Example Good Results

```
Overall Performance:
  Accuracy:     87.5%  ✓ Good (≥70%)
  Precision:    90.0%  ✓ Good
  Recall:       85.0%  ✓ Good
  F1 Score:     87.4%  ✓ Good
  FP Rate:      5.0%   ✓ Excellent (<10%)
  
By Category:
  FAKE:    90.0% accuracy  ✓ Catches most AI images
  EDITED:  90.0% accuracy  ✓ Doesn't over-flag edits
  REAL:    95.0% accuracy  ✓ Rarely flags real photos
```

### Example Poor Results (SPAI Current)

```
Overall Performance:
  Accuracy:     50.0%  ✗ Poor (below 70%)
  FP Rate:      0.0%   ⚠ Too conservative
  
By Category:
  FAKE:    0.0% accuracy   ✗ Misses all AI images
  EDITED:  100% accuracy   ✓ Good on edited
  REAL:    100% accuracy   ✓ Good on real
```

## Common Workflows

### Just Want to Test One Image

```bash
cd candidates/your_model
python detector.py /path/to/test_image.jpg
```

### Watch for Changes (Development)

```bash
# Auto-retest when detector.py changes
make watch NAME=your_model
```

### Clean Up

```bash
# Remove generated files (keep results)
make clean

# Remove everything (including results)
make clean-all

# Remove specific candidate
make clean-candidate NAME=old_model
```

## Troubleshooting

### "No baseline found"

```bash
# Create baseline first
make baseline
```

### "Test data not found"

```bash
# Link your test images
ln -s /path/to/your/images test_data
make setup-test-data
```

### "Detector import error"

```bash
# Check dependencies
cd candidates/your_model
pip install -r requirements.txt

# Verify
python -c "from detector import YourDetector; print('OK')"
```

### "Model weights not found"

```bash
# Download weights
cd candidates/your_model
# Add download to setup.sh or download manually to weights/
```

## Best Practices

### Test Data

- ✓ Use diverse AI generators (Midjourney, DALL-E, Stable Diffusion, etc.)
- ✓ Include various resolutions and styles
- ✓ Test with post-processed images
- ✓ Include real camera photos with EXIF
- ✗ Don't use synthetic test data
- ✗ Don't use only one AI generator

### Model Selection

- ✓ Test on your specific use case
- ✓ Consider inference speed
- ✓ Check license compatibility
- ✓ Verify active maintenance
- ✗ Don't rely solely on paper claims
- ✗ Don't ignore false positive rates

### Integration

- ✓ Test thoroughly before integration
- ✓ Add to orchestrator with proper ordering
- ✓ Document known limitations
- ✓ Set appropriate confidence thresholds
- ✗ Don't skip unit tests
- ✗ Don't integrate untested models

## Next Steps

1. **Read the full README**: `cat README.md`
2. **Link test data**: `ln -s /path/to/images test_data`
3. **Create baseline**: `make baseline`
4. **Try a new model**: `make new-candidate NAME=test_model`
5. **Review plugin docs**: `cat ../../docs/PLUGIN_DEVELOPMENT.md`

## Need Help?

- Review `README.md` for detailed documentation
- Check `framework/templates/` for implementation examples
- See `PLUGIN_DEVELOPMENT.md` for integration guide
- Review existing detectors in `../detectors/` for patterns
