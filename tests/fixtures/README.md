# Test Fixtures

This directory contains test data fixtures for validating the AI detection system.

## forensics_test_data.json

Real forensics data extracted from MongoDB analysis results (image IDs 54-70) used to validate the compliance audit detector.

### Dataset Composition

- **5 Real Photos (unedited)**: IDs 54, 55, 56, 63, 64
  - Natural camera photos with authentic noise patterns
  - Expected: 100% should pass as authentic

- **1 Real Photo (heavily edited)**: ID 62
  - Stock photo with heavy post-processing
  - Very low noise inconsistency (1.46) from editing
  - Expected: Correctly flagged as fake (manipulation detected)

- **11 AI-Generated Images**: IDs 57-61, 65-70
  - Various AI models and generation techniques
  - Some with post-processing to add realistic noise
  - Expected: ~73% detection rate (8/11)
  - 3 sophisticated AI (58, 66, 69) may pass due to high noise inconsistency

### Key Forensics Metrics

Each image includes:

- **noise_analysis**: Block variance inconsistency analysis
  - `inconsistency_score`: How uniform the noise is (lower = more synthetic)
  - `anomaly_count`: Number of blocks with outlier variance
  - `mean_variance`, `std_variance`: Statistical measures

- **frequency_analysis**: FFT-based checkerboard detection
  - Note: Found to be ineffective (flags everything)
  - Included for completeness but not used in scoring

- **dimensions**: Image width and height
  - Perfect squares (512x512, 1024x1024, etc.) indicate AI generation

### Detection Thresholds

Based on statistical analysis of real vs AI patterns:

**Noise Inconsistency:**
- `< 3.0`: HIGH risk (synthetic uniform noise) → -70 points
- `< 4.2`: MEDIUM risk (suspicious uniformity) → -40 points
- `> 5.5`: POSITIVE (natural sensor variation) → +35 points

**Dimensions:**
- Perfect squares (1024x1024, etc.): → -60 points

**Anomaly Count:**
- `< 100`: Too uniform → -10 points
- `> 500`: Natural variation → +15 points

### Expected Performance

Based on validation against this dataset:

| Category | Count | Accuracy |
|----------|-------|----------|
| Real photos (unedited) | 5 | 100% |
| Real photos (edited) | 1 | 100% (correctly flagged) |
| AI images | 11 | 73% (8/11) |
| **Overall** | **17** | **82%** |

### Usage

The integration test suite (`test_forensics_integration.py`) uses this data to:

1. Validate detection accuracy against real-world cases
2. Document scoring behavior with actual examples
3. Ensure thresholds remain calibrated
4. Provide transparency on why images are classified

Run tests with:
```bash
python -m unittest tests.test_forensics_integration -v
```

This will show detailed output including:
- Score for each image
- Which detectors triggered
- Why images were classified as fake or real
- Overall performance statistics
