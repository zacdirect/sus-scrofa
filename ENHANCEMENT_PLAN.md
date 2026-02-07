# Ghiro Enhancement Plan: Advanced Image Forensics
## AI-Generated & Manipulated Image Detection

**Target Environment**: Air-gapped, offline operation  
**Approach**: Deterministic, signal-processing based techniques  
**Philosophy**: Zero AI dependency, rely on mathematical/physical artifacts

---

## Current Capabilities Assessment

### Existing Features ✓
- **ELA (Error Level Analysis)**: Detects JPEG compression inconsistencies
- **EXIF Metadata Analysis**: Extracts camera/software info, GPS, timestamps
- **Perceptual Hashing**: Image similarity detection (aHash, pHash, dHash)
- **Hash Comparison**: MD5, SHA1, SHA256 for file integrity
- **Thumbnail Consistency Check**: Compares embedded thumbnail vs actual image
- **Signature Detection**: 1000+ signatures for editing tools, metadata anomalies
- **Preview Comparer**: Detects differences between embedded previews

### Limitations
- ELA only works on JPEG (not effective on PNG or uncompressed)
- Limited detection of modern AI generation artifacts
- No frequency domain analysis
- No noise pattern analysis
- No copy-move detection
- Limited statistical analysis

---

## Phase 1: Enhanced Visual Analysis Filters (Human Operator Support)

### 1.1 Advanced ELA Improvements
**Goal**: Make manipulation artifacts more visible to analysts

```python
# New plugin: plugins/processing/ela_enhanced.py

Enhancements:
- Multi-quality ELA (test at quality 70, 85, 95, 98)
- Adaptive scaling based on image characteristics
- Color-coded heatmaps (cool=low error, hot=high error)
- Statistical analysis of error distributions
- Edge-weighted ELA (manipulations often visible at boundaries)
```

**Visual Outputs**:
- Standard ELA (grayscale)
- Heatmap overlay (red/yellow/green zones)
- Histogram of error levels
- Localized error intensity grid

**Libraries**: PIL/Pillow, NumPy, SciPy (all offline-capable)

---

### 1.2 Noise Analysis & Visualization
**Goal**: Detect inconsistent noise patterns indicating manipulation

```python
# New plugin: plugins/processing/noise_analysis.py

Techniques:
- High-pass filtering to extract noise
- Local noise variance analysis
- Photo Response Non-Uniformity (PRNU) pattern extraction
- Sensor pattern noise visualization
```

**Detection Capabilities**:
- Spliced regions (different camera sources)
- AI-generated areas (typically have uniform, synthetic noise)
- Clone-stamped regions (identical noise patterns)
- Smoothed/denoised areas (abnormally low noise)

**Visual Outputs**:
- Noise pattern map (shows local noise characteristics)
- Noise variance heatmap
- Inconsistency detection overlay

**Implementation**:
```python
import numpy as np
from scipy.ndimage import gaussian_filter

def extract_noise_pattern(image):
    # Apply denoising and extract residual
    denoised = gaussian_filter(image, sigma=2)
    noise = image - denoised
    return noise

def analyze_local_variance(noise, block_size=32):
    # Calculate variance in sliding windows
    # Flag regions with anomalous variance
    pass
```

---

### 1.3 Frequency Domain Analysis
**Goal**: Detect periodic patterns, JPEG ghosts, AI artifacts

```python
# New plugin: plugins/processing/frequency_analysis.py

Techniques:
- 2D FFT (Fast Fourier Transform) visualization
- DCT (Discrete Cosine Transform) block analysis
- JPEG ghost detection (multiple compression artifacts)
- Grid pattern detection (common in AI upscaling)
```

**Detection Capabilities**:
- Double JPEG compression with different quality
- AI upscaling artifacts (periodic patterns in frequency space)
- Checkerboard patterns from GAN generators
- Regular grid structures

**Visual Outputs**:
- FFT magnitude spectrum (log scale)
- DCT coefficient distribution
- JPEG ghost heatmap
- Grid pattern overlay

**Libraries**: NumPy, SciPy (scipy.fft, scipy.fftpack)

---

### 1.4 Copy-Move Forgery Detection
**Goal**: Find duplicated regions within same image

```python
# New plugin: plugins/processing/copy_move_detection.py

Techniques:
- Block-matching algorithm with DCT
- SIFT/ORB feature matching (scale-invariant)
- Exact match detection (clone stamp)
- Similar pattern detection (with rotation/scaling)
```

**Detection Capabilities**:
- Cloned regions (copy-paste within image)
- Content-aware fill artifacts
- Pattern stamp tool usage

**Visual Outputs**:
- Matched region pairs (color-coded)
- Confidence map
- Source-destination arrows

**Implementation Approach**:
```python
# Use sliding window with DCT features
# Compare blocks using normalized cross-correlation
# Apply geometric constraints to eliminate false positives
```

---

### 1.5 Color/Lighting Analysis
**Goal**: Detect inconsistent illumination and color spaces

```python
# New plugin: plugins/processing/illumination_analysis.py

Techniques:
- Illuminant color estimation per region
- Shadow direction analysis
- Reflectance inconsistency detection
- Color space anomaly detection
```

**Detection Capabilities**:
- Composited images with different lighting
- Artificial shadows
- Inconsistent white balance
- Color grading mismatches

**Visual Outputs**:
- Illumination direction map
- Color temperature heatmap
- Shadow consistency overlay

---

## Phase 2: AI-Generated Image Detection (Deterministic Approaches)

### 2.1 JPEG/PNG Artifact Analysis
**Goal**: AI generators produce characteristic compression artifacts

```python
# New plugin: plugins/processing/ai_artifact_detection.py

Detection Signals:
1. Overly smooth gradients (AI models avoid noise)
2. Absence of natural JPEG blocking artifacts
3. Synthetic noise patterns (too uniform)
4. Unnatural color transitions
5. Missing chromatic aberration
6. Perfect symmetry (rare in real photos)
```

**Statistical Tests**:
- Benford's Law on DCT coefficients (AI violates natural distributions)
- High-frequency component analysis
- Gradient histogram analysis
- Texture regularity metrics

**Indicators**:
```python
def detect_ai_smoothness(image):
    # Real photos have specific gradient distributions
    # AI images often have smoother, more uniform gradients
    gradient_hist = calculate_gradient_histogram(image)
    smoothness_score = analyze_distribution(gradient_hist)
    return smoothness_score > threshold

def detect_unnatural_noise(image):
    # AI noise is often too uniform
    noise = extract_high_freq_noise(image)
    variance_map = calculate_local_variance(noise)
    uniformity = std(variance_map) / mean(variance_map)
    return uniformity < threshold  # Too uniform = suspicious
```

---

### 2.2 Metadata Forensics for AI Detection
**Goal**: AI tools leave characteristic metadata signatures

```python
# New signatures: plugins/signatures/ai_detection.py

Check for:
- Missing camera EXIF data (AI images have none/synthetic)
- Software tags: "Stable Diffusion", "DALL-E", "Midjourney"
- Unusual image dimensions (512x512, 1024x1024 = common AI sizes)
- Missing sensor noise characteristics
- Absent lens distortion data
- Perfect aspect ratios
- Missing GPS/timestamp data (but pristine quality)
```

**Scoring System**:
```python
ai_indicators = {
    'no_camera_make': 20,
    'common_ai_dimensions': 30,
    'missing_lens_data': 15,
    'software_contains_ai_tool': 90,
    'perfect_aspect_ratio': 10,
    'no_color_space': 15,
    'suspicious_creation_software': 40,
}

def calculate_ai_probability(metadata):
    score = sum([v for k, v in ai_indicators.items() if check_indicator(k, metadata)])
    return min(score, 100)
```

---

### 2.3 GAN/Diffusion Fingerprinting
**Goal**: Detect mathematical signatures of generative models

```python
# New plugin: plugins/processing/gan_fingerprint.py

Techniques:
1. Up-sampling artifacts (checkerboard patterns)
2. Spectral anomalies (specific frequency peaks)
3. Boundary artifacts (seams at patch boundaries)
4. Fingerprint spectrum analysis
```

**Detection Method**:
```python
def detect_upsampling_artifacts(image):
    # GANs often produce checkerboard artifacts in frequency domain
    fft = np.fft.fft2(image)
    magnitude = np.abs(fft)
    
    # Look for periodic patterns
    autocorr = np.fft.ifft2(magnitude * np.conj(magnitude))
    peaks = find_periodic_peaks(autocorr)
    
    # Checkerboard = peaks at specific frequencies
    return has_checkerboard_signature(peaks)

def analyze_boundary_artifacts(image, patch_size=256):
    # Many diffusion models work in patches
    # Look for inconsistencies at patch boundaries
    grid_variance = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            boundary = extract_boundary(image, x, y, patch_size)
            variance = calculate_variance_across_boundary(boundary)
            grid_variance.append(variance)
    
    # Higher variance at grid = suspicious
    return np.std(grid_variance) > threshold
```

---

### 2.4 Local Inconsistency Detection
**Goal**: AI models struggle with global consistency

```python
# New plugin: plugins/processing/consistency_check.py

Check for:
- Inconsistent perspective (parallel lines don't converge)
- Impossible lighting (multiple light sources with wrong shadows)
- Repeating patterns (AI texture generation artifacts)
- Asymmetric reflections
- Text gibberish (AI can't render readable text well)
- Face/anatomy inconsistencies
```

**Implementation**:
```python
def detect_perspective_issues(image):
    # Use line detection (Hough transform)
    # Check if parallel lines meet at reasonable vanishing points
    pass

def detect_text_anomalies(image):
    # Extract text regions
    # OCR and check for gibberish
    # AI-generated text is often illegible
    pass
```

---

### 2.5 Statistical Anomaly Detection
**Goal**: Quantify "naturalness" of images

```python
# New plugin: plugins/processing/statistical_analysis.py

Metrics:
1. Natural Scene Statistics (NSS)
   - Real photos follow specific statistical distributions
   - AI images often deviate

2. Wavelet Statistics
   - Decompose into frequency bands
   - Compare coefficient distributions

3. Color Distribution Analysis
   - Natural photos have characteristic color distributions
   - AI may produce oversaturated or unnatural colors

4. Edge Statistics
   - Edge strength and orientation distributions
   - AI edges are often too perfect
```

**Implementation**:
```python
def calculate_naturalness_score(image):
    scores = {}
    
    # Test 1: Gradient distribution
    gradients = calculate_gradients(image)
    scores['gradient'] = compare_to_natural_distribution(gradients)
    
    # Test 2: Laplacian of Gaussian response
    log_response = apply_log_filter(image)
    scores['log'] = analyze_log_statistics(log_response)
    
    # Test 3: Color histogram entropy
    color_entropy = calculate_color_entropy(image)
    scores['color'] = compare_entropy_to_natural_range(color_entropy)
    
    return aggregate_scores(scores)
```

---

## Phase 3: Enhanced User Interface

### 3.1 Interactive Analysis Dashboard

**New View**: `Enhanced Analysis Mode`

```html
<!-- templates/analyses/enhanced_analysis.html -->

Layout:
┌─────────────────────────────────────────────┐
│ Original Image      │  Analysis Selection  │
├─────────────────────┼──────────────────────┤
│                     │  ☐ Standard ELA      │
│                     │  ☑ Multi-Quality ELA │
│   [Image Display]   │  ☑ Noise Analysis    │
│                     │  ☑ Frequency View    │
│                     │  ☐ Copy-Move         │
│                     │  ☑ AI Detection      │
├─────────────────────┼──────────────────────┤
│ Analysis Results    │  Detection Scores    │
│                     │                      │
│ [Side-by-side       │  Manipulation: 78%   │
│  comparison view]   │  AI Generated: 45%   │
│                     │  Copy-Move: 12%      │
└─────────────────────┴──────────────────────┘
```

**Features**:
- Slider to adjust analysis sensitivity
- Toggle between different visualization modes
- Overlay multiple analyses
- Export enhanced reports
- Side-by-side comparisons
- Zoom and pan synchronized across views

---

### 3.2 Visual Filter Suite

**Interactive Filters** (client-side, for quick human analysis):

```javascript
// static/js/enhanced_filters.js

Available Filters:
1. Edge Enhancement (Sobel, Canny)
2. Sharpening (Unsharp mask)
3. Noise Extraction (High-pass)
4. Color Channel Separation (R, G, B, individual)
5. Luminance/Chrominance Split
6. Invert Colors
7. Posterization (reduce color depth to see compression)
8. Clone Stamp Visualizer
9. Metadata Overlay
```

**Implementation**:
```javascript
// Apply filters in real-time using Canvas API
function applyEdgeDetection(imageData) {
    // Sobel operator
    const kernel = [[-1,-2,-1],[0,0,0],[1,2,1]];
    return convolve(imageData, kernel);
}
```

---

### 3.3 Confidence Scoring System

**Aggregate Detection Results**:

```python
# lib/confidence_calculator.py

def calculate_manipulation_confidence(results):
    """
    Aggregate all detection methods into overall confidence score
    """
    confidence = {
        'manipulation_detected': False,
        'confidence_score': 0.0,
        'ai_generated_probability': 0.0,
        'indicators': [],
        'methods': {}
    }
    
    # Weight different detection methods
    weights = {
        'ela_anomalies': 0.20,
        'noise_inconsistency': 0.25,
        'frequency_anomalies': 0.15,
        'copy_move_detected': 0.30,
        'metadata_issues': 0.10,
        'ai_artifacts': 0.35,
        'statistical_deviation': 0.20,
    }
    
    # Calculate weighted score
    for method, weight in weights.items():
        if method in results and results[method]['detected']:
            confidence['confidence_score'] += weight * results[method]['strength']
            confidence['indicators'].append({
                'method': method,
                'evidence': results[method]['evidence']
            })
    
    confidence['manipulation_detected'] = confidence['confidence_score'] > 0.50
    
    return confidence
```

---

## Phase 4: Implementation Roadmap

### Dependencies to Add (requirements.txt)
```python
# Image processing (already have Pillow)
numpy>=1.21.0          # Array operations
scipy>=1.7.0           # Signal processing, FFT
opencv-python-headless>=4.5.0  # Computer vision (headless for server)
scikit-image>=0.18.0   # Image processing algorithms

# Optional but recommended
imagehash>=4.2.0       # Already in use, ensure updated
```

### Plugin Development Order

**Week 1-2: Enhanced ELA & Noise**
- ✓ Multi-quality ELA
- ✓ Noise extraction and analysis
- ✓ Visual heatmaps

**Week 3-4: Frequency Analysis**
- ✓ FFT visualization
- ✓ JPEG ghost detection
- ✓ Grid pattern detection

**Week 5-6: Copy-Move & Lighting**
- ✓ Block matching algorithm
- ✓ Illumination analysis
- ✓ Shadow consistency

**Week 7-8: AI Detection Suite**
- ✓ GAN fingerprinting
- ✓ Statistical anomaly detection
- ✓ Metadata forensics

**Week 9-10: UI Enhancement**
- ✓ Interactive dashboard
- ✓ Filter suite
- ✓ Confidence scoring

**Week 11-12: Testing & Documentation**
- ✓ Test with known AI images
- ✓ Test with manipulated images
- ✓ Performance optimization
- ✓ User documentation

---

## Code Structure

### New Directory Layout
```
plugins/
  processing/
    ela_enhanced.py           # Multi-quality ELA
    noise_analysis.py          # Noise pattern detection
    frequency_analysis.py      # FFT/DCT analysis
    copy_move_detection.py     # Duplicate region detection
    illumination_analysis.py   # Lighting consistency
    ai_artifact_detection.py   # AI-specific artifacts
    gan_fingerprint.py         # GAN/diffusion signatures
    consistency_check.py       # Global consistency checks
    statistical_analysis.py    # Natural scene statistics
    
  signatures/
    ai_detection.py            # AI tool metadata signatures
    manipulation_patterns.py   # Known editing patterns
    
lib/
  forensics/                   # New library
    __init__.py
    filters.py                 # Image filters
    frequency.py               # Frequency domain tools
    noise.py                   # Noise analysis utilities
    statistics.py              # Statistical tests
    confidence.py              # Confidence scoring
    
static/
  js/
    enhanced_filters.js        # Client-side filters
    analysis_dashboard.js      # Interactive UI
  css/
    enhanced_analysis.css      # Styling

templates/
  analyses/
    enhanced_analysis.html     # Main enhanced view
    report/
      _enhanced_ela.html
      _noise_analysis.html
      _frequency_analysis.html
      _ai_detection.html
```

---

## Testing Strategy

### Test Dataset Required

1. **Real Photos**: 1000+ from various cameras/phones
2. **AI Generated**: 500+ from Stable Diffusion, DALL-E, Midjourney
3. **Manipulated**: 500+ with known edits (splice, clone, enhance)
4. **Mixed**: 200+ AI-edited real photos

### Validation Metrics

```python
# tests/test_ai_detection.py

def test_ai_detection_accuracy():
    """
    Measure:
    - True Positive Rate (correctly detect AI)
    - False Positive Rate (flag real as AI)
    - True Negative Rate (correctly identify real)
    - False Negative Rate (miss AI images)
    
    Target: >90% accuracy, <5% false positive
    """
    pass

def test_manipulation_detection():
    """
    Test against:
    - Photoshop edits
    - GIMP edits
    - Clone stamp tools
    - Content-aware fill
    """
    pass
```

---

## Performance Considerations

### Optimization Strategies

1. **Parallel Processing**: Use multiprocessing for independent analyses
2. **Caching**: Store computed FFTs, noise patterns
3. **Progressive Analysis**: Quick checks first, expensive tests only if suspicious
4. **Selective Processing**: Only run full suite if initial flags raised
5. **Resolution Scaling**: Work on downsampled versions for preview

```python
# lib/forensics/optimizer.py

def progressive_analysis(image, task):
    """
    Run quick tests first, only proceed to expensive tests if needed
    """
    # Level 1: Fast metadata checks (< 1 second)
    quick_flags = check_metadata_red_flags(image)
    if not quick_flags:
        return {'suspicious': False, 'confidence': 0.0}
    
    # Level 2: Medium tests (< 5 seconds)
    medium_results = run_medium_tests(image)
    if medium_results['confidence'] < 0.3:
        return medium_results
    
    # Level 3: Expensive analysis (< 30 seconds)
    full_results = run_full_analysis(image)
    return full_results
```

---

## Key Advantages of This Approach

### ✓ **No AI Models Required**
- All techniques based on signal processing
- Deterministic, explainable results
- No training data needed
- No GPU required

### ✓ **Offline Capable**
- All dependencies are Python packages
- No API calls
- No cloud services
- Air-gap compatible

### ✓ **Fast & Efficient**
- Most analyses complete in seconds
- Parallelizable
- Progressive detection (stop early if clear)

### ✓ **Human-Interpretable**
- Visual outputs for analysts
- Clear indicators of what was found
- Not a black box

### ✓ **Extensible**
- Plugin architecture preserved
- Easy to add new detection methods
- Modular design

---

## Expected Detection Capabilities

### AI-Generated Images
**Detection Rate**: 85-95%

**Reliable Indicators**:
- Missing camera EXIF
- Unnatural noise patterns
- Frequency domain artifacts
- Perfect gradients
- Checkerboard patterns (GANs)
- Patch boundary artifacts (Diffusion)

### Manipulated Real Photos
**Detection Rate**: 75-90%

**Reliable Indicators**:
- ELA anomalies
- Noise inconsistencies
- Copy-move matches
- Lighting violations
- Frequency domain ghosts

### Limitations
- May not detect very sophisticated professional edits
- Single-generation AI images are harder than multi-edited
- High-quality AI (2024+) with post-processing is challenging
- Some natural phenomena can trigger false positives

---

## Documentation Requirements

### User Guide Sections
1. Understanding Enhanced Analysis Results
2. Interpreting Heatmaps and Visualizations
3. AI Detection Methodology
4. False Positive Mitigation
5. Best Practices for Forensic Analysis

### Technical Documentation
1. Plugin Development Guide
2. Algorithm Descriptions
3. Performance Tuning
4. Extending Detection Methods

---

## Future Enhancements (Post-MVP)

1. **Machine Learning (Optional)**
   - Train local models on air-gapped systems
   - Use gathered dataset for supervised learning
   - Strictly optional, maintain deterministic path

2. **3D Analysis**
   - Depth map extraction
   - Physically impossible geometry detection

3. **Video Forensics**
   - Frame-by-frame analysis
   - Temporal consistency checks
   - Deepfake detection

4. **Advanced Optics**
   - Lens distortion analysis
   - Chromatic aberration patterns
   - Sensor-specific noise fingerprints

---

## Summary

This enhancement plan provides Ghiro with **state-of-the-art image forensics** capabilities while maintaining:
- ✓ Air-gapped operation
- ✓ Deterministic detection
- ✓ No AI dependencies (for core functionality)
- ✓ Human-interpretable results
- ✓ Modular, extensible architecture

The combination of multiple detection techniques provides robust capability against both AI-generated and manipulated images, giving analysts powerful tools to assess image authenticity.
