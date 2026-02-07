# Enhanced Image Forensics - Implementation Guide

## Quick Start: Adding Your First Enhanced Detection Module

### Step 1: Install Additional Dependencies

```bash
# Add to requirements.txt
echo "numpy>=1.21.0" >> requirements.txt
echo "scipy>=1.7.0" >> requirements.txt
echo "opencv-python-headless>=4.5.0" >> requirements.txt
echo "scikit-image>=0.19.0" >> requirements.txt

# Install
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Create Enhanced Forensics Library

```bash
mkdir -p lib/forensics
touch lib/forensics/__init__.py
```

### Step 3: Implement Your First Module - Noise Analysis

**File**: `plugins/processing/noise_analysis.py`

```python
# Ghiro - Copyright (C) 2013-2025 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
import numpy as np
from scipy.ndimage import gaussian_filter

from lib.analyzer.base import BaseProcessingModule
from lib.utils import str2image, image2str
from lib.db import save_file

try:
    from PIL import Image
    IS_PIL = True
except ImportError:
    IS_PIL = False

logger = logging.getLogger(__name__)


class NoiseAnalysisProcessing(BaseProcessingModule):
    """Analyzes noise patterns to detect manipulation."""

    name = "Noise Pattern Analysis"
    description = "Extracts and analyzes noise patterns to detect inconsistencies indicating manipulation or AI generation."
    order = 25

    def check_deps(self):
        return IS_PIL

    def extract_noise(self, image_array):
        """Extract noise pattern using high-pass filter."""
        # Apply Gaussian blur to get low-frequency content
        denoised = gaussian_filter(image_array.astype(float), sigma=2)
        # Subtract to get high-frequency noise
        noise = image_array.astype(float) - denoised
        return noise

    def analyze_local_variance(self, noise, block_size=32):
        """Calculate noise variance in local blocks."""
        height, width = noise.shape[:2]
        variances = []
        positions = []
        
        for y in range(0, height - block_size, block_size // 2):
            for x in range(0, width - block_size, block_size // 2):
                block = noise[y:y+block_size, x:x+block_size]
                if block.shape[0] == block_size and block.shape[1] == block_size:
                    var = np.var(block)
                    variances.append(var)
                    positions.append((x, y))
        
        return np.array(variances), positions

    def detect_inconsistencies(self, variances):
        """Detect anomalous variance regions."""
        if len(variances) == 0:
            return 0, []
        
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        
        # Flag blocks that deviate significantly
        threshold = mean_var + 2 * std_var
        anomalies = np.where(variances > threshold)[0]
        
        # Calculate inconsistency score
        inconsistency_score = len(anomalies) / len(variances) * 100
        
        return inconsistency_score, anomalies

    def create_variance_map(self, variances, positions, image_shape):
        """Create heatmap visualization of noise variance."""
        height, width = image_shape[:2]
        variance_map = np.zeros((height, width))
        block_size = 32
        
        for i, (x, y) in enumerate(positions):
            variance_map[y:y+block_size, x:x+block_size] = variances[i]
        
        # Normalize to 0-255
        if variance_map.max() > 0:
            variance_map = (variance_map / variance_map.max() * 255).astype(np.uint8)
        
        return variance_map

    def run(self, task):
        try:
            # Load image
            pil_image = str2image(task.get_file_data)
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Handle grayscale
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            
            # Work with luminance channel for consistency
            if image_array.shape[2] >= 3:
                # Convert to YCbCr and use Y channel
                luminance = 0.299 * image_array[:,:,0] + 0.587 * image_array[:,:,1] + 0.114 * image_array[:,:,2]
            else:
                luminance = image_array[:,:,0]
            
            # Extract noise
            noise = self.extract_noise(luminance)
            
            # Analyze local variance
            variances, positions = self.analyze_local_variance(noise)
            
            # Detect inconsistencies
            inconsistency_score, anomalies = self.detect_inconsistencies(variances)
            
            # Store results
            self.results["noise_analysis"]["inconsistency_score"] = float(inconsistency_score)
            self.results["noise_analysis"]["suspicious"] = inconsistency_score > 15.0
            self.results["noise_analysis"]["mean_variance"] = float(np.mean(variances)) if len(variances) > 0 else 0
            self.results["noise_analysis"]["std_variance"] = float(np.std(variances)) if len(variances) > 0 else 0
            self.results["noise_analysis"]["anomaly_count"] = len(anomalies)
            
            # Create and save variance map visualization
            variance_map = self.create_variance_map(variances, positions, image_array.shape)
            
            # Convert to RGB for visualization (using a heatmap colormap)
            from PIL import ImageOps
            variance_image = Image.fromarray(variance_map, mode='L')
            variance_image = ImageOps.colorize(variance_image, 
                                              black='blue', 
                                              white='red',
                                              mid='green')
            
            # Resize if too large
            width, height = variance_image.size
            if width > 1800:
                variance_image.thumbnail([1800, 1800], Image.Resampling.LANCZOS)
            
            # Save variance map
            img_str = image2str(variance_image)
            self.results["noise_analysis"]["variance_map_id"] = save_file(img_str, content_type="image/jpeg")
            
            logger.info(f"[Task {task.id}]: Noise analysis complete - Inconsistency: {inconsistency_score:.2f}%")
            
        except Exception as e:
            logger.exception(f"[Task {task.id}]: Error in noise analysis: {e}")
        
        return self.results
```

### Step 4: Test Your Module

```bash
# Start the development server
make dev

# Upload a test image through the web interface
# Check the analysis results
```

### Step 5: Add Template for Visualization

**File**: `templates/analyses/report/_noise_analysis.html`

```html
{% load analyses_tags %}

<div class="box">
    <div class="wdgt-header">Noise Pattern Analysis
        <span class="pull-right">
            {% if analysis.report.noise_analysis.suspicious %}
                <span class="label label-warning">Suspicious</span>
            {% else %}
                <span class="label label-success">Normal</span>
            {% endif %}
        </span>
    </div>
    <div class="wdgt-body">
        {% if analysis.report.noise_analysis.variance_map_id %}
            <div class="row">
                <div class="col-md-6">
                    <h4>Variance Heatmap</h4>
                    <img src="{% url "analyses.views.image" analysis.report.noise_analysis.variance_map_id %}" 
                         class="img-responsive" 
                         alt="Noise Variance Map" />
                    <p class="help-block">
                        Blue = Low variance (smooth/synthetic)<br>
                        Green = Normal variance<br>
                        Red = High variance (potential manipulation)
                    </p>
                </div>
                <div class="col-md-6">
                    <h4>Analysis Results</h4>
                    <table class="table table-striped">
                        <tr>
                            <td><strong>Inconsistency Score:</strong></td>
                            <td>
                                <span class="label {% if analysis.report.noise_analysis.inconsistency_score > 15 %}label-warning{% else %}label-success{% endif %}">
                                    {{ analysis.report.noise_analysis.inconsistency_score|floatformat:2 }}%
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Mean Variance:</strong></td>
                            <td>{{ analysis.report.noise_analysis.mean_variance|floatformat:4 }}</td>
                        </tr>
                        <tr>
                            <td><strong>Std Deviation:</strong></td>
                            <td>{{ analysis.report.noise_analysis.std_variance|floatformat:4 }}</td>
                        </tr>
                        <tr>
                            <td><strong>Anomalous Regions:</strong></td>
                            <td>
                                <span class="badge">{{ analysis.report.noise_analysis.anomaly_count }}</span>
                            </td>
                        </tr>
                    </table>
                    
                    <div class="alert {% if analysis.report.noise_analysis.suspicious %}alert-warning{% else %}alert-info{% endif %}">
                        <strong>Interpretation:</strong>
                        {% if analysis.report.noise_analysis.suspicious %}
                            This image shows significant noise inconsistencies. This may indicate:
                            <ul>
                                <li>Regions from different sources (splicing)</li>
                                <li>AI-generated content (synthetic noise)</li>
                                <li>Heavy editing with selective smoothing</li>
                            </ul>
                        {% else %}
                            Noise patterns appear consistent across the image, suggesting it may be unedited or from a single source.
                        {% endif %}
                    </div>
                </div>
            </div>
        {% else %}
            <p class="text-muted">Noise analysis not available for this image type.</p>
        {% endif %}
    </div>
</div>
```

### Step 6: Integrate into Report View

**Edit**: `templates/analyses/report/static_report.html`

Add after the ELA section:

```html
    {% if analysis.report.noise_analysis %}
        <tr>
            <td style="padding-top:30px">
                <h2>Noise Pattern Analysis</h2>
            </td>
        </tr>
        <tr>
            <td>
                {% include 'analyses/report/_noise_analysis.html' %}
            </td>
        </tr>
    {% endif %}
```

### Step 7: Add to Main Report Navigation

**Edit**: `templates/analyses/report/show.html` (if exists) or similar view file

Add tab for noise analysis alongside ELA, EXIF, etc.

---

## Next Steps: Additional Modules

### Module 2: Frequency Analysis

**File**: `plugins/processing/frequency_analysis.py`

```python
import numpy as np
from scipy.fft import fft2, fftshift

class FrequencyAnalysisProcessing(BaseProcessingModule):
    """Analyzes frequency domain for manipulation artifacts."""
    
    name = "Frequency Domain Analysis"
    description = "Uses FFT to detect JPEG ghosts and periodic artifacts."
    order = 26
    
    def run(self, task):
        # Load image as grayscale
        # Compute 2D FFT
        # Create magnitude spectrum visualization
        # Detect periodic patterns
        # Store results
        pass
```

### Module 3: AI Artifact Detection

**File**: `plugins/processing/ai_artifact_detection.py`

```python
class AIArtifactDetection(BaseProcessingModule):
    """Detects artifacts specific to AI-generated images."""
    
    name = "AI Generation Detection"
    description = "Identifies characteristics of AI-generated content."
    order = 30
    
    def run(self, task):
        # Check metadata for AI signatures
        # Analyze gradient smoothness
        # Detect checkerboard patterns
        # Calculate AI probability score
        pass
```

---

## Testing Your Enhancements

### Create Test Cases

**File**: `tests/test_enhanced_forensics.py`

```python
import unittest
from analyses.models import Analysis, Case
from django.contrib.auth import get_user_model

class EnhancedForensicsTestCase(unittest.TestCase):
    
    def setUp(self):
        self.user = get_user_model().objects.create_user('test', 'test@test.com', 'test')
        self.case = Case.objects.create(name='Test Case', owner=self.user)
    
    def test_noise_analysis_on_real_photo(self):
        """Test noise analysis on a real photograph."""
        # Load test image
        # Run analysis
        # Assert expected results
        pass
    
    def test_noise_analysis_on_ai_image(self):
        """Test noise analysis on AI-generated image."""
        # Load AI-generated test image
        # Run analysis
        # Assert suspicious flag is raised
        pass
```

### Run Tests

```bash
python manage.py test tests.test_enhanced_forensics
```

---

## Performance Optimization Tips

### 1. Use Numpy Efficiently
```python
# Bad: Loop over pixels
for y in range(height):
    for x in range(width):
        pixel = image[y, x]

# Good: Vectorized operations
result = np.mean(image, axis=2)
```

### 2. Cache Expensive Computations
```python
# Store FFT results for reuse
if 'fft_cache' not in self.data:
    self.data['fft_cache'] = fft2(image)
```

### 3. Progressive Analysis
```python
# Quick check first
if not quick_metadata_check(task):
    return self.results  # Skip expensive analysis

# Only run full analysis if needed
full_analysis(task)
```

---

## Deployment Checklist

- [ ] All dependencies in requirements.txt
- [ ] Modules tested with sample images
- [ ] Templates created for visualizations
- [ ] Error handling implemented
- [ ] Logging added for debugging
- [ ] Documentation updated
- [ ] Performance acceptable (<30s per image)
- [ ] Memory usage reasonable (<1GB per analysis)

---

## Common Issues & Solutions

### Issue: "Module not found"
**Solution**: Ensure plugin is in `plugins/processing/` and has correct structure.

### Issue: "Out of memory"
**Solution**: Downsample large images before analysis:
```python
if width > 2000 or height > 2000:
    pil_image.thumbnail([2000, 2000], Image.Resampling.LANCZOS)
```

### Issue: "Analysis too slow"
**Solution**: Use multiprocessing or implement caching:
```python
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(analyze_block, blocks)
```

---

## Resources

- **NumPy Documentation**: https://numpy.org/doc/
- **SciPy Signal Processing**: https://docs.scipy.org/doc/scipy/reference/signal.html
- **OpenCV Python**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **Scikit-Image**: https://scikit-image.org/docs/stable/

---

## Getting Help

1. Check existing plugin code in `plugins/processing/`
2. Review base class in `lib/analyzer/base.py`
3. Test with simple images first
4. Add logging to debug issues
5. Check Django logs in `logs/`

---

## Success Metrics

Your enhancement is successful when:
- ✓ Module loads without errors
- ✓ Analysis completes in reasonable time
- ✓ Results are stored in database
- ✓ Visualization renders correctly
- ✓ Detection accuracy is validated
- ✓ No performance degradation

---

Good luck with your enhancements! Start with the noise analysis module and expand from there. Each module builds on the same patterns, making it easier as you go.
