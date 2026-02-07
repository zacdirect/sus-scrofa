# Ghiro Plugin Development Guide

Complete guide for developing image analysis plugins for Ghiro.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Plugin Types](#plugin-types)
3. [Creating Analysis Plugins](#creating-analysis-plugins)
4. [Adding Visualizations](#adding-visualizations)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### How Analysis Works

1. **User uploads image** → Saved to MongoDB GridFS
2. **Task created** → Added to processing queue
3. **AnalysisManager** → Loads all plugins from `plugins/analyzer/`
4. **AnalysisRunner** → Executes plugins in order
5. **Results stored** → Saved to MongoDB as nested dict
6. **Template renders** → Displays results in web UI

### Key Components

```
ghiro/
├── lib/
│   ├── analyzer/
│   │   ├── base.py              # BaseAnalyzerModule class
│   │   └── processing.py        # AnalysisManager, AnalysisRunner
│   ├── forensics/               # Reusable forensics utilities
│   │   ├── filters.py           # Image filtering
│   │   ├── statistics.py        # Statistical analysis
│   │   └── confidence.py        # Confidence scoring
│   ├── db.py                    # save_results(), save_file()
│   └── utils.py                 # str2image(), image2str(), AutoVivification
├── plugins/
│   └── analyzer/                # ⚠️ ANALYSIS PLUGINS GO HERE
│       ├── ela.py               # Error Level Analysis
│       ├── gexiv.py             # EXIF metadata
│       ├── noise_analysis.py    # Noise pattern detection
│       ├── frequency_analysis.py # FFT analysis
│       ├── ai_detection.py      # AI artifact detection
│       └── confidence_scoring.py # Overall assessment
└── templates/
    └── analyses/
        └── report/
            ├── show.html        # Main report (tab navigation)
            ├── _dashboard.html  # Summary tab
            └── _*.html          # Individual analysis tabs
```

---

## Plugin Types

### Analyzer Plugins (plugins/analyzer/)

**Purpose**: Process uploaded images and generate analysis results  
**Base Class**: `BaseAnalyzerModule`  
**Loading**: Automatically discovered by `AnalysisManager.load_modules()`  
**Execution**: Run sequentially by `order` attribute

### ⚠️ Common Mistake: plugins/processing/

**DO NOT create plugins in `plugins/processing/`** - this directory exists but is not used by the analysis system. Always use `plugins/analyzer/`.

---

## Creating Analysis Plugins

### 1. Basic Plugin Structure

```python
# plugins/analyzer/my_analysis.py

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import AutoVivification

class MyAnalysisModule(BaseAnalyzerModule):
    """Brief description of what this analysis does."""
    
    # Execution order (lower = earlier)
    # Standard plugins: 0-50
    # New plugins: 51+
    order = 55
    
    def check_deps(self):
        """
        Check if required dependencies are available.
        Return True if ready, False if missing dependencies.
        """
        try:
            import required_library
            return True
        except ImportError:
            return False
    
    def run(self, task):
        """
        Main analysis logic.
        
        Args:
            task: Analysis model instance with:
                - task.image_id: GridFS file ID for image
                - task.file_name: Original filename
                
        Returns:
            AutoVivification dict with results
        """
        # Initialize results
        results = AutoVivification()
        
        # Access data from previous plugins
        # self.data contains all results so far
        
        # Your analysis logic here
        results["my_analysis"]["status"] = "completed"
        results["my_analysis"]["score"] = 0.85
        
        return results
```

### 2. Common Plugin Patterns

#### Reading the Image

```python
from lib.utils import str2image
from PIL import Image

def run(self, task):
    results = AutoVivification()
    
    # Load image from GridFS
    image = str2image(task.image_id)
    if not image:
        return results
    
    # Convert to PIL Image if needed
    pil_image = Image.open(image)
    
    # Process image...
    
    return results
```

#### Saving Generated Images

```python
from lib.db import save_file
from lib.utils import image2str
from PIL import Image
import io

def run(self, task):
    results = AutoVivification()
    
    # Create a visualization image
    output_image = Image.new('RGB', (800, 600))
    # ... draw on output_image ...
    
    # Save to GridFS
    image_buffer = io.BytesIO()
    output_image.save(image_buffer, format='PNG')
    image_buffer.seek(0)
    
    image_id = save_file(image_buffer.read())
    results["my_analysis"]["visualization_id"] = image_id
    
    return results
```

#### Using NumPy for Processing

```python
import numpy as np
from lib.forensics.filters import get_luminance

def run(self, task):
    results = AutoVivification()
    
    # Load and convert to numpy array
    image = str2image(task.image_id)
    img_array = np.array(Image.open(image))
    
    # Extract luminance (grayscale)
    gray = get_luminance(img_array)
    
    # Process with numpy
    mean_value = np.mean(gray)
    std_value = np.std(gray)
    
    results["my_analysis"]["mean"] = float(mean_value)
    results["my_analysis"]["std"] = float(std_value)
    
    return results
```

#### Accessing Previous Results

```python
def run(self, task):
    results = AutoVivification()
    
    # Access ELA results from earlier plugin
    if "ela" in self.data:
        ela_max = self.data["ela"].get("max_difference", 0)
        results["my_analysis"]["uses_ela"] = ela_max
    
    # Access metadata if available
    if "metadata" in self.data and "Exif" in self.data["metadata"]:
        camera = self.data["metadata"]["Exif"].get("Model", "Unknown")
        results["my_analysis"]["camera"] = camera
    
    return results
```

### 3. Plugin Order Guidelines

| Order Range | Purpose | Examples |
|-------------|---------|----------|
| 0-10 | Basic metadata extraction | info.py (order=0), mime.py |
| 11-20 | Image analysis prep | hash.py, gexiv.py (EXIF) |
| 21-30 | Core forensics | ela.py (order=20) |
| 31-50 | Advanced analysis | noise_analysis (25), frequency_analysis (26), ai_detection (30) |
| 51-89 | Comparison/matching | hashcomparer.py, previewcomparer.py |
| 90+ | Aggregation/scoring | confidence_scoring (90) - runs last |

**Rule**: Set `order` based on dependencies:
- Need EXIF data? Set order > 15
- Need ELA results? Set order > 20
- Aggregate other analyses? Set order > 85

---

## Adding Visualizations

### 1. Create Template File

```html
<!-- templates/analyses/report/_my_analysis.html -->

<div class="wdgt-box">
    <div class="wdgt-header">
        <h4>My Analysis</h4>
        <span class="pull-right">
            {% if analysis.report.my_analysis.score > 0.7 %}
                <span class="label label-important">Suspicious</span>
            {% else %}
                <span class="label label-success">Normal</span>
            {% endif %}
        </span>
    </div>
    <div class="wdgt-body">
        {% if analysis.report.my_analysis.visualization_id %}
            <div class="row-fluid">
                <div class="span12">
                    <h4>Visualization</h4>
                    <a href="{% url 'image' analysis.report.my_analysis.visualization_id %}" class="fancybox">
                        <img src="{% url 'image' analysis.report.my_analysis.visualization_id %}" 
                             style="max-width: 100%;" 
                             alt="My Analysis" />
                    </a>
                </div>
            </div>
        {% endif %}
        
        <div class="row-fluid">
            <table class="table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Score</td>
                        <td>{{ analysis.report.my_analysis.score }}</td>
                    </tr>
                    <!-- Add more metrics -->
                </tbody>
            </table>
        </div>
    </div>
</div>
```

### 2. Add Tab to Main Report

Edit `templates/analyses/report/show.html`:

```html
<!-- Add to tab navigation (~line 110-145) -->
{% if analysis.report.my_analysis %}
<li>
    <a href="#myanalysis" data-toggle="tab">My Analysis</a>
</li>
{% endif %}

<!-- Add to tab content (~line 190-210) -->
{% if analysis.report.my_analysis %}
<div id="myanalysis" class="tab-pane">
    {% include 'analyses/report/_my_analysis.html' %}
</div>
{% endif %}
```

**Position**: Insert after existing tabs but before signatures tab. Check the current order in `show.html`.

### 3. Update Dashboard Summary

Edit `templates/analyses/report/_dashboard.html`:

```html
<tr>
    <td>My Analysis</td>
    <td>
        {% if analysis.report.my_analysis %}
            <span class="label label-success">Completed</span>
        {% else %}
            <span class="label label-inverse">Not Available</span>
        {% endif %}
    </td>
</tr>
```

---

## Common Patterns

### Error Handling

```python
import logging

logger = logging.getLogger(__name__)

def run(self, task):
    results = AutoVivification()
    
    try:
        # Analysis logic
        image = str2image(task.image_id)
        if not image:
            logger.warning("Could not load image for task %s", task.id)
            return results
        
        # Process...
        
    except Exception as e:
        logger.exception("Error in my_analysis for task %s: %s", task.id, e)
        results["my_analysis"]["error"] = str(e)
    
    return results
```

### Conditional Dependencies

```python
class MyAnalysisModule(BaseAnalyzerModule):
    order = 30
    
    def check_deps(self):
        """Check optional dependencies."""
        try:
            import numpy
            import scipy
            self.has_scipy = True
        except ImportError:
            self.has_scipy = False
            logger.warning("SciPy not available, some features disabled")
        
        # Return True to allow plugin to load with reduced functionality
        return True
    
    def run(self, task):
        results = AutoVivification()
        
        if self.has_scipy:
            # Use advanced features
            pass
        else:
            # Use basic analysis only
            pass
        
        return results
```

### Performance Optimization

```python
import numpy as np
from PIL import Image

def run(self, task):
    results = AutoVivification()
    
    # Load image once
    image = str2image(task.image_id)
    pil_image = Image.open(image)
    
    # Resize for faster processing if large
    max_size = 2048
    if max(pil_image.size) > max_size:
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy once
    img_array = np.array(pil_image)
    
    # Run multiple analyses on same array
    results["my_analysis"]["mean"] = float(np.mean(img_array))
    results["my_analysis"]["std"] = float(np.std(img_array))
    
    return results
```

---

## Troubleshooting

### Plugin Not Loading

**Symptom**: Plugin doesn't appear in analysis results

**Checklist**:
1. ✓ File is in `plugins/analyzer/` (NOT `plugins/processing/`)
2. ✓ Class inherits from `BaseAnalyzerModule`
3. ✓ `check_deps()` returns `True`
4. ✓ File has no syntax errors: `python -m py_compile plugins/analyzer/my_plugin.py`
5. ✓ Restart services: `make stop && make run`
6. ✓ Check logs for import errors

**Debug**:
```bash
# Check if plugin is discovered
grep "Found module" <(make run 2>&1)

# Look for import errors
grep "Unable to import" <(make run 2>&1)
```

### Template Not Showing

**Symptom**: Tab doesn't appear on analysis page

**Checklist**:
1. ✓ Plugin generated results (check MongoDB or debug)
2. ✓ Template file exists: `templates/analyses/report/_my_analysis.html`
3. ✓ Added to `show.html` navigation
4. ✓ Added to `show.html` tab content
5. ✓ Conditional check matches results key: `{% if analysis.report.my_analysis %}`
6. ✓ Clear browser cache / hard refresh (Ctrl+F5)

### Image Not Displaying

**Symptom**: Broken image in template

**Checklist**:
1. ✓ Used `save_file()` to store image
2. ✓ Stored file ID in results: `results["my_analysis"]["image_id"] = file_id`
3. ✓ URL uses correct pattern: `{% url 'image' analysis.report.my_analysis.image_id %}`
4. ✓ Not using old format: `{% url "analyses.views.image" ... %}` ← Wrong!
5. ✓ Image ID is valid (check MongoDB)

### ImportError for BaseProcessingModule

**Error**: `cannot import name 'BaseProcessingModule'`

**Solution**: Change to `BaseAnalyzerModule`:
```python
# WRONG
from lib.analyzer.base import BaseProcessingModule

# CORRECT
from lib.analyzer.base import BaseAnalyzerModule
```

Clear Python cache:
```bash
find plugins/analyzer -name "*.pyc" -delete
find plugins/analyzer -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

### Results Not Persisting

**Symptom**: Plugin runs but results don't show in UI

**Causes**:
1. Not returning `AutoVivification` dict
2. Exception swallowed - check logs
3. Wrong dictionary structure

**Solution**:
```python
from lib.utils import AutoVivification

def run(self, task):
    # WRONG - regular dict
    # results = {}
    
    # CORRECT - AutoVivification
    results = AutoVivification()
    
    results["my_analysis"]["key"] = "value"
    return results  # Must return!
```

---

## Testing Plugins

### Manual Testing

1. **Start services**: `make run`
2. **Upload test image**: http://localhost:8000/analyses/new/
3. **Wait for analysis**: Status changes to "Completed"
4. **Check results**: Click on analysis to view report
5. **Verify tab appears**: Look for your plugin's tab
6. **Check data**: View raw results in MongoDB

### MongoDB Inspection

```bash
# Connect to MongoDB
mongosh mongodb://localhost:27017/ghiro

# Find analysis
db.fs.files.find().limit(5)

# Check results
db.fs.files.findOne({filename: "analysis_results_XXXX"})
```

### Logging

Add debug logging to track execution:

```python
import logging
logger = logging.getLogger(__name__)

def run(self, task):
    logger.info("Starting my_analysis for task %s", task.id)
    results = AutoVivification()
    
    # ... analysis ...
    
    logger.debug("Generated %d results", len(results.get("my_analysis", {})))
    logger.info("Completed my_analysis for task %s", task.id)
    
    return results
```

View logs in terminal where `make run` is running.

---

## Best Practices

### 1. Naming Conventions

- **Plugin file**: `my_analysis.py` (lowercase, underscores)
- **Class name**: `MyAnalysisModule` (PascalCase)
- **Results key**: `my_analysis` (matches filename)
- **Template**: `_my_analysis.html` (leading underscore, matches key)

### 2. Results Structure

```python
results["plugin_name"]["key"] = value
results["plugin_name"]["subkey"]["nested"] = value
results["plugin_name"]["image_id"] = file_id  # For visualizations
```

### 3. Documentation

Always include:
- Module docstring explaining purpose
- `check_deps()` documenting requirements
- Inline comments for complex algorithms
- Sample output structure in docstring

### 4. Dependencies

- Use forensics library: `from lib.forensics import ...`
- Reuse existing utilities when possible
- Document required packages in module docstring
- Handle import errors gracefully

### 5. Performance

- Resize large images before processing
- Use numpy for batch operations
- Avoid repeated file I/O
- Cache expensive computations in `self` if used multiple times

---

## Example: Complete Plugin

See `plugins/analyzer/noise_analysis.py` for a complete, production-ready example showing:
- Dependency checking
- Image loading and processing
- NumPy-based analysis
- Heatmap visualization generation
- Results structuring
- Error handling
- Logging

---

## Need Help?

1. Check existing plugins in `plugins/analyzer/` for examples
2. Review `lib/forensics/` for reusable utilities
3. Examine `templates/analyses/report/` for visualization patterns
4. Check logs when plugins don't load
5. Test with known images to verify results

### Example: AI Detection Plugin

The `ai_detection.py` plugin demonstrates advanced ML integration:
- **Model**: GRIP-UNINA ResNet-50 (pre-trained on 400K images)
- **Setup**: See `AI_DETECTION_SETUP.md` for installation
- **Features**: Lazy loading, GPU support, graceful degradation
- **Dependencies**: PyTorch (optional - plugin disabled if unavailable)
- **Results**: Binary classification with probability scores

This shows how to integrate external deep learning models while maintaining fallback behavior when dependencies are missing.

---

**Last Updated**: February 2026  
**Ghiro Version**: modernize branch (Django 4.2 + Python 3.13)  
**Plugin API Version**: 2.0 (BaseAnalyzerModule)
