# AI Detection Integration Guide

This document explains how the AI detection module integrates with Ghiro.

## Architecture Overview

```
ghiro/
├── plugins/analyzer/ai_detection.py  # Plugin that uses the module
└── ai_detection/                     # Standalone AI detection module
    ├── Makefile                      # Independent setup/build system
    ├── requirements.txt              # AI-specific dependencies
    ├── spai/                         # SPAI implementation
    │   ├── config.py                 # Configuration system
    │   ├── inference.py              # High-level API (SPAIDetector class)
    │   ├── models/                   # Model architectures
    │   │   └── build.py              # Model builder (TODO: extract from SPAI repo)
    │   └── data/
    │       └── transforms.py         # Image preprocessing
    └── weights/                      # Model files (gitignored)
        └── spai.pth                  # ~100MB
```

## Design Principles

### 1. Modularity
- **Separate directory**: `ai_detection/` is self-contained
- **Own build system**: Independent Makefile with `make setup`, `make verify`, `make clean`
- **Isolated dependencies**: Separate virtual environment, doesn't pollute main Ghiro
- **Easy replacement**: When better AI detection methods emerge, swap in new module

### 2. Performance
- **Embedded code**: Direct Python imports, not subprocess calls
- **Model persistence**: Load once per processor instance, reuse for all images
- **Lazy loading**: Detector only initialized when first needed
- **Performance targets**:
  - First image: 3-5 seconds (includes model loading)
  - Subsequent images: <1 second each
  - 100x faster than subprocess approach

### 3. Integration Pattern
- **Plugin interface**: `plugins/analyzer/ai_detection.py` is the integration point
- **Lazy initialization**: `_get_detector()` method caches detector instance
- **Path manipulation**: Plugin adds `ai_detection/` to `sys.path` at runtime
- **Error handling**: Graceful fallback if module not available

## Plugin Integration

### How the Plugin Works

```python
class SPAIDetection(AnalyzerModule):
    def __init__(self):
        super().__init__()
        self._detector = None  # Lazy-loaded on first use
    
    def check_deps(self):
        """Add ai_detection to sys.path and verify imports"""
        ai_detection_path = Path(__file__).parent.parent.parent / "ai_detection"
        sys.path.insert(0, str(ai_detection_path))
        
        try:
            from spai.inference import SPAIDetector
            return True
        except ImportError:
            return False
    
    def _get_detector(self):
        """Get or create SPAI detector (lazy loading)"""
        if self._detector is None:
            from spai.inference import SPAIDetector
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._detector = SPAIDetector(weights_path, device)
            self.log("INFO", f"SPAI detector loaded successfully on {device}")
        return self._detector
    
    def run(self, task):
        """Run AI detection on image"""
        detector = self._get_detector()  # Reuses existing detector
        result = detector.predict(task.image_path)
        
        return {
            'ai_probability': result['score'],
            'likely_ai': result['is_ai_generated'],
            'confidence': result['confidence'],
            'method': 'SPAI (Spectral AI Detection)'
        }
```

### Key Integration Points

1. **Dependency Check**: `check_deps()` verifies module availability
   - Adds `ai_detection/` to Python path
   - Attempts to import `SPAIDetector`
   - Returns `True` if available, `False` if not installed

2. **Lazy Loading**: `_get_detector()` implements singleton pattern
   - First call: Loads model (3-5 seconds)
   - Subsequent calls: Returns cached instance (<1ms)
   - Detector persists for lifetime of processor

3. **Inference**: `run()` processes images
   - Gets detector (cached or new)
   - Calls `detector.predict(image_path)`
   - Returns formatted results for Ghiro

## Setup Workflow

### For Developers

```bash
# 1. Clone Ghiro repository
git clone <ghiro-repo>
cd ghiro

# 2. Main Ghiro setup
make setup
make install

# 3. AI detection setup (optional)
make ai-setup      # Creates venv, installs deps, downloads weights
make ai-verify     # Tests installation

# 4. Run Ghiro
make run
```

### For Users

If AI detection is not set up, Ghiro will:
- Log warning: "AI detection module not available"
- Continue processing without AI detection
- Other analysis plugins work normally

To enable AI detection later:
```bash
make ai-setup
```

## Makefile Integration

### Main Makefile Targets

```makefile
.PHONY: ai-setup ai-verify ai-clean

ai-setup:  ## Setup AI detection (creates venv, installs deps, downloads weights)
	cd ai_detection && $(MAKE) setup

ai-verify:  ## Verify AI detection installation
	cd ai_detection && $(MAKE) verify

ai-clean:  ## Clean AI detection environment
	cd ai_detection && $(MAKE) clean
```

### AI Detection Makefile Targets

```makefile
setup:    # Complete setup (venv + install + weights)
venv:     # Create virtual environment
install:  # Install PyTorch and dependencies
weights:  # Download SPAI model weights
verify:   # Comprehensive verification
clean:    # Remove venv and cache
```

## Replacement Strategy

### When to Replace

AI detection is rapidly evolving. Consider replacing when:
- New SOTA model published (e.g., CVPR/ICCV/NeurIPS)
- Better performance or accuracy available
- New detection capabilities needed (video, diffusion-specific, etc.)
- Licensing or dependency issues with current method

### How to Replace

1. **Create new module directory**:
   ```
   ai_detection/new_method/
   ├── __init__.py
   ├── config.py
   ├── inference.py       # Must implement same interface
   └── models/
   ```

2. **Implement same interface**:
   ```python
   class NewDetector:
       def __init__(self, weights_path, device):
           """Load model"""
       
       def predict(self, image):
           """Return dict with: score, confidence, is_ai_generated, logit"""
   ```

3. **Update plugin**:
   ```python
   # Old: from spai.inference import SPAIDetector
   # New: from new_method.inference import NewDetector
   ```

4. **Update Makefile**:
   - Point to new weights URL
   - Update dependencies in requirements.txt
   - Update verify target for new model

5. **Update documentation**:
   - README.md with new method description
   - Performance benchmarks
   - Hardware requirements

### Migration Checklist

- [ ] New module implements `predict(image)` interface
- [ ] Returns dict with: `score`, `confidence`, `is_ai_generated`, `logit`
- [ ] Makefile can download/setup new model
- [ ] Plugin updated to use new module
- [ ] Documentation updated
- [ ] Performance tested (first image, subsequent images)
- [ ] Error handling tested (missing weights, wrong device, etc.)

## Troubleshooting

### Plugin shows "AI detection module not available"

1. Check if ai_detection/ exists:
   ```bash
   ls -la ai_detection/
   ```

2. Check if module is set up:
   ```bash
   make ai-verify
   ```

3. If not set up:
   ```bash
   make ai-setup
   ```

### Model loads every image (slow)

This indicates detector is not being cached. Check:

1. Plugin has `_detector` instance variable
2. `_get_detector()` checks `if self._detector is None`
3. Logs show "SPAI detector loaded successfully" only once

### CUDA out of memory

1. Switch to CPU mode in plugin:
   ```python
   self._detector = SPAIDetector(weights_path, device='cpu')
   ```

2. Or use batch processing with smaller batches:
   ```python
   results = detector.predict_batch(images, batch_size=4)
   ```

### Import errors

Check Python path manipulation in plugin:
```python
import sys
from pathlib import Path

ai_detection_path = Path(__file__).parent.parent.parent / "ai_detection"
sys.path.insert(0, str(ai_detection_path))
```

## Performance Monitoring

### Expected Performance

**Hardware**: 8GB GPU, modern CPU
- First image: 3-5 seconds (model loading + inference)
- Image 2-10: <1 second each (cached model)
- Image 11-100: <1 second each (cached model)

**Hardware**: CPU only
- First image: 10-15 seconds (model loading + inference)
- Image 2-10: 2-3 seconds each (cached model)
- Image 11-100: 2-3 seconds each (cached model)

### Profiling

To measure actual performance:

```python
import time

# In plugin's run() method:
start = time.time()
result = detector.predict(image_path)
elapsed = time.time() - start

self.log("INFO", f"AI detection took {elapsed:.2f}s")
```

### Optimization Tips

1. **Batch processing**: For multiple images, use `predict_batch()`
2. **GPU mode**: Ensure CUDA is available and detected
3. **Model caching**: Verify `_detector` instance persists
4. **Image preprocessing**: Don't preprocess same image twice

## Security Considerations

### Model Weights

- **Size**: ~100MB, stored in `ai_detection/weights/`
- **Source**: Google Drive (official SPAI repository)
- **Verification**: SHA256 hash check in Makefile
- **Gitignored**: Weights not committed to repository

### Dependencies

- **Isolation**: Separate virtual environment
- **Requirements**: Pinned versions in `requirements.txt`
- **PyTorch**: Official PyPI index (CUDA 12.4)
- **Updates**: Review changelog before upgrading

### Access Control

- Plugin runs with same permissions as Ghiro
- No network access required after setup
- No external API calls during inference
- Model inference is local-only

## References

- **SPAI Paper**: [CVPR 2025](https://github.com/mever-team/spai)
- **Plugin Code**: `plugins/analyzer/ai_detection.py`
- **Module Code**: `ai_detection/`
- **Setup Guide**: `ai_detection/README.md`
- **Main Makefile**: `Makefile` (ai-* targets)
