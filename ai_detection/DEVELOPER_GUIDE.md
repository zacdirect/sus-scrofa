# Quick Reference: Adding a New Detector

## Step 1: Create Your Detector Class

```python
# my_detector.py
from ai_detection.detectors.base import BaseDetector, DetectionResult, DetectionMethod

class MyCustomDetector(BaseDetector):
    """
    Detects [specific characteristic].
    
    Type: [deterministic/ml-based/hybrid]
    Speed: [fast/medium/slow]
    """
    
    name = "MyCustomDetector"
    
    def check_deps(self) -> bool:
        """Check if required dependencies are available."""
        try:
            # Import any required libraries
            import some_required_lib
            return True
        except ImportError:
            return False
    
    def get_order(self) -> int:
        """
        Execution order priority.
        
        Lower = runs first (fast detectors)
        Higher = runs last (slow detectors)
        
        Guidelines:
        - 1-10: Fast heuristics (metadata, dimensions)
        - 11-20: Medium complexity (noise analysis, frequency)
        - 21-30: Slow ML models (neural networks)
        """
        return 15  # Adjust based on your detector's speed
    
    def detect(self, image_path: str, context=None) -> DetectionResult:
        """
        Analyze the image and return findings.
        
        Args:
            image_path: Path to image file
            context: Optional ResultStore — read from it if you want to
                     see what earlier detectors found ("better together"
                     pattern), or ignore it entirely.
            
        Returns:
            DetectionResult with:
            - confidence: 0-100 (how sure you are)
            - score: 0.0-1.0 (suspicion level, higher = more suspicious)
            - detected_types: List of finding types
            - evidence: Human-readable explanation
        """
        # Your detection logic here
        from PIL import Image
        img = Image.open(image_path)
        
        # Example analysis
        is_suspicious = False
        confidence = 0
        detected = []
        
        # ... your analysis ...
        
        return DetectionResult(
            method=DetectionMethod.HEURISTIC,  # or ML or HYBRID
            confidence=confidence,
            score=0.8 if is_suspicious else 0.2,
            detected_types=detected,
            evidence=f"MyCustomDetector: [your explanation]"
        )
```

## Step 2: Register in Orchestrator

```python
# ai_detection/detectors/orchestrator.py

from .my_detector import MyCustomDetector  # Add import

class MultiLayerDetector:
    def __init__(self, enable_ml: bool = True):
        self.detectors: List[BaseDetector] = []
        self.auditor = ComplianceAuditor()  # The gatekeeper
        
        # Register detectors in logical order
        self._register_detector(MetadataDetector())
        self._register_detector(MyCustomDetector())  # Add here
        
        if enable_ml:
            self._register_detector(SDXLDetector())
            self._register_detector(SPAIDetector())
```

## Step 3: Test Your Detector

```python
# tests/test_my_detector.py
import pytest
from ai_detection.detectors.my_detector import MyCustomDetector
from tests.fixtures import get_test_image

def test_my_detector_on_suspicious_image():
    detector = MyCustomDetector()
    
    # Test with suspicious image
    result = detector.detect(get_test_image('suspicious.jpg'))
    
    assert result.confidence > 70, "Should be confident on obvious case"
    assert result.score > 0.6, "Should have high suspicion score"
    assert len(result.detected_types) > 0, "Should detect something"

def test_my_detector_on_authentic_image():
    detector = MyCustomDetector()
    
    # Test with authentic image
    result = detector.detect(get_test_image('authentic.jpg'))
    
    assert result.score < 0.4, "Should have low suspicion on authentic"
```

## Step 4: Test Integration

```python
# Test with orchestrator
from ai_detection.detectors.orchestrator import MultiLayerDetector

def test_orchestrator_includes_my_detector():
    orch = MultiLayerDetector()
    
    # Verify your detector is registered
    detector_names = [d.name for d in orch.detectors]
    assert "MyCustomDetector" in detector_names
    
    # Test it runs
    result = orch.detect('test_image.jpg')
    
    # Check your detector's output in layer results
    layer_names = [layer['method'] for layer in result['layer_results']]
    assert any('MyCustomDetector' in str(layer) for layer in result['layer_results'])
```

## Common Patterns

### Pattern 1: Fast Heuristic Detector

```python
class FastHeuristicDetector(BaseDetector):
    name = "FastHeuristic"
    
    def get_order(self) -> int:
        return 5  # Run early
    
    def detect(self, image_path: str, context=None) -> DetectionResult:
        # Quick checks only
        img = Image.open(image_path)
        width, height = img.size
        
        # Example: Check for suspicious dimensions
        if width == height and width in [512, 1024, 2048]:
            return DetectionResult(
                method=DetectionMethod.HEURISTIC,
                confidence=80,
                score=0.7,
                detected_types=['suspicious_dimensions'],
                evidence="Perfect square dimensions typical of AI"
            )
        
        return DetectionResult(
            method=DetectionMethod.HEURISTIC,
            confidence=50,
            score=0.3,
            detected_types=[],
            evidence="No suspicious dimensions found"
        )
```

### Pattern 2: ML-Based Detector

```python
class MLModelDetector(BaseDetector):
    name = "MLModel"
    
    def __init__(self):
        super().__init__()
        self.model = None
    
    def check_deps(self) -> bool:
        try:
            import torch
            self.model = torch.load('model.pth')
            return True
        except:
            return False
    
    def get_order(self) -> int:
        return 25  # Run late (slow)
    
    def detect(self, image_path: str, context=None) -> DetectionResult:
        # Run ML model
        prediction = self.model.predict(image_path)
        
        return DetectionResult(
            method=DetectionMethod.ML,
            confidence=int(prediction['confidence'] * 100),
            score=prediction['ai_probability'],
            detected_types=['ml_detection'] if prediction['ai_probability'] > 0.5 else [],
            evidence=f"ML model confidence: {prediction['confidence']:.2f}"
        )
```

### Pattern 3: Hybrid Detector ("Better Together")

```python
class HybridDetector(BaseDetector):
    name = "Hybrid"
    
    def get_order(self) -> int:
        return 15  # Medium priority
    
    def detect(self, image_path: str, context=None) -> DetectionResult:
        # Combine heuristics and ML
        
        # 1. Fast heuristic checks
        img = Image.open(image_path)
        heuristic_score = self._check_heuristics(img)
        
        # 2. Optionally check what earlier detectors found
        if context:
            metadata_result = context.get('MetadataDetector')
            if metadata_result and metadata_result.is_ai_generated:
                # Metadata already flagged AI — adjust our approach
                heuristic_score = max(heuristic_score, 0.6)
        
        # 3. Only run ML if heuristics are uncertain
        if 0.3 < heuristic_score < 0.7:
            ml_score = self._run_ml_model(image_path)
            final_score = (heuristic_score + ml_score) / 2
        else:
            final_score = heuristic_score
        
        return DetectionResult(
            method=DetectionMethod.HYBRID,
            confidence=70,
            score=final_score,
            detected_types=['hybrid_detection'],
            evidence=f"Hybrid analysis: {final_score:.2f}"
        )
```

The `context` parameter is a `ResultStore` — a shared read/write store that
the orchestrator creates per analysis run.  Most detectors ignore it (they
just worry about themselves), but it's there for detectors that benefit from
knowing what earlier detectors found.

## Debugging Tips

### See Execution Flow

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from ai_detection.detectors.orchestrator import MultiLayerDetector

orch = MultiLayerDetector()
result = orch.detect('image.jpg')

# Watch for:
# - "Registered detector: YourDetector"
# - "YourDetector: [your evidence]"
# - "Auditor decision: ..."
```

### Check Detector Output

```python
result = orch.detect('image.jpg')

# See all detector outputs
for layer in result['layer_results']:
    print(f"{layer['method']}: confidence={layer['confidence']}, score={layer['score']}")
```

### Test Detector Independently

```python
from ai_detection.detectors.my_detector import MyCustomDetector

detector = MyCustomDetector()
result = detector.detect('test.jpg')

print(f"Confidence: {result.confidence}")
print(f"Score: {result.score}")
print(f"Evidence: {result.evidence}")
print(f"Detected: {result.detected_types}")
```

## What the Auditor Does With Your Results

After your detector runs, the orchestrator records your result into the
shared `ResultStore`.  Then:

1. **Reviews for early stopping**: `should_stop_early(results)`
   - Checks if your confidence is high enough to skip remaining detectors
   - Looks at your detected_types for definitive findings

2. **Includes in final summary**: `detect(image_path, context=store)`
   - The auditor reads your result from the store on its own
   - Your findings feed into the three-bucket consolidation
   - Your detected_types are added to final `detected_types` list

You don't need to worry about integration — just return accurate results!
The orchestrator handles recording, and the auditor handles reading.
