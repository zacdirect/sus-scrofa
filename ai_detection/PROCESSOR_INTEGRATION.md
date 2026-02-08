# Processor Integration with Detection Architecture

## Overview

The Ghiro processing system uses **analyzer modules** (processors) that run during image analysis. The AI detection architecture integrates seamlessly as one of these analyzer modules.

## Architecture Fit

```
┌─────────────────────────────────────────────────────────┐
│              AnalysisManager                            │
│  (Django processing system)                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Runs analyzer modules in order
                     │
     ┌───────────────┼───────────────────────────────┐
     │               │                               │
     ▼               ▼                               ▼
┌──────────┐  ┌─────────────┐  ┌────────────────────────┐
│  Meta    │  │ AIDetection │  │ ConfidenceScoring     │
│  Module  │  │   Module    │  │     Module             │
│ (order=5)│  │ (order=30)  │  │   (order=90)          │
└──────────┘  └──────┬──────┘  └────────────────────────┘
                     │
                     │ Uses detection architecture
                     │
         ┌───────────┴────────────┐
         │                        │
         ▼                        ▼
    ┌────────────┐        ┌──────────────┐
    │Orchestrator│        │   Auditor    │
    │  (runs)    │───────▶│ (gatekeeper) │
    └────────────┘        └──────────────┘
         │                        │
         │ Runs detectors         │ Consolidates
         ▼                        ▼
    ┌─────────────────┐    ┌────────────────┐
    │ MetadataDetector│    │ 3 Buckets:     │
    │  SPAIDetector   │    │ - Authenticity │
    └─────────────────┘    │ - AI Prob      │
                          │ - Manipulation │
                          └────────────────┘
```

## How It Works

### 1. AnalysisManager (Ghiro's Processor System)
- Discovers and loads all analyzer modules from `plugins/analyzer/`
- Sorts by execution order
- Runs each module in sequence
- Each module adds results to a shared data structure

### 2. AIDetection Analyzer Module
**Location**: `plugins/analyzer/ai_detection.py`
**Order**: 30 (runs after basic metadata but before final scoring)

**Role**: Bridge between Ghiro's processing system and our detection architecture

```python
class AIDetection(BaseAnalyzerModule):
    def run(self, task):
        # Uses the orchestrator
        detector = MultiLayerDetector(enable_ml=True)
        result = detector.detect(image_path)
        
        # Formats results for Ghiro
        return {
            'ai_detection': {
                'authenticity_score': result['authenticity_score'],
                'detected_types': result['detected_types'],
                'detection_layers': [...]
            }
        }
```

### 3. ConfidenceScoring Analyzer Module
**Location**: `plugins/analyzer/confidence_scoring.py`
**Order**: 90 (runs LAST to aggregate all results)

**Role**: Aggregates all analyzer results into final scores

```python
class ConfidenceScoringProcessing(BaseAnalyzerModule):
    def run(self, task):
        # Uses lib/forensics/confidence.py
        confidence = calculate_manipulation_confidence(self.data)
        
        # confidence.py ALREADY extracts auditor's three buckets:
        # - authenticity_score (0-100)
        # - ai_generated_probability (0-100)
        # - manipulation_probability (0-100)
        
        return confidence
```

### 4. Confidence Calculation
**Location**: `lib/forensics/confidence.py`

**Role**: Extracts auditor's consolidated results

```python
def calculate_manipulation_confidence(results):
    # PRIORITY 1: Check for auditor's authenticity score
    if 'ai_detection' in results:
        if 'authenticity_score' in results['ai_detection']:
            # Extract auditor's three-bucket consolidation
            auth_score = results['ai_detection']['authenticity_score']
            
            # Extract component probabilities from audit metadata
            for layer in results['ai_detection']['detection_layers']:
                if layer.get('method') == 'heuristic' and 'metadata' in layer:
                    ai_prob = layer['metadata'].get('ai_probability')
                    manip_prob = layer['metadata'].get('manipulation_probability')
            
            # Convert to verdict
            if auth_score <= 40:
                verdict = 'fake'
            elif auth_score >= 60:
                verdict = 'real'
            else:
                verdict = 'uncertain'
    
    # PRIORITY 2: Legacy forensic methods (fallback)
    # ... older detection logic ...
```

## Data Flow Example

### User Uploads Image
```
1. Django creates Analysis task
2. AnalysisManager picks up task
3. Runs analyzer modules in order:

   MetadataModule (order=5):
   └─ Extracts EXIF, dimensions, etc.
   
   AIDetection (order=30):
   ├─ Orchestrator.detect(image)
   │  ├─ MetadataDetector runs → finds AI dimensions
   │  ├─ Auditor.should_stop_early()? → No, continue
   │  ├─ SPAIDetector runs → finds AI patterns
   │  └─ Auditor.detect() → FINAL SUMMARY
   │     ├─ Authenticity: 15/100
   │     ├─ AI Probability: 85%
   │     └─ Manipulation Probability: 60%
   └─ Returns formatted results to Ghiro
   
   ConfidenceScoring (order=90):
   └─ Aggregates all results
      └─ Extracts auditor's three buckets
      └─ Stores final verdict

4. Results saved to database
5. User sees verdict in UI
```

## Key Integration Points

### 1. Orchestrator Instantiation
```python
# plugins/analyzer/ai_detection.py
self._detector = MultiLayerDetector(enable_ml=True)
```
- Creates orchestrator with detectors
- Auditor is automatically created inside orchestrator
- No need to manually manage auditor

### 2. Result Extraction
```python
# plugins/analyzer/ai_detection.py
detection_result = self._detector.detect(image_path)

# Extract auditor's results
authenticity_score = detection_result.get('authenticity_score')
detected_types = detection_result.get('detected_types')

# Extract from detection layers
for layer in detection_result['layer_results']:
    if 'metadata' in layer:  # Auditor's metadata
        ai_probability = layer['metadata'].get('ai_probability')
        manipulation_probability = layer['metadata'].get('manipulation_probability')
```

### 3. Confidence Consolidation
```python
# lib/forensics/confidence.py
# Extracts auditor's three buckets from ai_detection results
confidence['authenticity_score'] = ai_det['authenticity_score']
confidence['ai_generated_probability'] = audit_metadata.get('ai_probability')
confidence['confidence_score'] = audit_metadata.get('manipulation_probability')
```

## Why This Works

### Separation of Concerns
- **AnalysisManager**: Orchestrates all analyzer modules (Ghiro level)
- **AIDetection module**: Bridges to detection architecture
- **MultiLayerDetector**: Orchestrates detectors (detection level)
- **ComplianceAuditor**: Gatekeeper for verdicts (decision level)

Each level has clear responsibility:
1. Ghiro: Run modules, store results
2. AIDetection module: Call detection system, format output
3. Orchestrator: Run detectors efficiently
4. Auditor: Make all decisions, consolidate into three buckets

### Clean Integration
The detection architecture is **completely self-contained**:
- Ghiro doesn't know about detectors or auditors
- Detection system doesn't know about Django or AnalysisManager
- AIDetection module is the clean interface between them

### Extensibility
- Add new analyzer modules: Register in `plugins/analyzer/`
- Add new detectors: Register in orchestrator
- Add new detection types: Update auditor's consolidation logic
- No cross-contamination between systems

## Testing

All tests pass:
- ✅ `test_processing.py` - AnalysisManager works correctly
- ✅ `test_compliance_audit.py` - Auditor consolidates correctly (26 tests)
- ✅ `test_forensics_integration.py` - Integration with forensics data works

Run tests:
```bash
python manage.py test tests.test_processing
python manage.py test tests.test_compliance_audit
python manage.py test tests.test_forensics_integration
```

## Summary

**Processors fit the architecture perfectly**:
- Ghiro's processing system runs analyzer modules
- AIDetection module uses our detection architecture
- Orchestrator runs detectors, consults auditor
- Auditor consolidates into three buckets
- ConfidenceScoring extracts and stores final verdict

Clean separation, clear responsibilities, easy to extend.
