# AI Detection Architecture Summary

## Component Roles

### 🎯 Orchestrator (`MultiLayerDetector`)
**Responsibility**: Run detectors in operationally efficient order

**What it does:**
- ✅ Creates a shared `ResultStore` per analysis
- ✅ Runs detectors (fast → slow), passing the store as `context`
- ✅ Records each result into the store
- ✅ Consults auditor for early stopping
- ✅ Hands the store to the auditor for final summary

**What it does NOT do:**
- ❌ Make decisions about confidence
- ❌ Interpret detection results
- ❌ Calculate final verdict
- ❌ Serialize or inject results into the auditor

### 🔍 Detectors (`MetadataDetector`, `SDXLDetector`, `SPAIDetector`, etc.)
**Responsibility**: Analyze specific aspects - report what they find

**Specializations** (detectors focus on what they know):
- **AI-Focused**: Only detect AI generation (e.g., SPAI spectral model, SDXL Swin Transformer)
- **Manipulation-Focused**: Only detect traditional editing (e.g., ELA, clone detection)
- **Multi-Aspect**: Can detect both AI AND manipulation (e.g., metadata, noise analysis)

**Output**: `DetectionResult` with `confidence` + `score` + `detected_types`

**Key Point**: Detectors don't categorize into our three buckets - they just report findings. Some see AI evidence, some see manipulation evidence, some see both.

### ⚖️ Auditor (`ComplianceAuditor`)
**Responsibility**: Make ALL decisions about the image

**NOT a detector** - it's a separate decision-making component

**Two Roles:**

1. **Reviewer** (called after each detector):
   ```python
   should_stop_early(current_results) -> bool
   ```
   - Reviews results so far
   - Decides if we have enough evidence
   - Returns True to stop early and save compute

2. **Consolidator** (called once at the end):
   ```python
   detect(image_path, context=store) -> DetectionResult
   ```
   - Re-analyzes the image itself
   - **Reads prior detector results from the shared `ResultStore`**
   - **Consolidates varied detector findings into three buckets**:
     * Authenticity Score: fake ← → real (0-100)
     * AI Probability: synthetic content (0-100)
     * Manipulation Probability: traditional editing (0-100)
   - Takes AI-focused findings → AI bucket
   - Takes manipulation-focused findings → manipulation bucket
   - Combines all evidence → authenticity score
   - Returns consistent three-bucket output

## Workflow

```
┌────────────┐
│ 1. Upload  │
└─────┬──────┘
      │
      ▼
┌─────────────────────────────────┐
│ 2. Orchestrator: Run Detector 1│ ──▶ MetadataDetector
└─────┬───────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ 3. Auditor: Review Results      │ ──▶ should_stop_early()?
└─────┬───────────────────────────┘
      │
      ├─ YES (Stop) ──────────────┐
      │                            │
      └─ NO (Continue)             │
         │                         │
         ▼                         │
┌─────────────────────────────────┐│
│ 4. Orchestrator: Run Detector 2││ ──▶ SDXLDetector
└─────┬───────────────────────────┘│
      │                            │
      ▼                            │
┌─────────────────────────────────┐│
│ 5. Auditor: Review Results      ││ ──▶ should_stop_early()?
└─────┬───────────────────────────┘│
      │                            │
      ├─ YES (Stop) ──────────────┤
      │                            │
      └─ NO (Continue)             │
         │                         │
         ▼                         │
┌─────────────────────────────────┐│
│ 6. Orchestrator: Run Detector 3││ ──▶ SPAIDetector
└─────┬───────────────────────────┘│
      │                            │
      ▼                            │
┌─────────────────────────────────┐│
│ 7. Auditor: Review Results      ││ ──▶ should_stop_early()?
└─────┬───────────────────────────┘│
      │                            │
      └─ (All detectors done) ─────┤
                                   │
                                   ▼
                     ┌────────────────────────────────────────────────┐
                     │ 8. Auditor: Final Summary                   │
                     │    - Re-analyze image                       │
                     │    - Read ML results from ResultStore        │
                     │    - Calculate score                        │
                     │    - Return verdict                         │
                     └────────────────────────────────────────────────┘
```

## Key Principles

1. **Separation of Concerns**
   - Orchestrator = operational (runs things)
   - Detectors = analysis (find evidence)
   - Auditor = decision-making (interprets evidence)

2. **Auditor is Special**
   - NOT in the detectors list
   - Consulted after every detector
   - Always provides final summary
   - Single source of truth for verdicts

3. **Early Stopping**
   - Auditor decides when to stop
   - Orchestrator just executes the decision
   - Saves compute on obvious cases

4. **Always Complete**
   - Even if stopped early, auditor provides final summary
   - Ensures consistent output format
   - Re-analyzes image for complete findings

## Adding New Detectors

```python
# 1. Create detector
class MyDetector(BaseDetector):
    def detect(self, image_path: str, context=None) -> DetectionResult:
        # Your analysis
        # context is a ResultStore — read from it if you want to see
        # what earlier detectors found, or just ignore it.
        return DetectionResult(
            confidence=90,
            score=0.75,
            detected_types=['my_finding']
        )

# 2. Register in orchestrator
# Edit orchestrator.__init__():
self._register_detector(MyDetector())

# That's it! The orchestrator records your result into the store.
# The auditor reads it from the store automatically.
```

## Testing New Detectors

```python
from ai_detection.detectors.orchestrator import MultiLayerDetector

orch = MultiLayerDetector()
result = orch.detect('test_image.jpg')

# Check what ran
print(f"Detectors run: {len(result['layer_results'])}")
for layer in result['layer_results']:
    print(f"  - {layer['method']}: {layer['confidence']}")

# Check final verdict
print(f"Authenticity: {result['authenticity_score']}/100")
print(f"Verdict: {'FAKE' if result['overall_verdict'] else 'REAL'}")
```
