# Detector Specializations & Three-Bucket Consolidation

## The Problem: Detectors Have Different Expertise

Not all detectors detect the same things. Some are specialists:

```
┌─────────────────┐
│  SPAIDetector   │ → Only trained on: AI vs Real
└─────────────────┘   ❌ Doesn't detect: Traditional edits

┌─────────────────┐
│  ELADetector    │ → Only trained on: Photoshop edits
└─────────────────┘   ❌ Doesn't detect: AI generation

┌─────────────────┐
│MetadataDetector │ → Finds: AI tags OR edit software OR both
└─────────────────┘   ✅ Multi-aspect

┌─────────────────┐
│ NoiseAnalysis   │ → Finds: Synthetic noise OR edit artifacts OR both
└─────────────────┘   ✅ Multi-aspect
```

## The Solution: Auditor Consolidates Into Three Buckets

```
┌────────────────────────────────────────────────────────────┐
│                    DETECTORS (Varied)                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  SPAIDetector                                             │
│  ├─ detected_types: ['ai_generation']                     │
│  └─ confidence: 85                                        │
│                                                            │
│  MetadataDetector                                         │
│  ├─ detected_types: ['ai_dimensions', 'metadata_stripped']│
│  └─ confidence: 70                                        │
│                                                            │
│  ELADetector (future)                                     │
│  ├─ detected_types: ['ela_anomaly', 'clone_stamp']       │
│  └─ confidence: 60                                        │
│                                                            │
│  NoiseAnalysisDetector (future)                           │
│  ├─ detected_types: ['synthetic_noise', 'edit_artifact'] │
│  └─ confidence: 75                                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │   AUDITOR (Consolidator)        │
        │  ComplianceAuditor.detect()     │
        └─────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│              THREE STANDARD BUCKETS                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────────────────────────────────┐         │
│  │ BUCKET 1: Authenticity Score (0-100)        │         │
│  │ fake ← → real                               │         │
│  ├─────────────────────────────────────────────┤         │
│  │ Combines ALL evidence:                      │         │
│  │ • AI indicators from SPAI                   │         │
│  │ • AI dimensions from Metadata               │         │
│  │ • Edit anomalies from ELA                   │         │
│  │ • All noise findings                        │         │
│  │                                             │         │
│  │ Result: 23/100 (likely fake)                │         │
│  └─────────────────────────────────────────────┘         │
│                                                            │
│  ┌─────────────────────────────────────────────┐         │
│  │ BUCKET 2: AI Generation Probability         │         │
│  ├─────────────────────────────────────────────┤         │
│  │ Consolidates AI-related findings:           │         │
│  │ • 'ai_generation' from SPAI        → 40pts  │         │
│  │ • 'ai_dimensions' from Metadata    → 30pts  │         │
│  │ • 'synthetic_noise' from Noise     → 35pts  │         │
│  │                                             │         │
│  │ Result: 85% (high AI probability)           │         │
│  └─────────────────────────────────────────────┘         │
│                                                            │
│  ┌─────────────────────────────────────────────┐         │
│  │ BUCKET 3: Manipulation Probability          │         │
│  ├─────────────────────────────────────────────┤         │
│  │ Consolidates edit-related findings:         │         │
│  │ • 'ela_anomaly' from ELA           → 40pts  │         │
│  │ • 'clone_stamp' from ELA           → 40pts  │         │
│  │ • 'metadata_stripped' from Meta    → 15pts  │         │
│  │ • 'edit_artifact' from Noise       → 30pts  │         │
│  │                                             │         │
│  │ Result: 60% (moderate manipulation)         │         │
│  └─────────────────────────────────────────────┘         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Example: Multi-Aspect Detector

`NoiseAnalysisDetector` might return:

```python
DetectionResult(
    confidence=75,
    score=65,
    detected_types=[
        'synthetic_noise',    # ← AI-related finding
        'noise_inconsistency', # ← Manipulation-related finding
    ]
)
```

The auditor splits these:
- `'synthetic_noise'` → feeds into **AI Probability** bucket
- `'noise_inconsistency'` → feeds into **Manipulation Probability** bucket
- Both → contribute to **Authenticity Score** (overall fake/real)

## Why This Works

### For Detectors (Simple)
- Focus on what you know
- Report what you find
- Don't worry about categorization

```python
class MyDetector(BaseDetector):
    def detect(self, image_path):
        # Just report what you find
        return DetectionResult(
            confidence=80,
            detected_types=['whatever_you_found']
        )
```

### For Auditor (Complex)
- Knows about all finding types
- Maps findings to appropriate buckets
- Handles consolidation logic

```python
class ComplianceAuditor:
    AI_FINDING_TYPES = ['ai_generation', 'ai_dimensions', 'synthetic_noise', ...]
    MANIPULATION_FINDING_TYPES = ['ela_anomaly', 'clone_stamp', 'edit_artifact', ...]
    
    def _calculate_ai_probability(self, findings):
        # Weight AI-related findings
        
    def _calculate_manipulation_probability(self, findings):
        # Weight manipulation-related findings
        
    def detect(self, image_path):
        # Consolidate everything into three buckets
```

### For Users (Clear)
Always get three consistent answers:
1. "Is it fake?" → Authenticity Score
2. "Is it AI generated?" → AI Probability
3. "Was it edited?" → Manipulation Probability

Regardless of which detectors ran or what they're specialized in.

## Adding New Detectors

### Scenario 1: AI-Only Detector
```python
class NewAIDetector(BaseDetector):
    def detect(self, image_path):
        return DetectionResult(
            detected_types=['ai_indicator_xyz']
        )
```

Register it, add `'ai_indicator_xyz'` to auditor's `AI_FINDING_TYPES`. Done.

### Scenario 2: Manipulation-Only Detector
```python
class NewForensicDetector(BaseDetector):
    def detect(self, image_path):
        return DetectionResult(
            detected_types=['forensic_finding_abc']
        )
```

Register it, add `'forensic_finding_abc'` to auditor's `MANIPULATION_FINDING_TYPES`. Done.

### Scenario 3: Multi-Aspect Detector
```python
class NewHybridDetector(BaseDetector):
    def detect(self, image_path):
        return DetectionResult(
            detected_types=[
                'ai_thing',          # ← Goes to AI bucket
                'manipulation_thing' # ← Goes to manipulation bucket
            ]
        )
```

Register it, add both to appropriate auditor lists. Done.

## Key Takeaway

> **Detectors are specialists with varied expertise. The auditor consolidates everything into three standard buckets: Authenticity, AI, and Manipulation.**

This keeps detectors simple and focused, while providing users with consistent, understandable results.
