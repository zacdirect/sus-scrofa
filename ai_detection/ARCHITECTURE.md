# AI Detection System Architecture

## Overview

The AI detection system uses a **multi-layer detection architecture** where specialized detectors analyze different aspects of an image, and a compliance auditor aggregates all findings into a final verdict.

## Core Principles

### 1. Separation of Concerns
Each detector is responsible for its own domain of expertise:

**Detectors are specialized analyzers** - they focus on what they know:

- **AI Generation Detectors**: Look for AI-specific patterns
  - Example: `SPAIDetector` - ML model trained on AI vs real
  - Returns findings about AI generation likelihood

- **Manipulation Detectors**: Look for editing/tampering
  - Example: `ELADetector` - error level analysis for photoshop edits
  - Returns findings about traditional manipulation

- **Metadata Detectors**: Analyze file properties and EXIF
  - Example: `MetadataDetector` - checks dimensions, software tags
  - May find evidence of BOTH AI generation AND manipulation

- **Hybrid Detectors**: Analyze multiple aspects
  - Can report on AI indicators, manipulation signs, or both
  - Example: `NoiseAnalysisDetector` - noise patterns reveal both AI and edits

**Each detector returns**: `confidence` + `score` + `detected_types` (what they found)

**Important**: Detectors don't decide "fake or real" - they report what they see

### 2. Orchestration Pattern

The **orchestrator** (`MultiLayerDetector`) is responsible for:
- Running detectors in operationally efficient order (fast → slow)
- Consulting the auditor after each detector runs
- Stopping early if auditor determines we have enough evidence
- **NOT** making any decisions about confidence or verdicts

The orchestrator is purely operational - it doesn't interpret results.

### 3. Audit & Decision Centralization

The **compliance auditor** (`ComplianceAuditor`) is a **separate component** (not a detector) that serves as the single source of truth for:
- **Early stopping decisions**: Reviews results after each detector and decides if we can stop
- **Final verdict calculation**: Aggregates all detector results into single authenticity score
- **Three-bucket consolidation**: Takes varied detector inputs and produces:
  1. **Authenticity Score (0-100)**: Overall fake ← → real
  2. **AI Generation Probability (0-100)**: Synthetic/generated content
  3. **Manipulation Probability (0-100)**: Traditional editing/tampering
- **Risk assessment**: Determines overall confidence in the verdict

**Critical Distinction**: The auditor is NOT in the detectors list. It's a separate component that:
1. Is consulted after each detector runs (`should_stop_early()`)
2. Consolidates diverse findings into our three standard metrics
3. Provides final summary at the end (`detect()` for aggregation)

**Why consolidation matters**: Detectors have different specializations:
- Some only detect AI (SPAIDetector)
- Some only detect manipulation (ELADetector)
- Some detect both (MetadataDetector, NoiseAnalysis)
- Auditor unifies everything into consistent output

#### Auditor Responsibilities

```python
class ComplianceAuditor:
    # This is the GATEKEEPER - not a detector
    # It reviews, decides, and provides final verdict
    
    def should_stop_early(self, current_results: List[DetectionResult]) -> bool:
        """
        Review current detector results and decide if we can stop early.
        
        Called by orchestrator after EACH detector runs.
        
        Returns True if:
        - We have HIGH confidence (>90) from a reliable detector
        - The finding is definitive (clear AI generation or manipulation)
        - Running more detectors won't change the verdict
        
        This saves compute resources when verdict is certain.
        """
        pass
    
    def detect(self, image_path: str) -> DetectionResult:
        """
        Aggregate all findings into final verdict.
        
        Called by orchestrator ONCE at the end (regardless of early stop).
        
        Re-analyzes the image to generate final verdict based on all evidence
        collected by detectors that ran.
        
        Returns:
            DetectionResult with:
            - authenticity_score: 0-100 (0=fake, 100=real)
            - detected_types: List of all findings
            - metadata: {
                'ai_probability': 0-100,
                'manipulation_probability': 0-100,
                'findings_count': int
              }
        """
        pass
```

## Data Flow

```
┌─────────────────┐
│  Image Upload   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│           MultiLayerDetector                        │
│  (Orchestrator - Operational Only)                  │
└─────────────────────────────────────────────────────┘
         │
         │  Run detectors in operationally efficient order:
         │
         ├──▶ 1. MetadataDetector (fast, deterministic)
         │         │
         │         ├─ Check EXIF data
         │         ├─ Analyze dimensions
         │         └─ Return: confidence + score
         │         │
         │         ▼
         │    ┌─────────────────────────────────────┐
         │    │  Consult Auditor: should_stop_early? │
         │    │  (Auditor reviews MetadataDetector   │
         │    │   result and decides)                │
         │    └─────────────────────────────────────┘
         │         │
         │         ├─ If YES → Skip remaining, go to summary
         │         └─ If NO  → Continue...
         │
         ├──▶ 2. SPAIDetector (slower, ML-based)
         │         │
         │         ├─ Run ML model
         │         └─ Return: confidence + score
         │         │
         │         ▼
         │    ┌─────────────────────────────────────┐
         │    │  Consult Auditor: should_stop_early? │
         │    │  (Auditor reviews all results so far)│
         │    └─────────────────────────────────────┘
         │         │
         │         ├─ If YES → Go to summary
         │         └─ If NO  → Continue to next detector...
         │
         └──▶ 3. Final Summary (ALWAYS happens)
                   │
                   ▼
              ┌────────────────────────────────────────┐
              │  Auditor.detect(image_path)            │
              │  (Re-analyzes image, aggregates all    │
              │   findings from detectors that ran)    │
              └────────────────────────────────────────┘
                   │
                   ├─ Calculate authenticity_score (0-100)
                   ├─ Calculate component probabilities
                   ├─ Generate final verdict
                   └─ Return: Final DetectionResult
```

**Key Points:**
- Auditor is consulted AFTER each detector (may stop early)
- Auditor ALWAYS provides final summary (even if stopped early)
- Auditor is NOT in the detector pipeline - it's a separate reviewer

## Detection Result Schema

### Individual Detector Results (Varied by Specialization)

Detectors report what they find - specialization varies:

```python
# AI-focused detector
DetectionResult(
    confidence=90.0,
    score=85.0,
    detected_types=['ai_generation', 'synthetic_noise'],
    metadata=None
)

# Manipulation-focused detector
DetectionResult(
    confidence=75.0,
    score=60.0,
    detected_types=['ela_anomaly', 'clone_stamp'],
    metadata=None
)

# Multi-aspect detector (reports both)
DetectionResult(
    confidence=80.0,
    score=70.0,
    detected_types=['metadata_stripped', 'ai_dimensions', 'suspicious_software'],
    metadata=None
)
```

**Note**: Individual detectors don't categorize into our three buckets - they just report findings

### Compliance Audit Result (Final - Three Buckets)

The auditor consolidates all varied detector findings into three standard metrics:

```python
DetectionResult(
    authenticity_score=23,     # Bucket 1: Fake ← → Real (0-100)
    detected_types=[           # All findings from all detectors
        'ai_generation',       # ← From AI-focused detectors
        'metadata_stripped',   # ← From metadata detector
        'ela_anomaly',         # ← From manipulation detector
        'noise_analysis'       # ← From multi-aspect detector
    ],
    metadata={
        'ai_probability': 85.0,              # Bucket 2: AI Generation (0-100)
        'manipulation_probability': 60.0,    # Bucket 3: Traditional Edit (0-100)
        'findings_count': 7                  # Total findings consolidated
    }
)
```

**Consolidation Logic**:
- Takes findings from AI detectors → feeds into `ai_probability`
- Takes findings from manipulation detectors → feeds into `manipulation_probability`
- Combines both + metadata evidence → calculates `authenticity_score`
- Result: Consistent three-bucket output regardless of which detectors ran

## Early Stopping Logic

The auditor uses these criteria to determine if analysis can stop:

### Stop Conditions
1. **Definitive AI Generation** (confidence > 90, clear AI indicators)
   - Perfect square dimensions (512, 1024, 2048, etc.)
   - AI tool metadata (Midjourney, DALL-E, Stable Diffusion)
   - Synthetic noise patterns

2. **Definitive Manipulation** (confidence > 90, clear forensic evidence)
   - Severe ELA anomalies
   - Frequency domain tampering
   - Clone detection matches

3. **Definitive Authenticity** (confidence > 95, strong positive evidence)
   - High noise variance (>5.0)
   - Complete EXIF chain
   - Known camera model with matching characteristics

### Continue Conditions
- Confidence < 90 (uncertain, need more evidence)
- Mixed signals (some AI indicators, some authentic markers)
- Edge cases (unusual but potentially legitimate images)

## Three-Bucket Consolidation

The auditor consolidates varied detector findings into three standard metrics:

### Bucket 1: Authenticity Score (0-100)
Overall verdict: 0 = definitely fake, 100 = definitely real
- Combines evidence from ALL detectors
- Weighs AI indicators, manipulation signs, and authenticity markers
- Primary metric shown to users

### Bucket 2: AI Generation Probability (0-100)
Consolidates findings from detectors that report AI evidence:
- **AI-focused detectors**: SPAI model predictions, AI tool signatures
- **Multi-aspect detectors**: AI dimensions (512x512, 1024x1024), synthetic noise
- **Metadata detectors**: AI software tags (Midjourney, DALL-E, Stable Diffusion)
- Weights:
  - AI Indicators (40 pts): Direct AI tool evidence
  - Dimensions (30 pts): Perfect square ratios typical of generators
  - Synthetic Noise (35 pts): Unusually uniform patterns

### Bucket 3: Manipulation Probability (0-100)
Consolidates findings from detectors that report editing evidence:
- **Manipulation-focused detectors**: ELA anomalies, clone detection, frequency analysis
- **Multi-aspect detectors**: Noise inconsistencies, metadata stripping
- **Forensic detectors**: DCT coefficients, JPEG artifacts
- Weights:
  - Forensic Findings (40 pts): ELA, clone stamps, splicing
  - Noise Analysis (30 pts): Inconsistent patterns across regions
  - Frequency Domain (25 pts): Compression artifacts, tampering
  - Metadata (15 pts): Stripped EXIF, software mismatches

**Note**: POSITIVE findings (evidence of authenticity) REDUCE probability scores.

**Why three buckets**: Users want to know:
1. "Is it fake?" → Authenticity Score
2. "Was it AI generated?" → AI Probability  
3. "Was it edited in Photoshop?" → Manipulation Probability

Detectors can report on any aspect - auditor consolidates into these three answers.

## Frontend Display

The UI prioritizes the aggregated verdict:

```
┌─────────────────────────────────────────┐
│     AUTHENTICITY SCORE: 23/100          │
│         🔴 Likely Fake                  │
│                                         │
│  Large circular progress indicator      │
└─────────────────────────────────────────┘

╔═══════════════════════════════════════════╗
║  ℹ️  Individual Scores (Info Only)       ║
║                                           ║
║  These are shown for transparency but     ║
║  the overall score above is definitive.   ║
╚═══════════════════════════════════════════╝

AI Generation Probability: 85%  [Info Only]
Forensic Suspicion: 60%         [Info Only]
```

## Extension Points

### Adding a New Detector

1. Inherit from `BaseDetector`
2. Implement `detect(image_path: str) -> DetectionResult`
3. Return `confidence` and `score` (not `authenticity_score`)
4. Register in orchestrator at appropriate priority level

```python
class CustomDetector(BaseDetector):
    """Detects specific image characteristics."""
    
    def detect(self, image_path: str) -> DetectionResult:
        # Your detection logic
        return DetectionResult(
            confidence=90.0,
            score=75.0,
            detected_types=['custom_finding']
        )
```

### Modifying Early Stopping Logic

All changes go in `ComplianceAuditor.should_stop_early()`:
- Adjust confidence thresholds
- Add new stopping conditions
- Modify risk assessment logic

### Adjusting Component Probabilities

Modify weight distributions in:
- `ComplianceAuditor._calculate_ai_probability()`
- `ComplianceAuditor._calculate_manipulation_probability()`

## Testing Strategy

### Unit Tests
- Each detector tested independently
- Mock image inputs with known characteristics
- Verify confidence and score calculations

### Integration Tests
- Full orchestration flow
- Early stopping verification
- Audit aggregation correctness

### Performance Tests
- Measure detector execution time
- Verify early stopping saves compute
- Benchmark full pipeline throughput

## Future Enhancements

1. **Confidence Calibration**: ML model to adjust detector confidence weights
2. **Adaptive Ordering**: Reorder detectors based on image characteristics
3. **Parallel Execution**: Run independent detectors simultaneously
4. **Result Caching**: Store intermediate results for similar images
5. **Explainability**: Generate natural language explanations of findings
