# AI Detection Documentation Index

## Start Here

**New to the codebase?** Start with these in order:

1. **[README.md](README.md)** - Overview and quick start (5 min read)
2. **[ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)** - Component roles (10 min read)
3. **[DETECTOR_SPECIALIZATIONS.md](DETECTOR_SPECIALIZATIONS.md)** - How detectors work (10 min read)

## Core Concepts

### Architecture
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture
  - Three-component pattern (Orchestrator, Detectors, Auditor)
  - Three-bucket consolidation (Authenticity, AI, Manipulation)
  - Data flow diagrams
  - Early stopping logic

### Design Philosophy  
- **[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)** - Why we built it this way
  - Why auditor is separate from detectors
  - Why detectors have different specializations
  - Performance trade-offs
  - Future scalability

### Detector Specializations
- **[DETECTOR_SPECIALIZATIONS.md](DETECTOR_SPECIALIZATIONS.md)** - How varied detectors feed into three buckets
  - AI-only detectors
  - Manipulation-only detectors
  - Multi-aspect detectors
  - Consolidation examples

## Practical Guides

### For Developers Adding Features
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - How to add new detectors
  - Step-by-step tutorial
  - Code templates
  - Testing strategies
  - Common patterns

### For Understanding Changes
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - What changed and why
  - Before/after comparison
  - Naming decisions
  - Structural improvements

## Quick Reference

### Key Classes

```python
# Orchestrator - runs detectors
from ai_detection.detectors.orchestrator import MultiLayerDetector
orch = MultiLayerDetector()

# Auditor - the gatekeeper
from ai_detection.detectors.compliance_audit import ComplianceAuditor
auditor = ComplianceAuditor()

# Detector base class
from ai_detection.detectors.base import BaseDetector
class MyDetector(BaseDetector):
    pass
```

### Key Concepts

| Concept | Description | Doc |
|---------|-------------|-----|
| Orchestrator | Runs detectors in efficient order | [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) |
| Detectors | Specialists (AI/manipulation/both) | [DETECTOR_SPECIALIZATIONS.md](DETECTOR_SPECIALIZATIONS.md) |
| Auditor | Gatekeeper - consolidates into 3 buckets | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Three Buckets | Authenticity, AI Prob, Manipulation Prob | [DETECTOR_SPECIALIZATIONS.md](DETECTOR_SPECIALIZATIONS.md) |
| Early Stopping | Auditor decides when to stop | [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) |

### File Structure

```
ai_detection/
├── README.md                          ← Start here
├── ARCHITECTURE.md                    ← Full architecture
├── ARCHITECTURE_SUMMARY.md            ← Quick overview
├── DESIGN_DECISIONS.md                ← Why we built it this way
├── DETECTOR_SPECIALIZATIONS.md        ← Detector types & consolidation
├── DEVELOPER_GUIDE.md                 ← How to add detectors
├── REFACTORING_SUMMARY.md             ← Recent changes
└── detectors/
    ├── base.py                        ← BaseDetector interface
    ├── orchestrator.py                ← MultiLayerDetector
    ├── compliance_audit.py            ← ComplianceAuditor (gatekeeper)
    ├── metadata.py                    ← MetadataDetector
    └── spai_detector.py               ← SPAIDetector
```

## Common Questions

### "I want to add a new detector that only detects AI. Where do I start?"
1. Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Step 1: Create detector class
2. See [DETECTOR_SPECIALIZATIONS.md](DETECTOR_SPECIALIZATIONS.md) - "Scenario 1: AI-Only Detector"
3. Add your finding types to auditor's `AI_FINDING_TYPES` list

### "I want to add a detector that detects both AI and manipulation. How does that work?"
1. See [DETECTOR_SPECIALIZATIONS.md](DETECTOR_SPECIALIZATIONS.md) - "Example: Multi-Aspect Detector"
2. Your detector returns multiple `detected_types`
3. Auditor automatically routes them to appropriate buckets

### "Why is the auditor not a detector?"
See [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) - "Why Separate the Auditor from Detectors?"

### "How does early stopping work?"
See [ARCHITECTURE.md](ARCHITECTURE.md) - "Early Stopping Logic" section

### "What are the three buckets and why do we have them?"
See [DETECTOR_SPECIALIZATIONS.md](DETECTOR_SPECIALIZATIONS.md) - "The Solution: Auditor Consolidates Into Three Buckets"

### "How do I test my new detector?"
See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - "Step 3: Test Your Detector"

## Documentation Standards

When updating docs:
- Keep examples concrete and runnable
- Update all relevant docs (not just one)
- Add diagrams for complex concepts
- Link between related docs

## Version History

- **2026-02-08**: Major refactor - separated auditor from detectors, clarified three-bucket consolidation
- **Previous**: Hybrid approach with ComplianceAuditDetector
