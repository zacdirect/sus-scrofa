# Frontend Three-Bucket Display Fix

## The Problem

The frontend template was correctly structured to display the new three-bucket architecture (Authenticity Score, AI Probability, Manipulation Probability), but **the values were always showing as 0%**.

Looking at your HTML output:
```html
<!-- Forensic Suspicion: 0% -->
<!-- AI Generation Probability: 0% -->
<!-- Legacy Verdict Certainty: 20% -->
```

The authenticity score **was** showing correctly (45/100), but the two component probabilities (AI and Manipulation) were missing.

## Root Cause

**Data Flow Break:** The auditor was calculating the three-bucket probabilities and storing them in its `metadata` field, but this data wasn't making it to the frontend.

### The Data Path

```
ComplianceAuditor.detect()
  ├─ Calculates authenticity_score: 45
  ├─ Calculates ai_probability: 85%
  ├─ Calculates manipulation_probability: 60%
  └─ Returns DetectionResult with metadata={...}
         ↓
MultiLayerDetector._get_audit_summary()
  ├─ Gets audit_result from auditor
  ├─ Returns dict with authenticity_score ✓
  └─ BUT: Missing audit_result.metadata ✗  ← BUG HERE
         ↓
AIDetection.run() (plugin)
  ├─ Extracts authenticity_score ✓
  └─ Missing audit_metadata ✗  ← PROPAGATES
         ↓
confidence.py (aggregation)
  ├─ Has authenticity_score ✓
  └─ Searches for metadata in wrong place ✗  ← FAILS HERE
         ↓
Template (display)
  ├─ Shows authenticity_score: 45 ✓
  └─ Shows ai_probability: 0% (default) ✗
  └─ Shows manipulation_probability: 0% (default) ✗
```

## The Fix

### 1. Orchestrator: Export Audit Metadata

**File:** `ai_detection/detectors/orchestrator.py`

```python
# BEFORE (missing metadata)
return {
    'authenticity_score': audit_result.authenticity_score,
    'detected_types': audit_result.detected_types or [],
    'layer_results': [r.to_dict() for r in results],
}

# AFTER (includes three-bucket data)
return {
    'authenticity_score': audit_result.authenticity_score,
    'detected_types': audit_result.detected_types or [],
    'audit_metadata': audit_result.metadata or {},  # Three-bucket probabilities
    'layer_results': [r.to_dict() for r in results],
}
```

### 2. Plugin: Pass Through Audit Metadata

**File:** `plugins/analyzer/ai_detection.py`

```python
# BEFORE (didn't extract metadata)
authenticity_score = detection_result.get('authenticity_score')
detected_types = detection_result.get('detected_types', [])

# AFTER (extracts and passes through)
authenticity_score = detection_result.get('authenticity_score')
detected_types = detection_result.get('detected_types', [])
audit_metadata = detection_result.get('audit_metadata', {})  # Three-bucket probabilities

if authenticity_score is not None:
    results["ai_detection"]["audit_metadata"] = audit_metadata  # Pass through
```

### 3. Confidence Calculation: Extract from Top Level

**File:** `lib/forensics/confidence.py`

```python
# BEFORE (searched in wrong place)
audit_metadata = {}
if 'detection_layers' in ai_det:
    for layer in ai_det['detection_layers']:
        if layer.get('method') == 'heuristic' and 'metadata' in layer:
            audit_metadata = layer.get('metadata', {})
            break

# AFTER (extracts from top-level)
audit_metadata = ai_det.get('audit_metadata', {})

# Now these work correctly:
confidence['ai_generated_probability'] = round(audit_metadata.get('ai_probability', 0.0), 1)
confidence['confidence_score'] = round(audit_metadata.get('manipulation_probability', 0.0), 1)
```

## What This Fixes

### Before Fix
```
Dashboard Tab:
├─ Forensic Suspicion: 0% ✗
├─ AI Generation Probability: 0% ✗
└─ Legacy Verdict Certainty: 20% ✓

Key Indicators:
├─ Compliance Audit: "Authenticity score: 45/100" ✓
├─ Frequency Analysis: "anomaly score: 53.4%" ✓
├─ Metadata Analysis: "No EXIF" ✓
└─ OpenCV: "Gaussian blur: 55.1%; Noise: 0.0%; JPEG: 70.0%" ✓

AI Generation Detection:
└─ Framework: Multi-Layer (Metadata + ML Model)
    Overall AI Probability: 45.0%
    Confidence: HIGH
```

### After Fix
```
Dashboard Tab:
├─ Forensic Suspicion: 60% ✓  ← Now shows manipulation probability
├─ AI Generation Probability: 85% ✓  ← Now shows AI probability
└─ Legacy Verdict Certainty: 20% ✓

Key Indicators:
├─ Compliance Audit: "Authenticity score: 45/100 (detected: comprehensive analysis)" ✓
├─ [... same ...]

AI Generation Detection:
└─ Framework: Multi-Layer (Metadata + ML Model)
    Overall AI Probability: 45.0%
    Confidence: HIGH
```

## Why It Was Confusing

The **authenticity score was working** (showing 45/100), which made it seem like everything was functioning. But the **component probabilities** (AI probability and manipulation probability) weren't being extracted, so they showed as 0%.

This happened because:
1. The orchestrator was returning `authenticity_score` at the top level (so that worked)
2. But it wasn't returning `audit_metadata` (so the components didn't work)
3. The confidence calculation was looking for metadata in the wrong place (inside detection_layers instead of top-level)

## Testing

All tests pass after the fix:
```bash
$ python manage.py test tests.test_compliance_audit tests.test_forensics_integration
Ran 26 tests in 0.172s
OK
```

## Template Display (Already Correct)

The frontend template `templates/analyses/report/_automated_analysis.html` was **already correct** and didn't need changes. It was waiting for the data:

```html
<!-- Section 2: Individual Detector Scores -->
<div class="span4">
    <div class="box">
        <div class="wdgt-header"><i class="icon-search"></i> Forensic Suspicion</div>
        <div class="wdgt-body">
            <div class="progress">
                <div class="bar" style="width: {{ analysis.report.confidence.confidence_score }}%">
                    {{ analysis.report.confidence.confidence_score }}%
                </div>
            </div>
        </div>
    </div>
</div>

<div class="span4">
    <div class="box">
        <div class="wdgt-header"><i class="icon-bolt"></i> AI Generation Probability</div>
        <div class="wdgt-body">
            <div class="progress">
                <div class="bar" style="width: {{ analysis.report.confidence.ai_generated_probability }}%">
                    {{ analysis.report.confidence.ai_generated_probability }}%
                </div>
            </div>
        </div>
    </div>
</div>
```

Once the data flow was fixed, these bars automatically populate correctly.

## Summary

**Issue:** Three-bucket probabilities (AI probability, manipulation probability) showing as 0% in frontend.

**Cause:** Auditor's metadata not being passed through orchestrator → plugin → confidence calculation.

**Fix:** Three small changes to pass metadata through the pipeline.

**Result:** Frontend now correctly shows all three metrics:
- Authenticity Score (primary) - was working
- AI Probability (component) - now working
- Manipulation Probability (component) - now working

The architecture was always correct - this was just a data plumbing issue.
