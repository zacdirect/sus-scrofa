# Plugin Development Guide

How to add new analysis plugins to SusScrofa and integrate them into the
automated confidence scoring system.

---

## Architecture Overview

```
Image uploaded
  └─ AnalysisManager discovers all plugins in plugins/analyzer/
  └─ Sorts them by `order` attribute
  └─ For each plugin (in order):
       ├─ plugin.data = accumulated_results    ← READ previous plugins' output
       ├─ output = plugin.run(task)            ← DO analysis
       └─ accumulated_results.update(output)   ← WRITE results into shared dict
  └─ confidence_scoring (order=90) reads ALL results, computes final verdict
  └─ Everything is saved to the database
```

Every plugin gets to see what all previous plugins produced (via `self.data`)
and adds its own findings (via `self.results`). The confidence scoring plugin
runs last and aggregates everything into the final verdict.

## Current Plugin Inventory

| Order | File | Class | Result Key | Type |
|-------|------|-------|------------|------|
| 10 | `info.py` | `InfoAnalyzer` | `file_name`, `file_size` | Basic |
| 10 | `hash.py` | `HashAnalyzer` | `hash` | Basic |
| 10 | `mime.py` | `MimeAnalyzer` | `mime_type`, `file_type` | Basic |
| 10 | `gexiv.py` | `GexivAnalyzer` | `metadata` | Deterministic |
| 20 | `ela.py` | `ElaAnalyzer` | `ela` | Deterministic |
| 20 | `hashcomparer.py` | `HashComparerAnalyzer` | *(ORM lookup)* | Deterministic |
| 20 | `previewcomparer.py` | `PreviewComparerAnalyzer` | *(mutates metadata)* | Deterministic |
| 25 | `noise_analysis.py` | `NoiseAnalysisAnalyzer` | `noise_analysis` | Deterministic |
| 26 | `frequency_analysis.py` | `FrequencyAnalysisAnalyzer` | `frequency_analysis` | Deterministic |
| 30 | `ai_detection.py` | `AIDetectionAnalyzer` | `ai_detection` | AI/ML |
| 40 | `opencv_analysis.py` | `OpenCVAnalysisAnalyzer` | `opencv_analysis` | AI/ML |
| 65 | `opencv_manipulation.py` | `OpenCVManipulationAnalyzer` | `opencv_manipulation` | AI/ML |
| 80 | `signatures.py` | `SignaturesAnalyzer` | `signatures` | Rule-based |
| **90** | `confidence_scoring.py` | `ConfidenceScoringProcessing` | `confidence` | **Aggregator** |

## Step 1: Create the Plugin

Create a new Python file in `plugins/analyzer/`. It is auto-discovered — no
registration required.

```python
# plugins/analyzer/my_new_check.py

import logging
from lib.analyzer.base import BaseAnalyzerModule

logger = logging.getLogger(__name__)

# Guard your imports so the plugin is skipped if deps are missing
try:
    import some_library
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class MyNewCheckAnalyzer(BaseAnalyzerModule):
    """One-line description of what this plugin detects."""

    # Pick an order between 10-89.
    # Lower = runs earlier. Must run BEFORE confidence_scoring (90).
    # If you need data from another plugin, set your order higher than theirs.
    order = 35

    def check_deps(self):
        """Return True if all dependencies are available."""
        return HAS_DEPS

    def run(self, task):
        """
        Perform analysis.

        Args:
            task: Analysis model instance. Key attributes:
                  - task.id: database ID
                  - task.file_name: original filename
                  - task.get_file_data: raw file bytes
                  - task.image_id: GridFS file ID

        Returns:
            self.results (AutoVivification dict) with your findings.
        """
        try:
            # Read data from previous plugins via self.data
            mime_type = self.data.get("mime_type", "")
            if not mime_type.startswith("image/"):
                return self.results  # skip non-images

            # Do your analysis...
            raw_bytes = task.get_file_data
            score, details = some_library.analyze(raw_bytes)

            # Write results under a unique top-level key
            self.results["my_new_check"]["score"] = score
            self.results["my_new_check"]["suspicious"] = score > 0.7
            self.results["my_new_check"]["details"] = details
            self.results["my_new_check"]["enabled"] = True

        except Exception as e:
            logger.exception(f"[Task {task.id}]: MyNewCheck error: {e}")
            self.results["my_new_check"]["enabled"] = True
            self.results["my_new_check"]["error"] = str(e)

        return self.results
```

### Key Rules

1. **Unique result key** — Use a descriptive top-level key like
   `self.results["my_new_check"]`. Never collide with existing keys.
2. **Always return `self.results`** — Even on error. Other plugins and the
   confidence scorer will look for your key.
3. **Include `enabled` and `error` fields** — So the template can show
   "Unavailable" vs "Not configured" vs actual results.
4. **Order matters** — If you need metadata, use order > 10. If you need
   AI detection results, use order > 30. Stay under 90.
5. **Guard dependencies** — Use try/except on imports and check in
   `check_deps()`. If it returns False, the plugin is silently skipped.

## Step 2: Integrate with Confidence Scoring

The scoring engine lives in `lib/forensics/confidence.py`. It has three phases:

```
Phase 1: Forensic manipulation score (weighted aggregate)
          ↓
Phase 2: AI generation detection (override model)
          ↓
Phase 3: Final verdict (picks strongest signal)
```

### Deciding Where Your Plugin Fits

Ask yourself: **What question does my plugin answer?**

| Question | Phase | How it contributes |
|----------|-------|--------------------|
| "Has this image been edited/tampered with?" | Phase 1 | Weighted into `forensic_score` |
| "Was this image made by an AI?" | Phase 2 | Contributes to `ai_probability` |
| "Is there deterministic proof of origin?" | Phase 2 | Override (like metadata filename match) |

### Adding to Phase 1 (Manipulation Detection)

Most new forensic methods go here. You need to:

1. **Add a weight** to `forensic_weights`
2. **Read your plugin's output** from the results dict
3. **Normalize to 0-1** and apply the weight
4. **Append an indicator** for the evidence table

```python
# In lib/forensics/confidence.py, inside calculate_manipulation_confidence()

# PHASE 1 weights — must sum to ~1.0
forensic_weights = {
    'ela': 0.18,           # was 0.20
    'noise': 0.18,         # was 0.20
    'frequency': 0.12,     # was 0.15
    'metadata': 0.05,      # unchanged (weak signal)
    'opencv': 0.35,        # was 0.40
    'my_new_check': 0.12,  # NEW
}

# ... later in Phase 1 ...

# --- My New Check ---
if 'my_new_check' in results and results['my_new_check'].get('enabled', False):
    my_score = results['my_new_check'].get('score', 0)
    is_suspicious = results['my_new_check'].get('suspicious', False)

    if is_suspicious:
        # Normalize your score to 0-1 range
        normalized = min(my_score, 1.0)

        forensic_score += forensic_weights['my_new_check'] * normalized
        confidence['deterministic_methods']['my_new_check'] = normalized
        confidence['methods']['my_new_check'] = normalized
        confidence['indicators'].append({
            'method': 'My New Check (Deterministic)',
            'evidence': f"Suspicious pattern detected (score: {my_score:.2f})",
            'type': 'deterministic'  # or 'ai_ml' for ML-based methods
        })
```

### Adding to Phase 2 (AI Detection)

If your method detects AI generation specifically:

```python
# In Phase 2, after the existing ai_detection block

# --- My AI Detector ---
if 'my_ai_detector' in results and results['my_ai_detector'].get('enabled', False):
    my_ai_score = results['my_ai_detector'].get('ai_probability', 0)  # 0-100

    # If you have deterministic proof (not probabilistic):
    if results['my_ai_detector'].get('has_proof', False):
        ai_deterministic_proof = True
        ai_probability = max(ai_probability, max(my_ai_score, 95.0))
    else:
        # Probabilistic — blend with existing probability
        ai_probability = max(ai_probability, my_ai_score)
```

### Rebalancing Weights

When you add a new method, the existing weights must be rebalanced so they
still sum to approximately 1.0. The general principle:

```
Total weight budget = 1.0

Current allocation:
  ELA:        0.20  — classic forensic method, medium reliability
  Noise:      0.20  — good for manipulation, medium reliability
  Frequency:  0.15  — useful for AI detection, lower reliability alone
  Metadata:   0.05  — weak signal (easy to strip), keep low
  OpenCV:     0.40  — multiple sub-methods, highest reliability
              ----
              1.00
```

**Guidelines for setting your new weight:**

- **Strong, reliable method** (like OpenCV with multiple sub-checks): 0.15–0.25
- **Medium reliability** (like ELA or noise): 0.08–0.15
- **Weak/easily-fooled signal** (like metadata presence): 0.03–0.05

After adding your weight, reduce the others proportionally:

```python
# Example: adding a 0.12 weight for a medium-reliability method
# Scale down others by (1.0 - 0.12) / 1.0 = 0.88

forensic_weights = {
    'ela': 0.18,           # 0.20 × 0.88 ≈ 0.18
    'noise': 0.18,         # 0.20 × 0.88 ≈ 0.18
    'frequency': 0.13,     # 0.15 × 0.88 ≈ 0.13
    'metadata': 0.04,      # 0.05 × 0.88 ≈ 0.04
    'opencv': 0.35,        # 0.40 × 0.88 ≈ 0.35
    'my_new_check': 0.12,  # NEW
}                          # Total: 1.00
```

**Important**: Weights don't need to sum to exactly 1.0, but staying close
prevents the score from inflating or deflating. If they sum to 1.1, a
maximally suspicious image scores 110% which gets capped at 100%. If they
sum to 0.9, the maximum possible score is 90%.

### Understanding the Override Model

The scoring system uses two fundamentally different approaches:

**Weighted average** (Phase 1) — For manipulation detection where each method
provides partial evidence. No single method is conclusive. The weighted sum
of all method scores becomes the suspicion score.

**Override** (Phase 2) — For AI detection where some evidence is near-certain.
A filename containing `gemini_generated` is planted by the AI tool itself.
This can't be a mere 30% weight — it's proof. The override model says:
"If we have deterministic proof, the verdict is AI-generated regardless of
what the forensic score says."

**When to use which:**
- Your method produces a continuous score from 0-1 → Phase 1 weighted average
- Your method provides binary proof of AI origin → Phase 2 override
- Your method is a probabilistic AI classifier → Phase 2, blend with
  `ai_probability` using `max()`

### The Asymmetry Principle

**Presence of a signal is much stronger than absence of a signal.**

Example: If your plugin detects steganography watermarks from Midjourney,
finding one is strong evidence of AI generation. But NOT finding one means
nothing — most AI tools don't embed watermarks, and they can be stripped.

Apply this in your scoring:

```python
# GOOD: Presence = strong signal, absence = no signal
if watermark_found:
    ai_probability = max(ai_probability, 90.0)
# If not found, don't reduce ai_probability — absence proves nothing

# BAD: Treating absence as counter-evidence
if not watermark_found:
    ai_probability *= 0.5  # WRONG — penalizes for missing watermark
```

## Step 3: Add a Template Section (Optional)

If your plugin produces detailed results worth showing in their own section,
add it to `templates/analyses/report/_automated_analysis.html` in Section 4
(Method Details):

```html
<!-- 4c: My New Check -->
{% if analysis.report.my_new_check %}
<div class="row-fluid" style="margin-top: 15px;">
    <div class="span12">
        <div class="box">
            <div class="wdgt-header">
                <i class="icon-search"></i> My New Check
                <span class="pull-right">
                    {% if analysis.report.my_new_check.error %}
                        <span class="label label-inverse">Unavailable</span>
                    {% elif analysis.report.my_new_check.suspicious %}
                        <span class="label label-warning">Suspicious</span>
                    {% else %}
                        <span class="label label-success">Clean</span>
                    {% endif %}
                </span>
            </div>
            <div class="wdgt-body">
                {% if analysis.report.my_new_check.error %}
                    <div class="alert alert-warning" style="margin-bottom: 0;">
                        <strong>Not Available:</strong>
                        {{ analysis.report.my_new_check.error }}
                    </div>
                {% else %}
                    <p>Score: {{ analysis.report.my_new_check.score }}</p>
                    <p class="muted">{{ analysis.report.my_new_check.details }}</p>
                {% endif %}
            </div>
        </div>
    </div>
</div>
{% endif %}
```

Your plugin's findings will also automatically appear in the **Key Indicators**
evidence table (Section 3) via the `indicators` list — no template changes
needed for that.

## Step 4: Test

### Unit test the scoring

```python
from lib.forensics.confidence import calculate_manipulation_confidence

results = {
    'ela': {'max_difference': 25},
    'noise_analysis': {'inconsistency_score': 1.5, 'suspicious': False},
    'metadata': {'Exif': {'Image': {'Make': 'Canon'}}, 'Iptc': None, 'Xmp': None},
    # Add your plugin's output:
    'my_new_check': {
        'enabled': True,
        'score': 0.85,
        'suspicious': True,
        'details': 'Found suspicious pattern X',
    },
}

result = calculate_manipulation_confidence(results)
print(f"Verdict:    {result['verdict_label']}")
print(f"Certainty:  {result['verdict_confidence']}% ({result['verdict_certainty']})")
print(f"Suspicion:  {result['confidence_score']}%")
print(f"AI Prob:    {result['ai_generated_probability']}%")
for ind in result['indicators']:
    print(f"  [{ind['type']}] {ind['method']}: {ind['evidence']}")
```

### Re-score existing images after changing weights

```bash
.venv/bin/python -c "
import django, os
os.environ['DJANGO_SETTINGS_MODULE'] = 'sus_scrofa.settings'
django.setup()
from analyses.models import Analysis
from lib.forensics.confidence import calculate_manipulation_confidence

for a in Analysis.objects.all():
    if not a.report:
        continue
    a.report['confidence'] = calculate_manipulation_confidence(a.report)
    a.save()
print('Done')
"
```

### Clear bytecode cache after changes

```bash
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

Then restart the processor — it loads plugins at startup.

## Checklist

- [ ] Plugin file in `plugins/analyzer/` with unique filename
- [ ] Subclasses `BaseAnalyzerModule`
- [ ] Has `order` between 10–89
- [ ] `check_deps()` returns True/False based on imports
- [ ] `run(task)` writes to `self.results["your_key"]`
- [ ] `run(task)` always returns `self.results`
- [ ] Result includes `enabled` flag and `error` on failure
- [ ] Weight added to `forensic_weights` in `confidence.py` (Phase 1)
      **or** integrated into `ai_probability` logic (Phase 2)
- [ ] Existing weights rebalanced to sum ≈ 1.0
- [ ] Indicator appended to `confidence['indicators']` list
- [ ] Type label set: `'deterministic'` or `'ai_ml'`
- [ ] Tested with `calculate_manipulation_confidence()` directly
- [ ] Existing images re-scored after weight changes
- [ ] `__pycache__` cleared, processor restarted
