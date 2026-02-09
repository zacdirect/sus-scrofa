# Plugin Development Guide

How to add new analysis plugins to SusScrofa.

---

## Architecture Overview

```
Image uploaded
  │
  ▼
Engine Orchestrator (lib/analyzer/orchestrator.py)
  │
  ├─ Phase 1a: Static plugins  (plugins/static/)
  │     Sorted by `order`, run sequentially.
  │     Each plugin gets accumulated results via self.data,
  │     writes its own output via self.results.
  │
  ├─ Auditor checkpoint
  │     If static evidence is already decisive → skip Phase 1b.
  │
  ├─ Phase 1b: AI/ML plugins  (plugins/ai_ml/)
  │     Same interface, only runs when needed.
  │
  └─ Phase 2: Engine post-processing (NOT plugins)
        a) Compliance Auditor  → results['audit']
        b) Confidence Scoring  → results['confidence']
```

Plugins produce raw findings.  The auditor and confidence scorer
(engine internals) interpret them.  Plugins never decide "fake or real".

---

## Plugin Inventory

### Static plugins — `plugins/static/`

### AI/ML plugins — `plugins/ai_ml/`


---

## Step 1: Pick Your Tier

| Question | Tier | Directory |
|----------|------|-----------|
| Is it fast and deterministic (metadata, hashes, pixel math)? | Static | `plugins/static/` |
| Does it use ML models, GPU, or an external service? | AI/ML | `plugins/ai_ml/` |

**Rule of thumb**: if it runs in under 100 ms on a CPU with no network
calls, it's static.

The tier matters because:
- All static plugins **always** run.
- AI/ML plugins are **skipped** when static evidence alone is decisive.
- Within each tier, `order` controls execution sequence.

---

## Step 2: Create the Plugin

Create a `.py` file in the appropriate tier directory.

```python
# plugins/static/my_check.py          (or plugins/ai_ml/my_check.py)

import logging
from lib.analyzer.base import BaseAnalyzerModule

logger = logging.getLogger(__name__)

# Guard optional imports
try:
    import some_library
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class MyCheckAnalyzer(BaseAnalyzerModule):
    """One-line description of what this plugin detects."""

    order = 35  # Within-tier ordering (1–99)

    def check_deps(self):
        """Return True if all dependencies are available."""
        return HAS_DEPS

    def run(self, task):
        """
        Perform analysis.

        Args:
            task: Analysis model instance.
                  task.id            — database ID
                  task.file_name     — original filename
                  task.get_file_data — raw file bytes
                  task.image_id      — GridFS file ID

        Returns:
            self.results (AutoVivification dict).
        """
        try:
            raw_bytes = task.get_file_data

            score, details = some_library.analyze(raw_bytes)

            self.results["my_check"]["score"] = score
            self.results["my_check"]["suspicious"] = score > 0.7
            self.results["my_check"]["details"] = details
            self.results["my_check"]["enabled"] = True

        except Exception as e:
            logger.exception("[Task %s]: MyCheck error: %s", task.id, e)
            self.results["my_check"]["enabled"] = True
            self.results["my_check"]["error"] = str(e)

        return self.results
```

### Key Rules

1. **Unique result key** — use a descriptive top-level key like
   `self.results["my_check"]`.  Never collide with existing keys.
2. **Always return `self.results`** — even on error.  The auditor
   looks for your key.
3. **Include `enabled` and `error` fields** — templates use these to
   show "Unavailable" vs actual results.
4. **Guard dependencies** — try/except on imports + `check_deps()`.
   If it returns `False`, the plugin is silently skipped at startup.
5. **No scoring or verdicts** — raw findings only.  The compliance
   auditor (`lib/analyzer/auditor.py`) handles interpretation.

---

## Step 3: Teach the Auditor (if needed)

If your plugin produces evidence the auditor should consider, add a
check to `lib/analyzer/auditor.py`.  The auditor is a pure function
with 13 independent checks.  Each check:

1. Reads a specific key from the results dict.
2. Produces a `Finding` (risk level + score impact + description).
3. Appends it to the evidence list.

Example: adding a check for your plugin's output:

```python
# In lib/analyzer/auditor.py, inside audit()

# ── My Check ──────────────────────────────────────────
my_check = results.get("my_check", {})
if my_check.get("enabled") and my_check.get("suspicious"):
    score_val = my_check.get("score", 0)
    if score_val > 0.9:
        findings.append(Finding(
            RISK_HIGH, -50,
            "my_check",
            f"Strong suspicious pattern (score: {score_val:.2f})"
        ))
    elif score_val > 0.7:
        findings.append(Finding(
            RISK_MEDIUM, -30,
            "my_check",
            f"Moderate suspicious pattern (score: {score_val:.2f})"
        ))
```

The auditor aggregates all findings into three buckets:

| Bucket | Field | Meaning |
|--------|-------|---------|
| 1 | `authenticity_score` | 0 = fake, 100 = real |
| 2 | `ai_probability` | AI generation likelihood |
| 3 | `manipulation_probability` | Traditional editing likelihood |

Score impacts are additive from a base of 50.  Score capping ensures
the result never hits 0 or 100 when contradicting evidence exists.

---

## Step 4: Add a Template Section (Optional)

If your plugin produces detailed results worth their own UI section,
add a block to `templates/analyses/report/_automated_analysis.html`:

```html
{% if analysis.report.my_check %}
<div class="row-fluid" style="margin-top: 15px;">
  <div class="span12">
    <div class="box">
      <div class="wdgt-header">
        <i class="icon-search"></i> My Check
        <span class="pull-right">
          {% if analysis.report.my_check.error %}
            <span class="label label-inverse">Unavailable</span>
          {% elif analysis.report.my_check.suspicious %}
            <span class="label label-warning">Suspicious</span>
          {% else %}
            <span class="label label-success">Clean</span>
          {% endif %}
        </span>
      </div>
      <div class="wdgt-body">
        {% if analysis.report.my_check.error %}
          <div class="alert alert-warning">
            <strong>Not Available:</strong>
            {{ analysis.report.my_check.error }}
          </div>
        {% else %}
          <p>Score: {{ analysis.report.my_check.score }}</p>
          <p class="muted">{{ analysis.report.my_check.details }}</p>
        {% endif %}
      </div>
    </div>
  </div>
</div>
{% endif %}
```

---

## Step 5: Test

### Verify the plugin loads

```bash
.venv/bin/python manage.py test tests.test_processing -v2 2>&1 | grep -i "my_check\|Loaded"
```

You should see your plugin in the discovery log.

### Unit test the auditor integration

```python
from lib.analyzer.auditor import audit

results = {
    "metadata": {"dimensions": [1920, 1080]},
    "noise_analysis": {"inconsistency_score": 5.5},
    "my_check": {
        "enabled": True,
        "score": 0.85,
        "suspicious": True,
    },
}

verdict = audit(results)
print(f"Authenticity: {verdict['authenticity_score']}/100")
print(f"Findings: {verdict['findings_count']}")
for f in verdict["findings_summary"]:
    print(f"  [{f['risk']}] {f['description']}")
```

### Clear cache and restart

```bash
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
# Then restart the processor — plugins load at startup.
```

---

## Checklist

- [ ] File in `plugins/static/` or `plugins/ai_ml/`
- [ ] Subclasses `BaseAnalyzerModule`
- [ ] `order` set (1–99, within tier)
- [ ] `check_deps()` → `True` / `False`
- [ ] `run(task)` writes to `self.results["your_key"]`
- [ ] `run(task)` always returns `self.results`
- [ ] Result includes `enabled` flag and `error` on failure
- [ ] Auditor check added in `lib/analyzer/auditor.py` (if applicable)
- [ ] Tested with `audit()` directly
- [ ] `__pycache__` cleared, processor restarted
