# Engine Architecture

How SusScrofa processes an image from upload to verdict.

---

## High-Level Flow

```
Image upload
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                   Engine Orchestrator                        │
│              lib/analyzer/orchestrator.py                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1a ── Static plugins  (plugins/static/)               │
│     info, hash, mime, metadata, perceptual_hash,             │
│     ela, hashcomparer, previewcomparer, noise_analysis,      │
│     frequency_analysis, signatures                           │
│                                                              │
│  ── Auditor checkpoint ──                                    │
│     Run a preliminary audit on accumulated results.          │
│     If score ≤ 30 with ≥ 2 findings → skip 1b (already      │
│     failed).  Never skip when image looks authentic.         │
│                                                              │
│  Phase 1b ── AI/ML plugins  (plugins/ai_ml/)                 │
│     photoholmes, ai_detection, opencv_analysis,              │
│     opencv_manipulation                                      │
│     (only if the checkpoint says "continue")                 │
│                                                              │
│  Phase 2 ── Engine post-processing (NOT plugins)             │
│     a) Compliance Auditor  → results['audit']                │
│     b) Confidence Scoring  → results['confidence']           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
MongoDB (results)  +  Django ORM (task state)
```

---

## Key Modules

| File | Role |
|------|------|
| `lib/analyzer/orchestrator.py` | Discovers plugins, runs them in tier order, auditor checkpoint |
| `lib/analyzer/processing.py` | `AnalysisRunner` (worker process) and `AnalysisManager` (pool + queue) |
| `lib/analyzer/auditor.py` | Pure-function `audit(results)` — 13 checks, three-bucket scoring |
| `lib/analyzer/base.py` | `BaseAnalyzerModule` — base class every plugin inherits |
| `lib/forensics/confidence.py` | `calculate_manipulation_confidence(results)` — reads audit, builds verdict |

---

## Plugin Discovery

The orchestrator scans exactly two packages:

```
plugins/static/     → all .py files → tier = "static"
plugins/ai_ml/      → all .py files → tier = "ai_ml"
```

Within each tier, plugins are sorted by their `order` class attribute.
There is no cross-tier ordering — **all** static plugins run before
**any** AI/ML plugin.

Discovery happens once at startup in `AnalysisManager.__init__()`:

```python
static_raw, ai_ml_raw = discover_plugins()
self.static_plugins = check_deps(static_raw)
self.ai_ml_plugins  = check_deps(ai_ml_raw)
```

Plugins whose `check_deps()` returns `False` are silently removed.

---

## The Two-Tier Model

### Why two tiers?

Static plugins are fast (milliseconds) and deterministic — they examine
metadata, pixel statistics, hashes, and rule-based signatures.  AI/ML
plugins are slow (seconds to minutes), need GPU or external services,
and produce probabilistic results.

If the static tier already produces overwhelming evidence (e.g. EXIF
says "Made with Midjourney" and dimensions are 1024×1024), there's no
point running expensive ML inference.

### Auditor Checkpoint

After Phase 1a, the orchestrator calls `should_skip_ai_ml(results)`:

1. Run a preliminary `audit(results)` on the static-only data.
2. Check if `score <= 30` **and** `findings_count >= 2`.
3. If both conditions met → skip Phase 1b (already clearly fake).

The skip is **failure-only**.  When the image looks authentic after
static analysis, AI/ML plugins **always** run — a high static score
just means the image passed fast checks, but a sophisticated fake
could still fool metadata and ELA while getting caught by ML models.

A single metadata flag won't trigger a skip either — we need
convergent evidence from at least two independent checks.

If the checkpoint fails (exception), we **always continue** to AI/ML.

---

## Phase 2: Engine Post-Processing

After all plugins finish, two engine-internal steps run unconditionally:

### a) Compliance Auditor (`lib/analyzer/auditor.py`)

`audit(results)` is a pure function that reads the full accumulated
results dict and returns:

```python
{
    "authenticity_score": 0-100,       # 0 = fake, 100 = real
    "ai_probability": 0-100,           # AI generation likelihood
    "manipulation_probability": 0-100, # Traditional editing likelihood
    "findings_count": int,
    "findings_summary": [...],
    "evidence": {...},
    "detected_types": [...]
}
```

It runs 13 independent checks (metadata, dimensions, noise, frequency,
ELA, AI detector results, etc.) and aggregates them with score capping:
if contradicting evidence exists, the score never reaches 0 or 100.

### b) Confidence Scoring (`lib/forensics/confidence.py`)

`calculate_manipulation_confidence(results)` reads `results['audit']`
and builds the user-facing verdict:

```python
{
    "verdict": "fake" | "uncertain" | "real",
    "verdict_label": "Likely Fake",
    "verdict_confidence": 85,
    "confidence_score": 23,
    "indicators": [...]
}
```

This is what templates render.

---

## Data Flow Contract

Every plugin writes to the shared `results` dict under a unique key.
No plugin reads another plugin's key directly (they use `self.data` for
convenience, but the architecture doesn't mandate cross-plugin reads).

The **only** components that read across all keys are engine-internal:
- `audit()` reads everything to produce `results['audit']`
- `calculate_manipulation_confidence()` reads `results['audit']`

```
Plugin A → results["hash"]       ─┐
Plugin B → results["metadata"]    │
Plugin C → results["ela"]         ├─→ audit(results) → results["audit"]
Plugin D → results["ai_detection"]│                          │
...                               ─┘                         │
                                          ┌──────────────────┘
                                          ▼
                              calculate_manipulation_confidence()
                                          │
                                          ▼
                                  results["confidence"]
```

---

## Worker Architecture

`AnalysisManager` spawns a pool of `AnalysisRunner` worker processes
(one per CPU core, minus one for the DB).  SQLite forces parallelism
to 1.

Each worker receives the same `static_plugins` and `ai_ml_plugins`
lists (class references, not instances).  For each task, the worker
instantiates fresh plugin objects so there's no shared state.

```python
class AnalysisRunner(Process):
    def _process_image(self, task):
        results = {}
        results["file_data"] = task.image_id

        run_plugins(self.static_plugins, results, task)     # Phase 1a

        if self.ai_ml_plugins and should_skip_ai_ml(results):
            results["_engine"] = {"ai_ml_skipped": True, ...}
        else:
            run_plugins(self.ai_ml_plugins, results, task)  # Phase 1b

        results["audit"] = audit(results)                   # Phase 2a
        results["confidence"] = calculate_manipulation_confidence(results)  # Phase 2b

        save_results(results)                               # MongoDB
```

---

## Adding a New Plugin

See [Plugin Development Guide](PLUGIN_DEVELOPMENT.md) for the full
walkthrough.  The short version:

1. Decide the tier: is it fast & deterministic? → `plugins/static/`.
   Does it use ML or an external service? → `plugins/ai_ml/`.
2. Create a `.py` file in the chosen directory.
3. Subclass `BaseAnalyzerModule`, set `order`, implement `check_deps()`
   and `run(task)`.
4. Restart the processor.  Done.

---

## AI Detection Sub-Architecture

The `ai_detection` plugin in `plugins/ai_ml/` is itself an orchestrator
for multiple ML detectors (metadata heuristics, SDXL classifier, SPAI
spectral model).  See `ai_detection/ARCHITECTURE.md` for the inner
detector architecture — that's a layer below this document.

```
Engine Orchestrator
  └─ plugins/ai_ml/ai_detection.py (plugin)
       └─ ai_detection/detectors/orchestrator.py (inner detector runner)
            ├─ MetadataDetector
            ├─ SDXLDetector
            └─ SPAIDetector
```
