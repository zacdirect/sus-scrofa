# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
AI/ML analysis plugins — heavier detectors that benefit from GPU.

These plugins run as Phase 1b of the analysis pipeline, **only** when the
engine's auditor checkpoint (after Phase 1a static plugins) decides there
isn't enough evidence for a decisive verdict yet.

If the static evidence already proves the image is fake
(authenticity ≤ 30 with ≥ 2 findings), the engine orchestrator skips
this tier to save compute.  When the image looks authentic, AI/ML
plugins always run — passing static checks doesn't rule out a
sophisticated fake.

The engine orchestrator (``lib/analyzer/orchestrator.py``) discovers every
``BaseAnalyzerModule`` subclass in this package, sorts them by ``order``,
and runs them sequentially.

Current plugins (sorted by order):

=====  =========================  =========================
Order  Class                      Result key
=====  =========================  =========================
25     PhotoholmesDetector        ``photoholmes``
30     AIDetection                ``ai_detection``
40     OpenCVManipulation         ``opencv_manipulation``
=====  =========================  =========================

Adding a new AI/ML plugin
--------------------------
1. Create a file in this directory (``plugins/ai_ml/``).
2. Subclass ``BaseAnalyzerModule`` from ``lib.analyzer.base``.
3. Set ``order`` — lower runs earlier, must stay within 1–99.
4. Implement ``check_deps()`` → ``True`` / ``False``.
5. Implement ``run(task)`` → return ``self.results``.
6. If your plugin needs a containerised service (like OpenCV), document
   the startup command in ``check_deps()`` warnings.

The plugin will be auto-discovered on next restart.
"""
