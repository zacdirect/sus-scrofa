# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Static analysis plugins — fast, deterministic analysers.

These plugins run first in every analysis.  They examine intrinsic image
properties (metadata, hashes, pixel statistics, frequency response, ELA,
noise patterns, signatures) without any ML model inference.

The engine orchestrator (``lib/analyzer/orchestrator.py``) discovers every
``BaseAnalyzerModule`` subclass in this package, sorts them by ``order``,
and runs them as Phase 1a of the analysis pipeline.

After all static plugins complete, the engine's auditor checkpoint decides
whether Phase 1b (AI/ML plugins) is necessary.

Current plugins (sorted by order):

=====  ========================  =========================
Order  Class                     Result key
=====  ========================  =========================
10     InfoAnalyzer              ``file_name``, ``file_size``
10     HashAnalyzer              ``hash``
10     MimeAnalyzer              ``mime_type``, ``file_type``
10     MetadataModernAnalyzer    ``metadata``
15     PerceptualHashAnalyzer    ``perceptual_hash``
20     ElaAnalyzer               ``ela``
20     HashComparerAnalyzer      *(ORM side-effect)*
20     PreviewComparerAnalyzer   *(mutates metadata)*
25     NoiseAnalysisProcessing   ``noise_analysis``
26     FrequencyAnalysisProc.    ``frequency_analysis``
80     SignatureAnalyzer         ``signatures``
=====  ========================  =========================

Adding a new static plugin
--------------------------
1. Create a file in this directory (``plugins/static/``).
2. Subclass ``BaseAnalyzerModule`` from ``lib.analyzer.base``.
3. Set ``order`` — lower runs earlier, must stay within 1–99.
4. Implement ``check_deps()`` → ``True`` / ``False``.
5. Implement ``run(task)`` → return ``self.results``.

The plugin will be auto-discovered on next restart.
"""
