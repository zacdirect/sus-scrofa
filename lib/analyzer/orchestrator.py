# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Engine Orchestrator — the top-level plugin runner.

Architecture::

    ┌────────────────────────────────────────────────────┐
    │              Engine Orchestrator                    │
    │  (lib/analyzer/orchestrator.py — this file)        │
    ├────────────────────────────────────────────────────┤
    │                                                    │
    │  Phase 1a: Static plugins   (plugins/static/)      │
    │     metadata, info, hash, mime, ELA, noise,        │
    │     frequency, signatures, perceptual hash, etc.   │
    │                                                    │
    │  ── Auditor checkpoint ──                          │
    │     If the auditor can already reach a CERTAIN     │
    │     verdict from static evidence alone (e.g. AI    │
    │     tags in EXIF, known AI filename), skip the     │
    │     expensive AI/ML phase entirely.                │
    │                                                    │
    │  Phase 1b: AI/ML plugins   (plugins/ai_ml/)        │
    │     ai_detection (SDXL, SPAI, metadata detector),  │
    │     opencv_manipulation, photoholmes, etc.          │
    │                                                    │
    │  ── Research gate ──                               │
    │     If the image is likely NOT genuine after 1a+1b │
    │     (authenticity score < threshold), skip the     │
    │     research phase — no point cataloguing person   │
    │     attributes on a fabricated image.              │
    │                                                    │
    │  Phase 1c: Research plugins (plugins/research/)    │
    │     Content analysis, object detection, person     │
    │     attributes.  Only runs on likely-genuine       │
    │     images (gated by confidence threshold).        │
    │     Minimal auditor impact (LOW findings only).    │
    │                                                    │
    │  Phase 2: Engine post-processing (NOT plugins)     │
    │     a) Compliance Auditor  → results['audit']      │
    │     b) Confidence Scoring  → results['confidence'] │
    │                                                    │
    └────────────────────────────────────────────────────┘

The directory a plugin lives in determines its tier::

    plugins/static/my_plugin.py     → static tier
    plugins/ai_ml/my_plugin.py      → AI/ML tier
    plugins/research/my_plugin.py   → research tier

Within each tier, plugins are sorted by their ``order`` class attribute.

The engine orchestrator:
  1. Discovers plugins from plugins/static/, plugins/ai_ml/, and plugins/research/
  2. Runs all static plugins (sorted by order)
  3. Asks the auditor: "Is it worth running the expensive stuff?"
  4. If yes, runs all AI/ML plugins (sorted by order)
  5. Asks the auditor: "Is the image likely genuine?" (research gate)
  6. If yes, runs research plugins (sorted by order)
  7. Runs Phase 2 engine post-processing (auditor + confidence)

This means:
  - Plugin ordering is only relevant WITHIN a tier, never across tiers
  - No plugin needs to know what tier another plugin is in
  - The AI/ML checkpoint gates Phase 1b (skip when already clearly fake)
  - The research gate gates Phase 1c (skip when NOT genuine)
  - Adding a new plugin = drop a file in the right folder
"""

import logging
import pkgutil
import inspect
from typing import List, Tuple, Type

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import AutoVivification

logger = logging.getLogger(__name__)

# The three tier identifiers.  Directory determines tier.
TIER_STATIC = "static"
TIER_AI_ML = "ai_ml"
TIER_RESEARCH = "research"

# Auditor threshold: if the auditor produces a score this far from
# the midpoint (50), skip the AI/ML tier.  20 means ≤30 or ≥70.
EARLY_VERDICT_MARGIN = 30


def discover_plugins() -> Tuple[List[Type[BaseAnalyzerModule]],
                                List[Type[BaseAnalyzerModule]],
                                List[Type[BaseAnalyzerModule]]]:
    """
    Discover and categorize all analyzer plugins.

    Scans ``plugins/static/``, ``plugins/ai_ml/``, and ``plugins/research/``
    for subclasses of :class:`BaseAnalyzerModule`.  The directory a plugin
    lives in determines its tier — no class attribute needed.

    Returns:
        (static_plugins, ai_ml_plugins, research_plugins) — each list
        sorted by ``order``.
    """
    static: List[Type[BaseAnalyzerModule]] = []
    ai_ml: List[Type[BaseAnalyzerModule]] = []
    research: List[Type[BaseAnalyzerModule]] = []

    tier_packages: List[Tuple] = []

    try:
        import plugins.static as static_pkg
        tier_packages.append((static_pkg, TIER_STATIC))
    except ImportError:
        logger.warning("plugins.static package not found — no static plugins will load")

    try:
        import plugins.ai_ml as ai_ml_pkg
        tier_packages.append((ai_ml_pkg, TIER_AI_ML))
    except ImportError:
        logger.warning("plugins.ai_ml package not found — no AI/ML plugins will load")

    try:
        import plugins.research as research_pkg
        tier_packages.append((research_pkg, TIER_RESEARCH))
    except ImportError:
        logger.warning("plugins.research package not found — no research plugins will load")

    tier_buckets = {
        TIER_STATIC: static,
        TIER_AI_ML: ai_ml,
        TIER_RESEARCH: research,
    }

    for pkg, tier in tier_packages:
        for _loader, module_name, is_pkg in pkgutil.iter_modules(
                pkg.__path__, pkg.__name__ + "."):
            if is_pkg:
                continue
            try:
                module = __import__(module_name, globals(), locals(), ["dummy"], 0)
            except ImportError as exc:
                logger.warning("Unable to import plugin %s: %s", module_name, exc)
                continue
            except Exception as exc:
                logger.error("Error loading plugin %s: %s", module_name, exc)
                continue

            for _cls_name, cls in inspect.getmembers(module, inspect.isclass):
                if not issubclass(cls, BaseAnalyzerModule) or cls is BaseAnalyzerModule:
                    continue
                tier_buckets[tier].append(cls)
                logger.debug("Discovered plugin %s (tier=%s, order=%s)",
                             cls.__name__, tier, cls.order)

    static.sort(key=lambda c: c.order)
    ai_ml.sort(key=lambda c: c.order)
    research.sort(key=lambda c: c.order)
    return static, ai_ml, research


def check_deps(plugins: List[Type[BaseAnalyzerModule]]) -> List[Type[BaseAnalyzerModule]]:
    """Remove plugins whose dependencies aren't met."""
    available = []
    for cls in plugins:
        try:
            if cls().check_deps():
                available.append(cls)
            else:
                logger.warning("Kicked plugin (deps not met): %s", cls.__name__)
        except Exception as exc:
            logger.warning("Kicked plugin (deps check failed): %s — %s",
                           cls.__name__, exc)
    return available


def should_skip_ai_ml(results: dict) -> bool:
    """
    Ask the auditor whether static evidence already proves the image is fake.

    Runs a *lightweight* audit on the results accumulated so far.
    If the score is low enough (image already clearly failing), we can
    skip the expensive AI/ML tier — there's no point spending GPU time
    confirming what metadata and pixel analysis already proved.

    We NEVER skip when the image looks authentic.  A high static score
    just means the image *passed* fast checks — ML detectors might still
    catch a sophisticated fake that metadata and ELA missed.

    This is purely an optimisation — if we skip, the full audit in
    Phase 2 will still run on whatever data we have.
    """
    from lib.analyzer.auditor import audit  # deferred to avoid circular import

    try:
        preliminary = audit(results)
        score = preliminary.get("authenticity_score", 50)
        findings_count = preliminary.get("findings_count", 0)

        # Only skip if the image is already clearly FAILING and we have
        # convergent evidence from multiple static checks.
        if findings_count >= 2 and score <= (50 - EARLY_VERDICT_MARGIN):
            logger.info(
                "Auditor checkpoint: score %d/100 (%d findings) — "
                "SKIPPING AI/ML tier (already failed on static evidence)",
                score, findings_count)
            return True

        logger.info(
            "Auditor checkpoint: score %d/100 (%d findings) — "
            "CONTINUING to AI/ML tier",
            score, findings_count)
        return False

    except Exception as exc:
        logger.warning("Auditor checkpoint failed (%s) — continuing to AI/ML tier", exc)
        return False


def should_skip_research(results: dict) -> bool:
    """
    Gate the research tier based on image authenticity.

    After Phases 1a and 1b, run a preliminary audit.  If the
    authenticity score falls BELOW ``RESEARCH_CONFIDENCE_THRESHOLD``
    the image is likely fake (AI-generated or heavily manipulated)
    and there is no forensic value in cataloguing person attributes,
    so we skip the research tier entirely.

    This is the **inverse** of ``should_skip_ai_ml``:
      - AI/ML is skipped when the image is already clearly FAKE
        (low score + convergent evidence — save GPU time).
      - Research is skipped when the image is NOT genuine enough
        (score below threshold — no point profiling a fake person).

    The threshold is configurable via ``RESEARCH_CONFIDENCE_THRESHOLD``
    in settings (default: 40).
    """
    from lib.analyzer.auditor import audit  # deferred to avoid circular import
    from django.conf import settings

    threshold = getattr(settings, 'RESEARCH_CONFIDENCE_THRESHOLD', 40)

    try:
        preliminary = audit(results)
        score = preliminary.get("authenticity_score", 50)

        if score < threshold:
            logger.info(
                "Research gate: score %d/100 (threshold=%d) — "
                "SKIPPING research tier (image likely not genuine)",
                score, threshold)
            return True

        logger.info(
            "Research gate: score %d/100 (threshold=%d) — "
            "CONTINUING to research tier",
            score, threshold)
        return False

    except Exception as exc:
        logger.warning("Research gate failed (%s) — continuing to research tier", exc)
        return False


def run_plugins(plugins: List[Type[BaseAnalyzerModule]], results: dict,
                task) -> dict:
    """
    Run a list of plugins in order, accumulating results.

    Each plugin receives the full accumulated results dict via
    ``plugin.data`` and writes its output back.

    Args:
        plugins: sorted list of plugin classes
        results: mutable results dict (modified in place)
        task: the analysis task

    Returns:
        The same ``results`` dict (for convenience).
    """
    for plugin_cls in plugins:
        instance = plugin_cls()
        instance.data = results
        try:
            output = instance.run(task)
        except Exception as exc:
            logger.exception("Plugin %s failed, skipping: %s",
                             plugin_cls.__name__, exc)
            continue

        if isinstance(output, (dict, AutoVivification)):
            results.update(output)
        else:
            logger.warning("Plugin %s returned non-dict results",
                           plugin_cls.__name__)

    return results
