"""
Multi-layer AI detection orchestrator.

Runs multiple ML/heuristic detection methods in sequence and returns
raw per-detector results.  This module does **not** score, audit, or
produce verdicts — those responsibilities belong to the engine-level
compliance auditor (``lib/analyzer/auditor.py``) which reads the
accumulated results dict after *all* plugins have run.

Early-stop logic is retained as a performance optimisation: if a
detector returns CERTAIN confidence we skip the slower models.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

from .base import BaseDetector, DetectionResult, ConfidenceLevel, ResultStore
from .metadata import MetadataDetector
from .spai_detector import SPAIDetector
from .sdxl_detector import SDXLDetector


logger = logging.getLogger(__name__)


class MultiLayerDetector:
    """
    Orchestrates multiple AI detection methods.

    Responsibilities:
    - Register available detectors (fast → slow)
    - Run each detector, record results in a shared ResultStore
    - Optionally stop early when a detector returns CERTAIN confidence
    - Return raw per-detector results (no scoring, no audit)

    The engine-level auditor (``lib/analyzer/auditor.py``) is a peer of
    the processing pipeline, not part of this orchestrator.
    """

    def __init__(self, enable_ml: bool = True):
        """
        Initialise orchestrator with detectors.

        Args:
            enable_ml: Whether to enable ML model detection (slower)
        """
        self.detectors: List[BaseDetector] = []

        # Register detectors in priority order (fast → slow)
        self._register_detector(MetadataDetector())

        if enable_ml:
            self._register_detector(SDXLDetector())
            self._register_detector(SPAIDetector())

        logger.info(f"Orchestrator initialized with {len(self.detectors)} detectors")

    def _register_detector(self, detector: BaseDetector):
        """Register a detector if its dependencies are available."""
        if detector.check_deps():
            self.detectors.append(detector)
            logger.info(f"Registered detector: {detector.name}")
        else:
            logger.warning(f"Detector {detector.name} disabled - dependencies not met")

    def detect(self, image_path: str, early_stop: bool = True,
               original_filename: str = None) -> Dict:
        """
        Run multi-layer detection on an image.

        Args:
            image_path: Path to image file (may be temporary file)
            early_stop: Stop at first CERTAIN-confidence result
            original_filename: Original filename (for pattern matching
                when using temp files)

        Returns:
            Dictionary with raw results::

                {
                    'detection_layers': [  # one dict per detector that ran
                        {
                            'method': str,
                            'evidence': str,
                            'verdict': 'AI' | 'Real' | 'Unknown',
                            'confidence': str,   # CERTAIN/HIGH/MEDIUM/LOW/NONE
                            'score': float,       # 0.0-1.0
                        },
                        ...
                    ],
                    'enabled': True,
                    'methods_run': [str, ...],
                }
        """
        if not Path(image_path).exists():
            return self._error_result("Image file not found")

        store = ResultStore()
        sorted_detectors = sorted(self.detectors, key=lambda d: d.get_order())

        for detector in sorted_detectors:
            try:
                if detector.name == 'MetadataDetector' and original_filename:
                    result = detector.detect(
                        image_path, original_filename=original_filename,
                        context=store)
                else:
                    result = detector.detect(image_path, context=store)

                store.record(detector.name, result)
                logger.debug(f"{detector.name}: {result.evidence}")

                # Performance optimisation: skip slow models when we already
                # have a CERTAIN result (e.g. definitive metadata match).
                if early_stop and self._should_stop_early(store.get_all()):
                    logger.info(
                        f"Early stop after {detector.name} — CERTAIN result")
                    break

            except Exception as e:
                logger.error(f"Error running {detector.name}: {e}")
                continue

        return self._build_result(store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_stop_early(results: List[DetectionResult]) -> bool:
        """Stop early if any detector returned CERTAIN confidence."""
        return any(
            r.confidence == ConfidenceLevel.CERTAIN
            for r in results
            if r.confidence is not None
        )

    @staticmethod
    def _build_result(store: ResultStore) -> Dict:
        """Package raw detector results for storage in the results dict."""
        layers = []
        for entry in store.get_all():
            layer = {
                'method': entry.method.value,
                'evidence': entry.evidence,
            }
            if entry.confidence is not None:
                layer['confidence'] = entry.confidence.name
            if entry.score is not None:
                layer['score'] = entry.score
            if entry.is_ai_generated is not None:
                layer['verdict'] = (
                    'AI' if entry.is_ai_generated
                    else 'Real' if entry.is_ai_generated is False
                    else 'Unknown')
            layers.append(layer)

        return {
            'detection_layers': layers,
            'methods_run': store.names(),
            'enabled': True,
        }

    @staticmethod
    def _error_result(error_msg: str) -> Dict:
        """Return error result."""
        return {
            'detection_layers': [],
            'methods_run': [],
            'enabled': False,
            'error': error_msg,
        }
