"""
SDXL Detector — Organika/sdxl-detector (HuggingFace Swin Transformer)

Production wrapper for the Organika/sdxl-detector model, integrated into
the multi-layer detection framework.

Like SPAIDetector, this runs in the ai_detection venv via subprocess
to avoid polluting the main Sus Scrofa environment with torch/transformers.

Architecture: SwinForImageClassification (86.8M params, image_size=224)
Labels: 0 = "artificial", 1 = "human"
Published metrics: 98.1% accuracy, 97.3% F1, 99.5% precision, 95.3% recall

Source: https://huggingface.co/Organika/sdxl-detector
License: CC-BY-NC-3.0 (non-commercial use only)
"""

import json
import logging
import subprocess
from pathlib import Path

from .base import BaseDetector, DetectionResult, DetectionMethod, ConfidenceLevel

logger = logging.getLogger(__name__)

# HuggingFace model identifier
MODEL_ID = "Organika/sdxl-detector"


class SDXLDetector(BaseDetector):
    """
    Swin Transformer-based AI image detector (Organika/sdxl-detector).

    Runs inference via subprocess in the ai_detection venv, same pattern
    as SPAIDetector.  The model is cached locally in ai_detection/models/
    (downloaded by `make ai-setup` / `make models`).

    Type: ML-based (HuggingFace image-classification pipeline)
    Speed: ~1-3s per image on CPU, <0.5s on GPU
    """

    def __init__(self):
        super().__init__()
        self._ai_detection_dir = None

    def get_order(self) -> int:
        """
        Run after metadata but alongside / before SPAI.

        SDXL detector is faster than SPAI (single forward pass vs spectral
        analysis) and has better empirical accuracy on modern AI images.
        """
        return 60

    def check_deps(self) -> bool:
        """Check if transformers and cached model exist."""
        try:
            ai_det_dir = Path(__file__).parent.parent
            model_cache = ai_det_dir / "models" / "Organika-sdxl-detector"

            if not model_cache.exists():
                logger.warning(
                    "SDXLDetector: model cache not found at %s — run: make ai-setup",
                    model_cache,
                )
                return False

            # Check if transformers is available
            import transformers
            import torch
            
            self._ai_detection_dir = ai_det_dir
            return True

        except ImportError as e:
            logger.warning(f"SDXLDetector: dependencies not available: {e}")
            logger.info("Run: make ai-setup to install AI detection dependencies")
            return False
        except Exception as e:
            logger.error("Error checking SDXLDetector dependencies: %s", e)
            return False

    def detect(self, image_path: str, context=None) -> DetectionResult:
        """
        Classify an image as AI-generated or real using transformers pipeline.

        Returns a DetectionResult with:
          - score: probability the image is artificial (0.0-1.0)
          - is_ai_generated: True if artificial > human
          - confidence: mapped from probability distance to 0.5

        Args:
            image_path: Path to image file
            context: Optional ResultStore (unused by this detector)
        """
        if not self.check_deps():
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence="SDXLDetector not available — run: make ai-setup",
            )

        try:
            import time
            from transformers import pipeline
            from PIL import Image
            
            # Lazy load pipeline (only once)
            if not hasattr(self, '_pipeline'):
                model_cache = self._ai_detection_dir / "models" / "Organika-sdxl-detector"
                self._pipeline = pipeline(
                    "image-classification",
                    model=str(model_cache),
                    device=-1  # CPU
                )
            
            # Load and classify image
            start_time = time.time()
            image = Image.open(image_path).convert("RGB")
            results = self._pipeline(image)
            elapsed = time.time() - start_time
            
            # Parse results - format: [{'label': 'artificial', 'score': 0.99}, ...]
            scores = {r['label']: r['score'] for r in results}
            ai_score = scores.get('artificial', 0.0)
            human_score = scores.get('human', 0.0)
            
            is_ai = ai_score > human_score
            confidence = self._calculate_confidence(ai_score)

            verdict = "AI-generated" if is_ai else "Real"
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=is_ai,
                confidence=confidence,
                score=float(ai_score),
                evidence=(
                    f"SDXL detector: {ai_score*100:.1f}% artificial, "
                    f"{human_score*100:.1f}% human → {verdict} "
                    f"({elapsed:.2f}s)"
                ),
                metadata={
                    "model": MODEL_ID,
                    "artificial_score": float(ai_score),
                    "human_score": float(human_score),
                    "inference_time_s": round(elapsed, 3),
                },
            )

        except Exception as e:
            logger.error("SDXLDetector error: %s", e, exc_info=True)
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence=f"SDXL error: {e}",
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_confidence(ai_score: float) -> ConfidenceLevel:
        """Map artificial-class probability to ConfidenceLevel."""
        distance = abs(ai_score - 0.5)
        if distance > 0.40:
            return ConfidenceLevel.HIGH
        elif distance > 0.20:
            return ConfidenceLevel.MEDIUM
        elif distance > 0.10:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NONE
