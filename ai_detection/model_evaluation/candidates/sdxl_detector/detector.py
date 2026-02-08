"""
SDXL Detector — Organika/sdxl-detector

A Swin Transformer fine-tuned on Wikimedia-SDXL image pairs for detecting
AI-generated images, especially those from modern diffusion models (SDXL, etc.)

Built on top of umm-maybe/AI-image-detector with improved performance on
recent diffusion model outputs and non-artistic imagery.

Architecture: SwinForImageClassification (86.8M params, image_size=224)
Labels: 0 = "artificial", 1 = "human"
Published metrics: 98.1% accuracy, 97.3% F1, 99.5% precision, 95.3% recall

Source: https://huggingface.co/Organika/sdxl-detector
License: CC-BY-NC-3.0 (non-commercial use only)
"""

import logging
import time
from pathlib import Path
from typing import Optional

try:
    from transformers import pipeline
    from PIL import Image
    import torch
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    _import_error = str(e)

# Import base classes from the detector framework
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ai_detection.detectors.base import (
    BaseDetector,
    DetectionResult,
    DetectionMethod,
    ConfidenceLevel
)

logger = logging.getLogger(__name__)

# HuggingFace model identifier
MODEL_ID = "Organika/sdxl-detector"

# Use the shared ai_detection/models/ cache (populated by `make ai-setup` / `make models`).
# Falls back to a local candidate weights/ dir if the shared cache doesn't exist.
_AI_DETECTION_DIR = Path(__file__).parent.parent.parent.parent
_SHARED_CACHE = _AI_DETECTION_DIR / "models" / "Organika-sdxl-detector"
_LOCAL_CACHE = Path(__file__).parent / "weights"
DEFAULT_CACHE_DIR = _SHARED_CACHE if _SHARED_CACHE.exists() else _LOCAL_CACHE


class SDXLDetector(BaseDetector):
    """
    Swin Transformer-based AI image detector (Organika/sdxl-detector).

    Fine-tuned from umm-maybe/AI-image-detector on Wikimedia-SDXL pairs.
    Outputs two classes: "artificial" (label 0) and "human" (label 1).

    References:
        - Model: https://huggingface.co/Organika/sdxl-detector
        - Base: https://huggingface.co/umm-maybe/AI-image-detector
        - Architecture: microsoft/swin-large-patch4-window7-224
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._pipe = None

        # Auto-detect device
        if device is None:
            self.device = "cuda" if (DEPS_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def get_order(self) -> int:
        """
        Run after metadata checks but alongside other ML models.
        Uses HuggingFace pipeline — moderate speed.
        """
        return 60

    def check_deps(self) -> bool:
        """Verify transformers + torch + PIL are importable."""
        if not DEPS_AVAILABLE:
            logger.warning(f"SDXLDetector deps missing: {_import_error}")
            return False
        return True

    def detect(self, image_path: str) -> DetectionResult:
        """
        Classify an image as AI-generated ("artificial") or real ("human").

        The HuggingFace pipeline handles:
          - Loading & resizing to 224x224
          - Swin-specific normalization
          - Softmax over the two logits

        Returns a DetectionResult with:
          - score: probability that the image is artificial (0.0-1.0)
          - is_ai_generated: True if artificial prob > 0.5
          - confidence: mapped from the probability distance to 0.5
        """
        if not self.check_deps():
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence="SDXLDetector dependencies not installed (need transformers, torch, Pillow)",
            )

        # Lazy-load model on first call
        if self._pipe is None:
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"SDXLDetector model load failed: {e}", exc_info=True)
                return DetectionResult(
                    method=DetectionMethod.ML_MODEL,
                    is_ai_generated=None,
                    confidence=ConfidenceLevel.NONE,
                    score=0.0,
                    evidence=f"Model load failed: {e}",
                )

        try:
            # Open image via PIL (the pipeline expects a PIL Image or path)
            image = Image.open(image_path).convert("RGB")

            start = time.time()
            outputs = self._pipe(image)  # list of {label, score} dicts
            elapsed = time.time() - start

            # Parse pipeline output
            # outputs looks like:
            #   [{'label': 'artificial', 'score': 0.98}, {'label': 'human', 'score': 0.02}]
            scores = {item["label"]: item["score"] for item in outputs}
            ai_score = scores.get("artificial", 0.0)
            human_score = scores.get("human", 0.0)

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
                    f"{human_score*100:.1f}% human -> {verdict} "
                    f"({elapsed:.2f}s)"
                ),
                metadata={
                    "model": self.model_id,
                    "artificial_score": float(ai_score),
                    "human_score": float(human_score),
                    "inference_time_s": round(elapsed, 3),
                    "device": self.device,
                },
            )

        except Exception as e:
            logger.error(f"SDXLDetector error: {e}", exc_info=True)
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence=f"Detection error: {e}",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Download (or load from cache) and build the classification pipeline."""
        logger.info(f"Loading {self.model_id} (cache: {self.cache_dir}, device: {self.device})")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._pipe = pipeline(
            "image-classification",
            model=self.model_id,
            device=self.device,
            model_kwargs={"cache_dir": str(self.cache_dir)},
        )
        logger.info("SDXLDetector model loaded successfully")

    def _calculate_confidence(self, ai_score: float) -> ConfidenceLevel:
        """
        Map the artificial-class probability to a ConfidenceLevel.

        The further from 0.5, the more confident we are.
        """
        distance = abs(ai_score - 0.5)

        if distance > 0.40:   # >0.90 or <0.10
            return ConfidenceLevel.HIGH
        elif distance > 0.20: # >0.70 or <0.30
            return ConfidenceLevel.MEDIUM
        elif distance > 0.10: # >0.60 or <0.40
            return ConfidenceLevel.LOW
        else:                 # 0.40-0.60 -- too close to call
            return ConfidenceLevel.NONE


# ======================================================================
# Standalone quick test
# ======================================================================
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path> [image_path2 ...]")
        sys.exit(1)

    detector = SDXLDetector()

    for path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"Image: {path}")
        print(f"{'='*60}")
        result = detector.detect(path)
        print(f"  AI-generated : {result.is_ai_generated}")
        print(f"  Confidence   : {result.confidence.value}")
        print(f"  Score        : {result.score:.4f}")
        print(f"  Evidence     : {result.evidence}")
        if result.metadata:
            print(f"  Inference    : {result.metadata.get('inference_time_s', '?')}s")
