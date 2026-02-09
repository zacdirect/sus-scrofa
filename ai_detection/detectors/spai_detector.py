"""
SPAI ML Model Detector.

Wrapper around the SPAI spectral learning model for integration
into the multi-layer detection framework.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .base import BaseDetector, DetectionResult, DetectionMethod, ConfidenceLevel


logger = logging.getLogger(__name__)


class SPAIDetector(BaseDetector):
    """
    SPAI (Spectral AI-Generated Image Detector) wrapper.
    
    Uses frequency domain analysis to detect AI-generated images.
    This is slower than metadata checks but works when metadata is stripped.
    """
    
    def __init__(self):
        super().__init__()
        self._spai = None
        self._ai_detection_dir = None
    
    def get_order(self) -> int:
        """Run last - slowest method."""
        return 100
    
    def check_deps(self) -> bool:
        """Check if SPAI environment is available."""
        # Check for ai_detection directory and venv
        try:
            ai_det_dir = Path(__file__).parent.parent
            venv_python = ai_det_dir / '.venv' / 'bin' / 'python'
            
            if not venv_python.exists():
                logger.warning("SPAI venv not found - run: make ai-setup")
                return False
            
            self._ai_detection_dir = ai_det_dir
            return True
            
        except Exception as e:
            logger.error(f"Error checking SPAI dependencies: {e}")
            return False
    
    def detect(self, image_path: str, context=None) -> DetectionResult:
        """
        Analyze image using SPAI model.
        
        Args:
            image_path: Path to image file
            context: Optional ResultStore (unused by this detector)
            
        Returns:
            DetectionResult with ML model verdict
        """
        if not self.check_deps():
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence="SPAI not available - run: make ai-setup"
            )
        
        try:
            # Import SPAI from isolated environment
            # Note: This runs in the main process but SPAI is in its own venv
            # We use subprocess approach from the plugin
            import subprocess
            import json
            
            venv_python = self._ai_detection_dir / '.venv' / 'bin' / 'python'
            infer_script = self._ai_detection_dir / 'spai_infer.py'
            weights_path = self._ai_detection_dir / 'weights' / 'spai.pth'
            
            # No timeout - let model finish inference once loaded
            # CPU: ~60s model load + ~30-60s inference = ~90-120s total
            # GPU: ~3-5s total
            logger.info("Starting SPAI inference (may take 1-2 minutes on CPU)...")
            result = subprocess.run(
                [str(venv_python), str(infer_script), str(weights_path), image_path],
                capture_output=True,
                text=True,
                cwd=str(self._ai_detection_dir)
            )
            
            if result.returncode != 0:
                return DetectionResult(
                    method=DetectionMethod.ML_MODEL,
                    is_ai_generated=None,
                    confidence=ConfidenceLevel.NONE,
                    score=0.0,
                    evidence=f"SPAI inference failed: {result.stderr[:200]}"
                )
            
            # Parse JSON result
            inference_result = json.loads(result.stdout)
            
            if not inference_result.get('success', False):
                error_msg = inference_result.get('error', 'Unknown error')
                return DetectionResult(
                    method=DetectionMethod.ML_MODEL,
                    is_ai_generated=None,
                    confidence=ConfidenceLevel.NONE,
                    score=0.0,
                    evidence=f"SPAI error: {error_msg}"
                )
            # Extract results
            probability = inference_result['score']
            logit = inference_result['logit']
            timing = inference_result.get('timing', {})
            
            if timing:
                logger.info(f"SPAI completed: load={timing.get('load_time')}s, "
                          f"inference={timing.get('inference_time')}s, "
                          f"total={timing.get('total_time')}s")
            
            # Map probability to confidence level
            # Map probability to confidence level
            # Note: SPAI tends to produce very low probabilities for AI images
            # in our tests, so we use logit for better discrimination
            if probability > 0.7:
                confidence = ConfidenceLevel.HIGH
            elif probability > 0.5:
                confidence = ConfidenceLevel.MEDIUM
            elif logit > -35:  # Relative suspicion threshold
                confidence = ConfidenceLevel.LOW
            else:
                confidence = ConfidenceLevel.NONE
            
            is_ai = probability > 0.5
            
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=is_ai,
                confidence=confidence,
                score=float(probability),
                evidence=f"SPAI spectral analysis: {probability*100:.4f}% AI probability (logit: {logit:.2f})",
                metadata={
                    'logit': logit,
                    'timing': timing
                }
            )
            
        except Exception as e:
            logger.error(f"SPAI detection error: {e}")
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence=f"SPAI error: {str(e)}"
            )
