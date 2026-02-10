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
        self._weights_path = None
    
    def get_order(self) -> int:
        """Run last - slowest method."""
        return 100
    
    def check_deps(self) -> bool:
        """Check if SPAI dependencies and weights are available."""
        try:
            # Check for weights file
            ai_det_dir = Path(__file__).parent.parent
            weights_path = ai_det_dir / 'weights' / 'spai.pth'
            
            if not weights_path.exists():
                logger.warning(f"SPAI model weights not found at {weights_path} — run: make ai-setup")
                return False
            
            # Try importing SPAI dependencies
            import torch
            import torchvision
            
            # Add SPAI module to path if not already there
            spai_dir = ai_det_dir / 'spai'
            if str(spai_dir) not in sys.path:
                sys.path.insert(0, str(spai_dir))
            
            self._weights_path = weights_path
            return True
            
        except ImportError as e:
            logger.warning(f"SPAI dependencies not available: {e}")
            logger.info("Run: make ai-setup to install AI detection dependencies")
            return False
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
            # Lazy import SPAI (only when needed)
            if self._spai is None:
                from spai.inference import SPAIDetector as SPAI
                self._spai = SPAI(str(self._weights_path))
            
            # Run inference
            result = self._spai.predict(image_path)
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                return DetectionResult(
                    method=DetectionMethod.ML_MODEL,
                    is_ai_generated=None,
                    confidence=ConfidenceLevel.NONE,
                    score=0.0,
                    evidence=f"SPAI error: {error_msg}"
                )
            
            # Extract results
            probability = result['score']
            logit = result['logit']
            
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
                    'model': 'SPAI',
                    'logit': logit,
                    'probability': probability
                }
            )
            
        except Exception as e:
            logger.error(f"SPAI detection error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence=f"SPAI error: {str(e)}"
            )

