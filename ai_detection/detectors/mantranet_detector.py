"""
ManTraNet Detector — Image Forgery Localization

Implements manipulation detection using ManTraNet (CVPR 2019), which produces
pixel-level forgery masks showing WHERE in an image manipulation occurred.

Unlike binary classifiers (SDXL/SPAI), ManTraNet outputs a spatial heatmap
identifying manipulated regions (clone tool, splicing, copy-move, etc).

Architecture: VGG-style convolutional network with custom Bayar convolution layers
Input: RGB image (any size)
Output: Single-channel manipulation mask (0=pristine, 1=forged)

Paper: "ManTra-Net: Manipulation Tracing Network for Detection and Localization
        of Image Forgeries With Anomalous Features" (CVPR 2019)
        https://arxiv.org/abs/1812.08045

Model: Modern TensorFlow 2.20/Keras 3.x implementation, weights from official pretrained model
"""

import json
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

from .base import BaseDetector, DetectionResult, DetectionMethod, ConfidenceLevel

logger = logging.getLogger(__name__)


class ManTraNetDetector(BaseDetector):
    """
    Pixel-level forgery localization using ManTraNet.
    
    Returns both a DetectionResult (for auditor integration) and saves
    the forgery mask to GridFS for visualization in the UI.
    
    The mask is analyzed to generate audit findings based on the percentage
    of the image that appears manipulated:
    - HIGH (>20% manipulated): Strong evidence of forgery
    - MEDIUM (5-20% manipulated): Moderate manipulation detected
    - LOW (1-5% manipulated): Minor manipulation detected
    - POSITIVE (<0.5% manipulated): Likely pristine image
    """
    
    def __init__(self):
        super().__init__()
        self._ai_detection_dir = None
        self._mantranet_env = None
        
    def get_order(self) -> int:
        """
        Run after AI detection (SDXL/SPAI) since it's slower.
        
        ManTraNet is complementary to AI detection:
        - AI detectors: "Is this AI-generated?"
        - ManTraNet: "What parts were manipulated?"
        """
        return 80
    
    def check_deps(self) -> bool:
        """Check if ManTraNet model and architecture files exist."""
        try:
            # Model in project root models/weights/
            model_path = Path(__file__).parent.parent.parent / "models" / "weights" / "mantranet" / "ManTraNet_Ptrain4.h5"
            # Architecture in ai_detection/mantranet/src/
            arch_path = Path(__file__).parent.parent / "mantranet" / "src" / "modelCore.py"
            
            if not model_path.exists():
                logger.warning(
                    "ManTraNet model not found at %s — run: make mantranet-setup",
                    model_path
                )
                return False
            
            if not arch_path.exists():
                logger.warning(
                    "ManTraNet architecture not found at %s — run: make mantranet-setup",
                    arch_path
                )
                return False
            
            # Check TensorFlow and Keras
            import tensorflow as tf
            import keras
            
            self._model_path = model_path
            return True
            
        except Exception as e:
            logger.error("Error checking ManTraNet dependencies: %s", e)
            return False
    
    def detect(self, image_path: str, context=None) -> DetectionResult:
        """
        Detect and localize image manipulations using ManTraNet.
        
        Runs inference via subprocess to isolate TensorFlow 1.14 environment.
        Analyzes the output mask to generate audit findings and a detection result.
        
        Args:
            image_path: Path to image file
            context: Optional ResultStore for accessing metadata and saving mask
            
        Returns:
            DetectionResult with manipulation evidence and confidence
        """
        if not self.check_deps():
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence="ManTraNet not available — run: make mantranet-setup"
            )
        
        try:
            # Run inference directly (TensorFlow 2.x with compat mode)
            infer_script = Path(__file__).parent.parent / "mantranet_infer.py"
            model_path = self._model_path
            
            # Use sys.executable to run in current environment
            # Suppress stderr to hide TensorFlow warnings
            import sys
            result = subprocess.run(
                [sys.executable, str(infer_script), str(model_path), str(image_path)],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes max
                env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3', 'CUDA_VISIBLE_DEVICES': ''}
            )
            
            if result.returncode != 0:
                logger.error("ManTraNet inference failed: %s", result.stderr)
                return DetectionResult(
                    method=DetectionMethod.ML_MODEL,
                    is_ai_generated=None,
                    confidence=ConfidenceLevel.NONE,
                    score=0.0,
                    evidence=f"ManTraNet inference error: {result.stderr[:200]}"
                )
            
            # Parse JSON output
            output = json.loads(result.stdout)
            
            if not output.get("success"):
                return DetectionResult(
                    method=DetectionMethod.ML_MODEL,
                    is_ai_generated=None,
                    confidence=ConfidenceLevel.NONE,
                    score=0.0,
                    evidence=f"ManTraNet error: {output.get('error', 'Unknown error')}"
                )
            
            # Extract mask analysis
            analysis = output.get("analysis", {})
            manipulated_pct = analysis.get("manipulated_percentage", 0.0)
            region_count = analysis.get("region_count", 0)
            max_confidence = analysis.get("max_confidence", 0.0)
            
            # Save mask to GridFS if context provided
            mask_id = None
            if "mask_bytes" in output:
                if context:
                    try:
                        from lib.db import save_file
                        import base64
                        
                        mask_data = base64.b64decode(output["mask_bytes"])
                        
                        # Use save_file() like other plugins (ELA, etc) - returns UUID not ObjectId
                        mask_id = save_file(mask_data, content_type="image/png")
                        logger.info("Saved ManTraNet mask to GridFS: %s", mask_id)
                    except Exception as e:
                        logger.error("Failed to save mask to GridFS: %s", e)
                else:
                    logger.debug("No context provided, mask not saved to GridFS")
            else:
                logger.warning("No mask_bytes in ManTraNet output")
            
            # Generate audit findings
            audit_findings = self._create_audit_findings(analysis)
            
            # Determine overall detection result
            is_manipulated = manipulated_pct > 1.0  # >1% threshold
            confidence = self._calculate_confidence(manipulated_pct)
            
            # Calculate score (0=pristine, 1=heavily manipulated)
            score = min(manipulated_pct / 20.0, 1.0)  # 20% = full manipulation
            
            evidence = (
                f"ManTraNet: {manipulated_pct:.1f}% of image shows manipulation "
                f"({region_count} region{'s' if region_count != 1 else ''}, "
                f"max confidence: {max_confidence:.2f})"
            )
            
            metadata = {
                "manipulated_percentage": manipulated_pct,
                "region_count": region_count,
                "max_confidence": max_confidence,
                "inference_time_s": output.get("timing", {}).get("total_time", 0),
                "mask_id": mask_id,  # UUID from save_file()
                "audit_findings": audit_findings  # Store for auditor extraction
            }
            
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=is_manipulated,
                confidence=confidence,
                score=score,
                evidence=evidence,
                metadata=metadata
            )
            
        except subprocess.TimeoutExpired:
            logger.error("ManTraNet inference timed out")
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence="ManTraNet inference timed out (>2 minutes)"
            )
        except Exception as e:
            logger.error("ManTraNet detection error: %s", e)
            return DetectionResult(
                method=DetectionMethod.ML_MODEL,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence=f"ManTraNet error: {str(e)}"
            )
    
    def _calculate_confidence(self, manipulated_pct: float) -> ConfidenceLevel:
        """Map manipulation percentage to confidence level."""
        if manipulated_pct > 20.0:
            return ConfidenceLevel.HIGH
        elif manipulated_pct > 5.0:
            return ConfidenceLevel.MEDIUM
        elif manipulated_pct > 1.0:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NONE
    
    def _create_audit_findings(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate audit findings from mask analysis.
        
        Findings are scored by the auditor based on manipulation percentage:
        - HIGH: >20% manipulated (serious forgery)
        - MEDIUM: 5-20% manipulated (moderate editing)
        - LOW: 1-5% manipulated (minor touch-ups)
        - POSITIVE: <0.5% manipulated (likely pristine)
        
        Args:
            analysis: Dict with manipulated_percentage, region_count, max_confidence
            
        Returns:
            List of audit findings for auditor consumption
        """
        findings = []
        manipulated_pct = analysis.get("manipulated_percentage", 0.0)
        region_count = analysis.get("region_count", 0)
        max_conf = analysis.get("max_confidence", 0.0)
        
        if manipulated_pct > 20.0 and max_conf > 0.7:
            # HIGH: Serious forgery detected
            findings.append({
                "level": "HIGH",
                "category": "forgery_localization",
                "description": (
                    f"Extensive manipulation detected: {manipulated_pct:.1f}% of image "
                    f"shows forgery signatures across {region_count} region(s)"
                ),
                "is_positive": False,
                "confidence": float(max_conf)
            })
        elif manipulated_pct > 5.0:
            # MEDIUM: Moderate manipulation
            findings.append({
                "level": "MEDIUM",
                "category": "forgery_localization",
                "description": (
                    f"Moderate manipulation detected: {manipulated_pct:.1f}% of image "
                    f"shows editing in {region_count} region(s)"
                ),
                "is_positive": False,
                "confidence": float(max_conf)
            })
        elif manipulated_pct > 1.0:
            # LOW: Minor manipulation
            findings.append({
                "level": "LOW",
                "category": "forgery_localization",
                "description": (
                    f"Minor manipulation detected: {manipulated_pct:.1f}% of image "
                    f"shows possible editing"
                ),
                "is_positive": False,
                "confidence": float(max_conf)
            })
        elif manipulated_pct < 0.5:
            # POSITIVE: Likely pristine
            findings.append({
                "level": "MEDIUM",  # Use MEDIUM for positive findings (+15 points)
                "category": "forgery_localization",
                "description": (
                    "No significant manipulation detected: image appears pristine "
                    f"({manipulated_pct:.1f}% noise threshold)"
                ),
                "is_positive": True,
                "confidence": 1.0 - float(max_conf) if max_conf < 0.5 else 0.9
            })
        
        return findings
