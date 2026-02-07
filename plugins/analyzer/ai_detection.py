# Ghiro - Copyright (C) 2013-2026 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
import sys
import subprocess
import json
import os
from pathlib import Path

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import AutoVivification, str2temp_file

logger = logging.getLogger(__name__)


class SPAIDetection(BaseAnalyzerModule):
    """
    AI-generated image detection using SPAI (Spectral AI-Generated Image Detector).
    
    Based on: "Any-Resolution AI-Generated Image Detection by Spectral Learning"
    Paper: CVPR 2025
    Authors: Karageorgiou et al., CERTH/University of Amsterdam
    
    SPAI uses spectral learning to learn the spectral distribution of real images
    under a self-supervised setup, then detects AI-generated images as 
    out-of-distribution samples.
    
    Key advantages:
    - Works on any image resolution (no resizing needed)
    - Uses frequency domain analysis for subtle artifact detection
    - Trained on latest generators (SD3, Midjourney v6.1, etc.)
    - State-of-the-art accuracy (CVPR 2025)
    
    License: Apache 2.0 (compatible with Ghiro)
    GitHub: https://github.com/mever-team/spai
    
    Note: Model is loaded once per processor instance, then reused for all images.
    """

    name = "AI Generation Detection"
    description = "Spectral learning detector for any-resolution images (SPAI/CVPR 2025)"
    order = 30

    def __init__(self):
        super().__init__()
        self._spai_available = None

    def check_deps(self):
        """Check if SPAI is installed and model weights are available."""
        if self._spai_available is not None:
            return self._spai_available
        
        # Check if ai_detection venv exists
        ai_detection_dir = Path(__file__).parent.parent.parent / 'ai_detection'
        venv_python = ai_detection_dir / '.venv' / 'bin' / 'python'
        
        if not venv_python.exists():
            logger.warning("AI detection virtual environment not found")
            logger.info("Run: make ai-setup")
            self._spai_available = False
            return False
        
        # Check if SPAI module is available in the venv
        try:
            result = subprocess.run(
                [str(venv_python), '-c', 'from spai.inference import SPAIDetector; print("OK")'],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(ai_detection_dir)
            )
            
            if result.returncode != 0 or 'OK' not in result.stdout:
                logger.warning(f"SPAI module not available: {result.stderr}")
                logger.info("Run: make ai-setup")
                self._spai_available = False
                return False
            
            # Check for model weights
            weights_path = self._get_weights_path()
            if not weights_path.exists():
                logger.warning(f"SPAI model weights not found at {weights_path}")
                logger.info("Run: make ai-setup")
                logger.info("Or download from: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view")
                self._spai_available = False
                return False
            
            self._spai_available = True
            return True
            
        except subprocess.TimeoutExpired:
            logger.warning("SPAI module check timed out")
            self._spai_available = False
            return False
        except Exception as e:
            logger.error(f"Error checking SPAI availability: {e}")
            self._spai_available = False
            return False
    
    def _get_weights_path(self):
        """Get path to SPAI model weights."""
        # Use ai_detection module's weights directory
        weights_path = Path(__file__).parent.parent.parent / 'ai_detection' / 'weights' / 'spai.pth'
        return weights_path

    def run(self, task):
        """Run SPAI AI detection on the image."""
        results = AutoVivification()
        
        if not self.check_deps():
            results["ai_detection"]["error"] = "SPAI not available - run: make ai-setup"
            results["ai_detection"]["enabled"] = False
            return results
        
        # Create temporary file from image data
        tmp_file = None
        try:
            # Get image data and create temp file
            tmp_file = str2temp_file(task.get_file_data)
            tmp_file.flush()  # Ensure data is written
            image_path = Path(tmp_file.name)
            
            weights_path = self._get_weights_path()
            ai_detection_dir = Path(__file__).parent.parent.parent / 'ai_detection'
            venv_python = ai_detection_dir / '.venv' / 'bin' / 'python'
            infer_script = ai_detection_dir / 'spai_infer.py'
            
            # Run inference via subprocess
            logger.info(f"[Task {task.id}]: Running SPAI inference on {task.file_name}")
            result = subprocess.run(
                [str(venv_python), str(infer_script), str(weights_path), str(image_path)],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=str(ai_detection_dir)
            )
            
            if result.returncode != 0:
                logger.error(f"[Task {task.id}]: SPAI inference failed: {result.stderr}")
                results["ai_detection"]["error"] = f"Inference failed: {result.stderr[:200]}"
                results["ai_detection"]["enabled"] = False
                return results
            
            # Parse JSON result
            inference_result = json.loads(result.stdout)
            
            if not inference_result.get('success', False):
                error_msg = inference_result.get('error', 'Unknown error')
                logger.error(f"[Task {task.id}]: SPAI inference error: {error_msg}")
                results["ai_detection"]["error"] = error_msg
                results["ai_detection"]["enabled"] = False
                return results
            
            # Extract results
            probability = inference_result['score'] * 100.0  # Convert to percentage
            logit = inference_result['logit']
            
        except subprocess.TimeoutExpired:
            logger.error(f"[Task {task.id}]: SPAI inference timed out")
            results["ai_detection"]["error"] = "Inference timed out (>30s)"
            results["ai_detection"]["enabled"] = False
            return results
        except json.JSONDecodeError as e:
            logger.error(f"[Task {task.id}]: Failed to parse SPAI output: {e}")
            results["ai_detection"]["error"] = "Invalid inference output"
            results["ai_detection"]["enabled"] = False
            return results
        except Exception as e:
            logger.error(f"[Task {task.id}]: SPAI inference failed: {e}")
            results["ai_detection"]["error"] = str(e)
            results["ai_detection"]["enabled"] = False
            return results
        finally:
            # Clean up temporary file
            if tmp_file:
                try:
                    tmp_file.close()
                except:
                    pass
        
        # Format results
        results["ai_detection"]["enabled"] = True
        results["ai_detection"]["ai_probability"] = round(probability, 2)
        results["ai_detection"]["logit"] = round(logit, 4)
        results["ai_detection"]["likely_ai"] = probability > 50
        
        # Determine classification and confidence
        if probability >= 90:
            classification = "AI-generated"
            confidence = "very_high"
        elif probability >= 70:
            classification = "Likely AI-generated"
            confidence = "high"
        elif probability >= 30:
            classification = "Uncertain"
            confidence = "medium"
        elif probability >= 10:
            classification = "Likely authentic"
            confidence = "high"
        else:
            classification = "Authentic"
            confidence = "very_high"
        
        results["ai_detection"]["confidence"] = confidence
        results["ai_detection"]["interpretation"] = classification
        
        # Add detection method info
        results["ai_detection"]["model"] = "SPAI (Spectral Learning)"
        results["ai_detection"]["training"] = "CVPR 2025: SD3, Midjourney v6.1, DALL-E, etc."
        results["ai_detection"]["paper"] = "Karageorgiou et al., CVPR 2025"
        
        # Add device info
        import torch
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        results["ai_detection"]["device"] = device
        
        # Evidence based on probability
        evidence = []
        if probability > 50:
            if probability >= 90:
                evidence.append("Very strong spectral anomalies detected")
            elif probability >= 70:
                evidence.append("Significant spectral inconsistencies")
            else:
                evidence.append("Moderate spectral anomalies")
            
            evidence.append("Frequency domain patterns inconsistent with natural images")
        else:
            if probability <= 10:
                evidence.append("Spectral distribution consistent with authentic images")
            else:
                evidence.append("No significant spectral anomalies")
        
        results["ai_detection"]["evidence"] = evidence
        
        logger.info(f"[Task {task.id}]: SPAI detection complete - Probability: {probability:.1f}%")
        
        return results

