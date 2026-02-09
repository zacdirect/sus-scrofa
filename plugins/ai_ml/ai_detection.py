# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
import sys
from pathlib import Path

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import AutoVivification, str2temp_file

logger = logging.getLogger(__name__)


class AIDetection(BaseAnalyzerModule):
    """
    Multi-layer AI-generated image detection.
    
    Uses multiple complementary detection methods:
    
    1. **Metadata Analysis** (Fast, 100% accurate when present)
       - Checks EXIF/XMP for AI generator tags
       - Looks for software signatures (Midjourney, DALL-E, Stable Diffusion, etc.)
       - Checks for C2PA content credentials
    
    2. **SDXL Detector** (Moderate speed, high accuracy on modern AI images)
       - Swin Transformer (Organika/sdxl-detector)
       - Trained on Wikimedia-SDXL image pairs
       - 98.1% published accuracy
    
    3. **SPAI ML Model** (Slower, for enrichment when metadata stripped)
       - SPAI (Spectral AI-Generated Image Detector) - CVPR 2025
       - Frequency domain analysis
       - Works on any image resolution
    
    Detection Strategy:
    - Run fast metadata checks first
    - If high-confidence result found, stop (early exit)
    - Otherwise, run ML models for deeper analysis
    - Raw results stored for the engine-level compliance auditor
    
    This plugin does NOT score or produce verdicts.  It writes raw
    detector results under ``ai_detection``.  The compliance auditor
    (lib/analyzer/auditor.py) is a separate engine component that reads
    these results, along with every other plugin's output, and produces
    the authoritative verdict.
    """

    name = "AI Generation Detection (Multi-Layer)"
    description = "Metadata + ML model detection for AI-generated images"
    order = 30

    def __init__(self):
        super().__init__()
        self._detector = None
        self._detector_available = None

    def check_deps(self):
        """Check if detection framework is available."""
        if self._detector_available is not None:
            return self._detector_available
        
        try:
            # Check if ai_detection module exists
            ai_detection_dir = Path(__file__).parent.parent.parent / 'ai_detection'
            detectors_dir = ai_detection_dir / 'detectors'
            
            if not detectors_dir.exists():
                logger.warning("AI detection module not found")
                logger.info("The ai_detection module structure has been created but needs setup")
                self._detector_available = False
                return False
            
            # Import multi-layer detector
            sys.path.insert(0, str(ai_detection_dir))
            from detectors.orchestrator import MultiLayerDetector
            
            # Initialize with ML enabled (will auto-disable if not available)
            self._detector = MultiLayerDetector(enable_ml=True)
            
            # At least one detector should be available
            if len(self._detector.detectors) == 0:
                logger.warning("No AI detection methods available")
                logger.info("Run: make ai-setup")
                self._detector_available = False
                return False
            
            detector_names = [d.name for d in self._detector.detectors]
            logger.info(f"AI detection initialized with {len(self._detector.detectors)} methods: {', '.join(detector_names)}")
            self._detector_available = True
            return True
            
        except ImportError as e:
            logger.warning(f"AI detection module not available: {e}")
            logger.info("Run: make ai-setup to enable AI detection")
            self._detector_available = False
            return False
        except Exception as e:
            logger.error(f"Error initializing AI detection: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self._detector_available = False
            return False

    def run(self, task):
        """Run multi-layer AI detection on the image.

        Writes raw detector results under ``ai_detection``.  The engine-level
        auditor and confidence scorer (Phase 2 in processing.py) consume these
        results later — this plugin does NOT produce scores or verdicts.
        """
        results = AutoVivification()

        if not self.check_deps():
            results["ai_detection"]["error"] = "AI detection not available - run: make ai-setup"
            results["ai_detection"]["enabled"] = False
            return results

        tmp_file = None
        try:
            tmp_file = str2temp_file(task.get_file_data, suffix=f"-{task.file_name}")
            tmp_file.flush()
            image_path = tmp_file.name

            logger.info(f"[Task {task.id}]: Running multi-layer AI detection on {task.file_name}")
            detection_result = self._detector.detect(image_path, early_stop=True)

            # Store raw results — no scoring, no interpretation
            results["ai_detection"]["enabled"] = detection_result.get("enabled", True)
            results["ai_detection"]["detection_layers"] = detection_result.get("detection_layers", [])
            results["ai_detection"]["methods_run"] = detection_result.get("methods_run", [])

            if "error" in detection_result:
                results["ai_detection"]["error"] = detection_result["error"]

            methods = detection_result.get("methods_run", [])
            logger.info(f"[Task {task.id}]: AI detection complete — {len(methods)} method(s) ran: {', '.join(methods)}")

        except Exception as e:
            logger.error(f"[Task {task.id}]: AI detection failed: {e}")
            results["ai_detection"]["error"] = str(e)
            results["ai_detection"]["enabled"] = False
            return results
        finally:
            if tmp_file:
                try:
                    tmp_file.close()
                except Exception:
                    pass

        return results
