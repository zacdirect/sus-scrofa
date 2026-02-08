# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of SusScrofa.
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
    
    2. **ML Model Detection** (Slower, for enrichment when metadata stripped)
       - SPAI (Spectral AI-Generated Image Detector) - CVPR 2025
       - Frequency domain analysis
       - Works on any image resolution
    
    Detection Strategy:
    - Run fast metadata checks first
    - If high-confidence result found, stop (early exit)
    - Otherwise, run ML model for deeper analysis
    - Combine results with confidence-weighted decision logic
    
    This multi-layer approach is robust against:
    - Metadata stripping
    - Post-processing
    - New AI generators
    - Edge cases
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
        """Run multi-layer AI detection on the image."""
        results = AutoVivification()
        
        if not self.check_deps():
            results["ai_detection"]["error"] = "AI detection not available - run: make ai-setup"
            results["ai_detection"]["enabled"] = False
            return results
        
        # Create temporary file from image data
        tmp_file = None
        try:
            # Get image data and create temp file with original filename as suffix
            # This preserves filename for pattern matching (e.g., "gemini_generated")
            tmp_file = str2temp_file(task.get_file_data, suffix=f"-{task.file_name}")
            tmp_file.flush()  # Ensure data is written
            image_path = tmp_file.name
            
            # Run multi-layer detection
            logger.info(f"[Task {task.id}]: Running multi-layer AI detection on {task.file_name}")
            detection_result = self._detector.detect(image_path, early_stop=True)
            
            # Extract results (handle both individual detector and audit aggregated results)
            verdict = detection_result.get('overall_verdict')
            confidence = detection_result.get('overall_confidence', 'NONE')
            score = detection_result.get('overall_score', 0.0)
            authenticity_score = detection_result.get('authenticity_score')  # 0-100 from audit
            evidence = detection_result.get('evidence', 'No evidence')
            detected_types = detection_result.get('detected_types', [])
            layer_results = detection_result.get('layer_results', [])
            
            # Format for SusScrofa
            results["ai_detection"]["enabled"] = detection_result.get('enabled', True)
            results["ai_detection"]["verdict"] = "AI-Generated" if verdict else "Real" if verdict is False else "Unknown"
            results["ai_detection"]["confidence"] = confidence.lower() if confidence != 'AUDIT' else 'high'
            results["ai_detection"]["ai_probability"] = round(score * 100.0, 2)
            results["ai_detection"]["likely_ai"] = verdict is True
            
            # Add authenticity score if available (from compliance audit)
            if authenticity_score is not None:
                results["ai_detection"]["authenticity_score"] = authenticity_score
                results["ai_detection"]["detected_types"] = detected_types
            
            # Add evidence
            results["ai_detection"]["evidence"] = evidence
            
            # Add layer-by-layer results
            results["ai_detection"]["detection_layers"] = []
            for layer in layer_results:
                layer_info = {
                    'method': layer.get('method'),
                    'evidence': layer.get('evidence', '')
                }
                
                # Handle individual detector format
                if 'is_ai_generated' in layer:
                    layer_info['verdict'] = "AI" if layer.get('is_ai_generated') else "Real" if layer.get('is_ai_generated') is False else "Unknown"
                    layer_info['confidence'] = layer.get('confidence')
                    layer_info['score'] = layer.get('score', 0.0)
                
                # Handle compliance audit format
                if 'authenticity_score' in layer:
                    layer_info['authenticity_score'] = layer.get('authenticity_score')
                    layer_info['is_fake'] = layer.get('is_fake')
                    layer_info['detected_types'] = layer.get('detected_types', [])
                
                results["ai_detection"]["detection_layers"].append(layer_info)
            
            # Add interpretation
            if confidence == 'AUDIT':
                # Audit-based interpretation using authenticity_score
                if authenticity_score <= 20:
                    interpretation = "Definitive: AI-generated or heavily manipulated"
                elif authenticity_score <= 40:
                    interpretation = "High confidence: Likely AI-generated or manipulated"
                elif authenticity_score >= 80:
                    interpretation = "High confidence: Likely authentic photograph"
                elif authenticity_score >= 60:
                    interpretation = "Moderate confidence: Possibly authentic"
                else:
                    interpretation = "Low confidence: Uncertain origin"
            elif confidence == 'CERTAIN':
                interpretation = "Definitive: " + ("AI-generated" if verdict else "Authentic photograph")
            elif confidence == 'HIGH':
                interpretation = "High confidence: " + ("Likely AI-generated" if verdict else "Likely authentic")
            elif confidence == 'MEDIUM':
                interpretation = "Moderate confidence: " + ("Possibly AI-generated" if verdict else "Possibly authentic")
            elif confidence == 'LOW':
                interpretation = "Low confidence: Uncertain origin"
            else:
                interpretation = "Unable to determine with available methods"
            
            results["ai_detection"]["interpretation"] = interpretation
            
            # Add detection info
            results["ai_detection"]["detection_framework"] = "Multi-Layer (Metadata + ML Model)"
            results["ai_detection"]["available_methods"] = [d.name for d in self._detector.detectors]
            
            logger.info(f"[Task {task.id}]: AI detection complete - {interpretation}")
            
        except Exception as e:
            logger.error(f"[Task {task.id}]: AI detection failed: {e}")
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
        
        return results
