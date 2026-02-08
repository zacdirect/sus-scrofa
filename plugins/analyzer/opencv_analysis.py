# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
import requests
import base64
from pathlib import Path

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import AutoVivification, str2temp_file

logger = logging.getLogger(__name__)


class OpenCVAnalysis(BaseAnalyzerModule):
    """
    OpenCV-based image manipulation detection.
    
    Communicates with the containerized OpenCV service to perform:
    - Gaussian blur difference analysis (manipulation detection)
    - Noise pattern analysis
    - JPEG compression artifact detection
    
    The service runs in a separate container to isolate OpenCV dependencies
    from the main SusScrofa environment.
    """
    
    order = 40  # Run after basic metadata/EXIF but before AI detection
    
    def check_deps(self):
        """Check if OpenCV service is available."""
        try:
            # Check if service is running
            response = requests.get(
                'http://localhost:8080/health',
                timeout=2
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    logger.info("OpenCV service is available")
                    return True
            
            logger.warning("OpenCV service health check failed")
            return False
            
        except requests.exceptions.ConnectionError:
            logger.warning("OpenCV service not available - run: make opencv-start")
            return False
        except Exception as e:
            logger.error(f"Error checking OpenCV service: {e}")
            return False
    
    def run(self, task):
        """Run OpenCV analysis on the image."""
        results = AutoVivification()
        
        if not self.check_deps():
            results["opencv_analysis"]["error"] = "OpenCV service not available"
            results["opencv_analysis"]["enabled"] = False
            results["opencv_analysis"]["info"] = "Run 'make opencv-start' to enable OpenCV analysis"
            return results
        
        # Create temporary file from image data
        tmp_file = None
        try:
            # Get image data
            image_data = task.get_file_data
            
            # Send to OpenCV service
            logger.info(f"[Task {task.id}]: Sending {task.file_name} to OpenCV service")
            
            response = requests.post(
                'http://localhost:8080/analyze',
                files={'image': ('image.jpg', image_data, 'image/jpeg')},
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = f"OpenCV service returned status {response.status_code}"
                logger.error(f"[Task {task.id}]: {error_msg}")
                results["opencv_analysis"]["error"] = error_msg
                results["opencv_analysis"]["enabled"] = False
                return results
            
            # Parse response
            analysis_data = response.json()
            
            if not analysis_data.get('success'):
                error_msg = analysis_data.get('error', 'Unknown error')
                logger.error(f"[Task {task.id}]: OpenCV analysis failed: {error_msg}")
                results["opencv_analysis"]["error"] = error_msg
                results["opencv_analysis"]["enabled"] = False
                return results
            
            # Extract results
            opencv_results = analysis_data['results']
            
            # Overall assessment
            results["opencv_analysis"]["enabled"] = True
            results["opencv_analysis"]["is_suspicious"] = opencv_results['is_suspicious']
            results["opencv_analysis"]["overall_confidence"] = round(opencv_results['overall_confidence'] * 100, 2)
            
            # Manipulation detection
            manip = opencv_results['manipulation_detection']
            results["opencv_analysis"]["manipulation"] = {
                'detected': manip['is_manipulated'],
                'confidence': round(manip['confidence'] * 100, 2),
                'num_anomalies': manip['num_anomalies'],
                'anomaly_percentage': round(manip['anomaly_percentage'], 2),
                'evidence': manip['evidence']
            }
            
            # Noise analysis
            noise = opencv_results['noise_analysis']
            results["opencv_analysis"]["noise"] = {
                'inconsistent': noise['is_noise_inconsistent'],
                'consistency_score': round(noise['noise_consistency'] * 100, 2),
                'overall_noise_level': round(noise['overall_noise'], 2),
                'coefficient_variation': round(noise['coefficient_variation'], 3)
            }
            
            # JPEG artifacts
            artifacts = opencv_results['jpeg_artifacts']
            results["opencv_analysis"]["jpeg_artifacts"] = {
                'inconsistent': artifacts['has_inconsistent_artifacts'],
                'confidence': round(artifacts['confidence'] * 100, 2),
                'compression_variation': round(artifacts['compression_variation'], 3),
                'evidence': artifacts['evidence']
            }
            
            # Create human-readable summary
            if opencv_results['is_suspicious']:
                issues = []
                if manip['is_manipulated']:
                    issues.append(f"manipulation detected ({manip['num_anomalies']} anomalies)")
                if noise['is_noise_inconsistent']:
                    issues.append("inconsistent noise patterns")
                if artifacts['has_inconsistent_artifacts']:
                    issues.append("irregular JPEG compression")
                
                summary = f"Image shows signs of: {', '.join(issues)}"
                results["opencv_analysis"]["verdict"] = "Suspicious"
            else:
                summary = "No significant anomalies detected"
                results["opencv_analysis"]["verdict"] = "Authentic"
            
            results["opencv_analysis"]["summary"] = summary
            results["opencv_analysis"]["method"] = "OpenCV multi-technique analysis"
            
            logger.info(f"[Task {task.id}]: OpenCV analysis complete - {summary}")
            
        except requests.exceptions.Timeout:
            logger.error(f"[Task {task.id}]: OpenCV service request timed out")
            results["opencv_analysis"]["error"] = "Request timed out (>30s)"
            results["opencv_analysis"]["enabled"] = False
        except requests.exceptions.RequestException as e:
            logger.error(f"[Task {task.id}]: OpenCV service request failed: {e}")
            results["opencv_analysis"]["error"] = f"Service communication error: {str(e)}"
            results["opencv_analysis"]["enabled"] = False
        except Exception as e:
            logger.error(f"[Task {task.id}]: OpenCV analysis error: {e}", exc_info=True)
            results["opencv_analysis"]["error"] = str(e)
            results["opencv_analysis"]["enabled"] = False
        finally:
            # Clean up if needed
            if tmp_file:
                try:
                    tmp_file.close()
                except:
                    pass
        
        return results
