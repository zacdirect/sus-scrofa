# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
import requests
from pathlib import Path

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import AutoVivification, str2temp_file

logger = logging.getLogger(__name__)


class OpenCVManipulation(BaseAnalyzerModule):
    """
    Image manipulation detection using OpenCV service.
    
    Analyzes images for signs of manipulation using multiple OpenCV techniques:
    - Gaussian blur difference detection
    - Laplacian noise analysis
    - JPEG artifact analysis
    
    Runs as a containerized service for isolation and easy updates.
    """
    
    order = 40
    depends_on = []
    enabled = True
    description = "Image Manipulation Detection (OpenCV)"
    name = "opencv_manipulation"
    
    # Service configuration
    SERVICE_URL = "http://localhost:8080"
    TIMEOUT = 30  # seconds
    
    def __init__(self):
        self._service_available = False
    
    def check_deps(self):
        """Check if OpenCV service is running and accessible."""
        try:
            response = requests.get(
                f"{self.SERVICE_URL}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    logger.info("OpenCV service is healthy and available")
                    self._service_available = True
                    return True
            
            logger.warning(f"OpenCV service unhealthy: {response.status_code}")
            self._service_available = False
            return False
            
        except requests.exceptions.ConnectionError:
            logger.warning("OpenCV service not reachable - run: make opencv-start")
            self._service_available = False
            return False
        except Exception as e:
            logger.error(f"Error checking OpenCV service: {e}")
            self._service_available = False
            return False
    
    def run(self, task):
        """Run OpenCV manipulation detection on the image."""
        results = AutoVivification()
        
        if not self.check_deps():
            results["opencv_manipulation"]["error"] = "OpenCV service not available - run: make opencv-start"
            results["opencv_manipulation"]["enabled"] = False
            return results
        
        # Create temporary file from image data
        tmp_file = None
        try:
            # Get image data and create temp file with original filename
            tmp_file = str2temp_file(task.get_file_data, suffix=f"-{task.file_name}")
            tmp_file.flush()
            
            # Send to OpenCV service for analysis
            logger.info(f"[Task {task.id}]: Running OpenCV manipulation analysis on {task.file_name}")
            
            with open(tmp_file.name, 'rb') as f:
                files = {'image': (task.file_name, f, 'image/*')}
                response = requests.post(
                    f"{self.SERVICE_URL}/analyze",
                    files=files,
                    timeout=self.TIMEOUT
                )
            
            if response.status_code != 200:
                logger.error(f"[Task {task.id}]: OpenCV service returned {response.status_code}")
                results["opencv_manipulation"]["error"] = f"Service error: {response.status_code}"
                results["opencv_manipulation"]["enabled"] = False
                return results
            
            # Parse response
            analysis = response.json()
            
            if not analysis.get('success'):
                error_msg = analysis.get('error', 'Unknown error')
                logger.error(f"[Task {task.id}]: Analysis failed: {error_msg}")
                results["opencv_manipulation"]["error"] = error_msg
                results["opencv_manipulation"]["enabled"] = False
                return results
            
            # Extract results
            opencv_results = analysis.get('results', {})
            
            # Format for SusScrofa
            results["opencv_manipulation"]["enabled"] = True
            results["opencv_manipulation"]["is_suspicious"] = opencv_results.get('is_suspicious', False)
            results["opencv_manipulation"]["overall_confidence"] = opencv_results.get('overall_confidence', 0.0)
            
            # Manipulation detection
            if 'manipulation_detection' in opencv_results:
                manip = opencv_results['manipulation_detection']
                results["opencv_manipulation"]["manipulation_detection"] = {
                    'method': manip.get('method'),
                    'is_manipulated': manip.get('is_manipulated', False),
                    'confidence': manip.get('confidence', 0.0),
                    'num_anomalies': manip.get('num_anomalies', 0),
                    'anomaly_percentage': manip.get('anomaly_percentage', 0.0),
                    'evidence': manip.get('evidence', '')
                }
            
            # Noise analysis
            if 'noise_analysis' in opencv_results:
                noise = opencv_results['noise_analysis']
                results["opencv_manipulation"]["noise_analysis"] = {
                    'method': noise.get('method'),
                    'is_noise_inconsistent': noise.get('is_noise_inconsistent', False),
                    'noise_consistency': noise.get('noise_consistency', 0.0),
                    'overall_noise': noise.get('overall_noise', 0.0),
                    'coefficient_variation': noise.get('coefficient_variation', 0.0),
                    'quadrant_variances': noise.get('quadrant_variances', [])
                }
            
            # JPEG artifact analysis
            if 'jpeg_artifacts' in opencv_results:
                jpeg = opencv_results['jpeg_artifacts']
                results["opencv_manipulation"]["jpeg_artifacts"] = {
                    'method': jpeg.get('method'),
                    'has_inconsistent_artifacts': jpeg.get('has_inconsistent_artifacts', False),
                    'confidence': jpeg.get('confidence', 0.0),
                    'compression_variation': jpeg.get('compression_variation', 0.0),
                    'evidence': jpeg.get('evidence', '')
                }
            
            # Add interpretation
            confidence = opencv_results.get('overall_confidence', 0.0)
            is_suspicious = opencv_results.get('is_suspicious', False)
            
            if is_suspicious:
                if confidence >= 0.7:
                    interpretation = "High confidence: Image shows signs of manipulation"
                elif confidence >= 0.5:
                    interpretation = "Moderate confidence: Possible manipulation detected"
                else:
                    interpretation = "Low confidence: Some suspicious patterns found"
            else:
                interpretation = "No significant manipulation detected"
            
            results["opencv_manipulation"]["interpretation"] = interpretation
            
            # Add detection info
            results["opencv_manipulation"]["service"] = "OpenCV Container"
            results["opencv_manipulation"]["methods"] = [
                "Gaussian Blur Difference",
                "Laplacian Noise Analysis",
                "JPEG Artifact Analysis"
            ]
            
            logger.info(f"[Task {task.id}]: OpenCV analysis complete - {interpretation}")
            
        except requests.exceptions.Timeout:
            logger.error(f"[Task {task.id}]: OpenCV service timed out")
            results["opencv_manipulation"]["error"] = "Analysis timed out (>30s)"
            results["opencv_manipulation"]["enabled"] = False
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"[Task {task.id}]: Request error: {e}")
            results["opencv_manipulation"]["error"] = str(e)
            results["opencv_manipulation"]["enabled"] = False
            return results
        except Exception as e:
            logger.error(f"[Task {task.id}]: OpenCV analysis failed: {e}")
            results["opencv_manipulation"]["error"] = str(e)
            results["opencv_manipulation"]["enabled"] = False
            return results
        finally:
            # Clean up temporary file
            if tmp_file:
                try:
                    tmp_file.close()
                except:
                    pass
        
        return results
