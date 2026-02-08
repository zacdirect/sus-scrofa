"""
OpenCV Image Analysis Service

REST API for image manipulation detection and analysis.
Provides endpoints for detecting image tampering, noise analysis, and other
OpenCV-based detection methods.
"""

import os
import io
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def decode_image(image_data):
    """
    Decode base64 image data to OpenCV format.
    
    Args:
        image_data: Base64 encoded image string or raw bytes
        
    Returns:
        OpenCV image (numpy array)
    """
    try:
        # Handle base64 encoded data
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        return img
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise


def detect_manipulation(image):
    """
    Detect image manipulation using OpenCV techniques.
    
    Based on the Medium article approach:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Calculate absolute difference
    4. Threshold to find anomalies
    5. Find contours (manipulated regions)
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Dictionary with detection results
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Calculate absolute difference
        difference = cv2.absdiff(gray, blurred)
        
        # Threshold to binary
        _, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresholded.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate metrics
        num_anomalies = len(contours)
        total_anomaly_area = sum(cv2.contourArea(c) for c in contours)
        image_area = gray.shape[0] * gray.shape[1]
        anomaly_percentage = (total_anomaly_area / image_area) * 100
        
        # Determine if manipulated
        # Thresholds based on empirical testing
        is_manipulated = num_anomalies > 5 and anomaly_percentage > 1.0
        
        # Calculate confidence
        if num_anomalies == 0:
            confidence = 0.0
        elif num_anomalies > 50:
            confidence = min(0.95, 0.5 + (anomaly_percentage / 20.0))
        else:
            confidence = min(0.8, anomaly_percentage / 10.0)
        
        return {
            'is_manipulated': bool(is_manipulated),
            'confidence': float(confidence),
            'num_anomalies': int(num_anomalies),
            'anomaly_percentage': float(anomaly_percentage),
            'method': 'gaussian_blur_difference',
            'evidence': f"Found {num_anomalies} anomalous regions covering {anomaly_percentage:.2f}% of image"
        }
        
    except Exception as e:
        logger.error(f"Manipulation detection error: {e}")
        raise


def analyze_noise_patterns(image):
    """
    Analyze noise patterns in the image.
    
    Natural photos have consistent noise patterns.
    Manipulated/AI images often have inconsistent noise.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Dictionary with noise analysis results
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (noise estimate)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Split into quadrants and check consistency
        h, w = gray.shape
        quadrants = [
            gray[0:h//2, 0:w//2],      # Top-left
            gray[0:h//2, w//2:w],      # Top-right
            gray[h//2:h, 0:w//2],      # Bottom-left
            gray[h//2:h, w//2:w]       # Bottom-right
        ]
        
        # Calculate noise variance for each quadrant
        quadrant_vars = [cv2.Laplacian(q, cv2.CV_64F).var() for q in quadrants]
        
        # Calculate coefficient of variation (CV)
        mean_var = np.mean(quadrant_vars)
        std_var = np.std(quadrant_vars)
        cv = (std_var / mean_var) if mean_var > 0 else 0
        
        # Inconsistent noise suggests manipulation
        is_inconsistent = cv > 0.5  # Threshold based on empirical testing
        
        return {
            'overall_noise': float(laplacian_var),
            'noise_consistency': float(1.0 - cv),  # Higher = more consistent
            'is_noise_inconsistent': bool(is_inconsistent),
            'quadrant_variances': [float(v) for v in quadrant_vars],
            'coefficient_variation': float(cv),
            'method': 'laplacian_noise_analysis'
        }
        
    except Exception as e:
        logger.error(f"Noise analysis error: {e}")
        raise


def detect_jpeg_artifacts(image):
    """
    Detect JPEG compression artifacts and inconsistencies.
    
    Manipulated images often have inconsistent compression artifacts
    in different regions.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Dictionary with JPEG artifact analysis
    """
    try:
        # Convert to YCrCb color space (similar to JPEG)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        # Apply DCT to 8x8 blocks (JPEG compression uses 8x8 blocks)
        h, w = y_channel.shape
        block_diffs = []
        
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = y_channel[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # High-frequency components (bottom-right of DCT)
                high_freq = np.sum(np.abs(dct_block[4:, 4:]))
                block_diffs.append(high_freq)
        
        if len(block_diffs) == 0:
            return {
                'has_inconsistent_artifacts': False,
                'confidence': 0.0,
                'method': 'jpeg_artifact_analysis',
                'evidence': 'Image too small for analysis'
            }
        
        # Calculate statistics
        mean_diff = np.mean(block_diffs)
        std_diff = np.std(block_diffs)
        cv_blocks = (std_diff / mean_diff) if mean_diff > 0 else 0
        
        # High variation suggests inconsistent compression (manipulation)
        has_inconsistent = cv_blocks > 1.0
        confidence = min(0.7, cv_blocks / 2.0)
        
        return {
            'has_inconsistent_artifacts': bool(has_inconsistent),
            'confidence': float(confidence),
            'compression_variation': float(cv_blocks),
            'method': 'jpeg_artifact_analysis',
            'evidence': f"Compression variation coefficient: {cv_blocks:.2f}"
        }
        
    except Exception as e:
        logger.error(f"JPEG artifact detection error: {e}")
        raise


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'opencv-analysis'})


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    
    Accepts multipart/form-data with 'image' file or
    JSON with base64 encoded image.
    
    Returns JSON with analysis results.
    """
    try:
        # Get image from request
        if request.files and 'image' in request.files:
            # From multipart form
            file = request.files['image']
            image_bytes = file.read()
            img = decode_image(image_bytes)
        elif request.is_json:
            # From JSON body
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            img = decode_image(data['image'])
        else:
            return jsonify({'error': 'Invalid request format'}), 400
        
        # Run all detection methods
        logger.info("Running manipulation detection...")
        manipulation_result = detect_manipulation(img)
        
        logger.info("Running noise analysis...")
        noise_result = analyze_noise_patterns(img)
        
        logger.info("Running JPEG artifact detection...")
        artifact_result = detect_jpeg_artifacts(img)
        
        # Combine results
        overall_suspicious = (
            manipulation_result['is_manipulated'] or
            noise_result['is_noise_inconsistent'] or
            artifact_result['has_inconsistent_artifacts']
        )
        
        # Calculate overall confidence (weighted average)
        confidences = [
            manipulation_result['confidence'] * 0.4,
            (1.0 - noise_result['noise_consistency']) * 0.3,
            artifact_result['confidence'] * 0.3
        ]
        overall_confidence = sum(confidences)
        
        response = {
            'success': True,
            'results': {
                'is_suspicious': overall_suspicious,
                'overall_confidence': overall_confidence,
                'manipulation_detection': manipulation_result,
                'noise_analysis': noise_result,
                'jpeg_artifacts': artifact_result
            }
        }
        
        logger.info(f"Analysis complete - Suspicious: {overall_suspicious}, Confidence: {overall_confidence:.2f}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect-manipulation', methods=['POST'])
def detect_manipulation_endpoint():
    """Endpoint specifically for manipulation detection."""
    try:
        if request.files and 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
            img = decode_image(image_bytes)
        elif request.is_json:
            data = request.get_json()
            img = decode_image(data['image'])
        else:
            return jsonify({'error': 'Invalid request format'}), 400
        
        result = detect_manipulation(img)
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"Manipulation detection error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting OpenCV Analysis Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
