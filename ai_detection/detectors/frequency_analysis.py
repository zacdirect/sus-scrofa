"""
Frequency Analysis Detector for AI-generated images.

Detects checkerboard artifacts and periodic patterns in the frequency domain
that are typical of AI upsampling techniques used in GANs and Diffusion models.
"""

import logging
from typing import Optional

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult, DetectionMethod, ConfidenceLevel


logger = logging.getLogger(__name__)


class FrequencyAnalysisDetector(BaseDetector):
    """
    Detect AI-generated images through frequency domain analysis.
    
    AI models often use upsampling to increase resolution, which leaves
    periodic patterns (checkerboard effect) visible in the FFT spectrum.
    """
    
    name = "Frequency Analysis"
    
    def check_deps(self) -> bool:
        """Check if numpy is available."""
        try:
            import numpy
            import scipy.fft
            return True
        except ImportError:
            logger.warning("numpy or scipy not available for frequency analysis")
            return False
    
    def get_order(self) -> int:
        """Run after metadata but before heavy ML models."""
        return 50
    
    def detect(self, image_path: str, original_filename: str = None) -> DetectionResult:
        """
        Analyze image in frequency domain for AI artifacts.
        
        Args:
            image_path: Path to image file
            original_filename: Original filename (unused)
            
        Returns:
            DetectionResult with confidence based on artifact detection
        """
        try:
            # Load image and convert to grayscale for analysis
            img = Image.open(image_path).convert('L')
            
            # Resize to manageable size for performance (max 512px)
            max_size = 512
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Compute 2D FFT
            fft = np.fft.fft2(img_array)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Analyze for checkerboard pattern
            checkerboard_score = self._detect_checkerboard(magnitude)
            
            # Analyze for periodic artifacts
            periodic_score = self._detect_periodic_artifacts(magnitude)
            
            # Combine scores
            combined_score = max(checkerboard_score, periodic_score)
            
            # Determine verdict
            if combined_score > 0.7:
                return DetectionResult(
                    method=DetectionMethod.FREQUENCY_ANALYSIS,
                    is_ai_generated=True,
                    confidence=ConfidenceLevel.HIGH,
                    score=combined_score,
                    evidence=f"Strong frequency artifacts detected (checkerboard: {checkerboard_score:.2f}, periodic: {periodic_score:.2f})"
                )
            elif combined_score > 0.5:
                return DetectionResult(
                    method=DetectionMethod.FREQUENCY_ANALYSIS,
                    is_ai_generated=True,
                    confidence=ConfidenceLevel.MEDIUM,
                    score=combined_score,
                    evidence=f"Moderate frequency artifacts detected (checkerboard: {checkerboard_score:.2f}, periodic: {periodic_score:.2f})"
                )
            elif combined_score > 0.3:
                return DetectionResult(
                    method=DetectionMethod.FREQUENCY_ANALYSIS,
                    is_ai_generated=None,
                    confidence=ConfidenceLevel.LOW,
                    score=combined_score,
                    evidence=f"Weak frequency artifacts detected (checkerboard: {checkerboard_score:.2f}, periodic: {periodic_score:.2f})"
                )
            else:
                return DetectionResult(
                    method=DetectionMethod.FREQUENCY_ANALYSIS,
                    is_ai_generated=False,
                    confidence=ConfidenceLevel.MEDIUM,
                    score=combined_score,
                    evidence="No significant frequency artifacts detected"
                )
        
        except Exception as e:
            logger.error(f"Frequency analysis failed: {e}")
            return DetectionResult(
                method=DetectionMethod.FREQUENCY_ANALYSIS,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence=f"Analysis failed: {str(e)}"
            )
    
    def _detect_checkerboard(self, magnitude: np.ndarray) -> float:
        """
        Detect checkerboard artifacts in frequency domain.
        
        Checkerboard pattern appears as strong peaks at specific frequencies
        corresponding to the upsampling grid.
        """
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Look for peaks at specific frequencies (typical upsampling artifacts)
        # These appear at 1/2, 1/4, 3/4 positions from center
        suspicious_freqs = [
            (center_h + h // 4, center_w),  # Vertical
            (center_h - h // 4, center_w),
            (center_h, center_w + w // 4),  # Horizontal
            (center_h, center_w - w // 4),
        ]
        
        # Calculate average magnitude at suspicious frequencies
        suspicious_magnitude = 0.0
        for y, x in suspicious_freqs:
            if 0 <= y < h and 0 <= x < w:
                # Sample small region around point
                region = magnitude[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
                suspicious_magnitude += np.mean(region)
        
        suspicious_magnitude /= len(suspicious_freqs)
        
        # Calculate baseline magnitude (excluding center)
        mask = np.ones_like(magnitude, dtype=bool)
        mask[center_h-10:center_h+10, center_w-10:center_w+10] = False
        baseline_magnitude = np.mean(magnitude[mask])
        
        # Calculate ratio (higher = more suspicious)
        if baseline_magnitude > 0:
            ratio = suspicious_magnitude / baseline_magnitude
            # Normalize to 0-1 range
            score = min(1.0, max(0.0, (ratio - 1.0) / 5.0))  # ratio > 6.0 = score 1.0
            return score
        
        return 0.0
    
    def _detect_periodic_artifacts(self, magnitude: np.ndarray) -> float:
        """
        Detect periodic patterns that shouldn't exist in natural photos.
        
        AI upsampling can create regular grid patterns visible in FFT.
        """
        h, w = magnitude.shape
        
        # Calculate radial power spectrum
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Bin by radius
        max_radius = min(center_h, center_w)
        bins = np.linspace(0, max_radius, 50)
        
        power_spectrum = []
        for i in range(len(bins) - 1):
            mask = (r >= bins[i]) & (r < bins[i + 1])
            if np.any(mask):
                power_spectrum.append(np.mean(magnitude[mask]))
            else:
                power_spectrum.append(0)
        
        power_spectrum = np.array(power_spectrum)
        
        # Detect periodicities in power spectrum (shouldn't have strong peaks)
        if len(power_spectrum) > 10:
            # Calculate variance of differences (periodic = low variance)
            smoothed = np.convolve(power_spectrum, np.ones(3)/3, mode='valid')
            diffs = np.diff(smoothed)
            
            # Look for regular oscillations (sign of upsampling grid)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            regularity = sign_changes / len(diffs) if len(diffs) > 0 else 0
            
            # High regularity (0.4-0.6) is suspicious
            if 0.35 < regularity < 0.65:
                score = 1.0 - abs(regularity - 0.5) / 0.15  # Peak at 0.5
                return min(1.0, score)
        
        return 0.0
