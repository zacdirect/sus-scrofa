# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
import numpy as np
from scipy.fft import fft2, fftshift

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import str2image, image2str
from lib.db import save_file
from lib.forensics.filters import get_luminance, normalize_array

try:
    from PIL import Image
    IS_PIL = True
except ImportError:
    IS_PIL = False

logger = logging.getLogger(__name__)


class FrequencyAnalysisProcessing(BaseAnalyzerModule):
    """Analyzes frequency domain for manipulation artifacts."""

    name = "Frequency Domain Analysis"
    description = "Uses FFT to detect JPEG ghosts, periodic artifacts, and GAN signatures."
    order = 26

    def check_deps(self):
        return IS_PIL

    def compute_fft_spectrum(self, image_array):
        """Compute FFT magnitude spectrum."""
        # Apply 2D FFT
        fft_result = fft2(image_array)
        fft_shifted = fftshift(fft_result)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_shifted)
        
        # Log scale for better visualization
        magnitude_log = np.log1p(magnitude)
        
        return magnitude_log

    def detect_periodic_patterns(self, fft_magnitude):
        """Detect periodic patterns that may indicate AI generation or manipulation."""
        # Look for peaks in frequency domain (excluding DC component)
        center_y, center_x = np.array(fft_magnitude.shape) // 2
        
        # Mask out center (DC component)
        mask = np.ones_like(fft_magnitude, dtype=bool)
        mask[center_y-10:center_y+10, center_x-10:center_x+10] = False
        
        masked_fft = fft_magnitude.copy()
        masked_fft[~mask] = 0
        
        # Calculate statistics
        mean_val = np.mean(masked_fft[mask])
        std_val = np.std(masked_fft[mask])
        max_val = np.max(masked_fft)
        
        # Detect significant peaks (potential periodic patterns)
        threshold = mean_val + 3 * std_val
        peaks = np.sum(masked_fft > threshold)
        
        # Normalize peak count
        total_pixels = np.sum(mask)
        peak_ratio = peaks / total_pixels * 100
        
        # Check for checkerboard pattern (common in GANs)
        # Look for energy at specific frequencies
        quarter_y, quarter_x = center_y // 2, center_x // 2
        
        # Sample at frequencies that would indicate checkerboard
        checkerboard_regions = [
            fft_magnitude[center_y-quarter_y-5:center_y-quarter_y+5, center_x-quarter_x-5:center_x-quarter_x+5],
            fft_magnitude[center_y-quarter_y-5:center_y-quarter_y+5, center_x+quarter_x-5:center_x+quarter_x+5],
            fft_magnitude[center_y+quarter_y-5:center_y+quarter_y+5, center_x-quarter_x-5:center_x-quarter_x+5],
            fft_magnitude[center_y+quarter_y-5:center_y+quarter_y+5, center_x+quarter_x-5:center_x+quarter_x+5],
        ]
        
        checkerboard_energy = sum([np.mean(region) for region in checkerboard_regions if region.size > 0])
        checkerboard_score = min((checkerboard_energy / (mean_val + 1e-10)) / 4.0 * 100, 100)
        
        return {
            'peak_ratio': float(peak_ratio),
            'max_magnitude': float(max_val),
            'mean_magnitude': float(mean_val),
            'checkerboard_score': float(checkerboard_score)
        }

    def run(self, task):
        try:
            # Load image
            pil_image = str2image(task.get_file_data)
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Work with luminance channel
            luminance = get_luminance(image_array)
            
            # Compute FFT spectrum
            fft_magnitude = self.compute_fft_spectrum(luminance)
            
            # Detect periodic patterns
            pattern_analysis = self.detect_periodic_patterns(fft_magnitude)
            
            # Determine suspiciousness
            suspicious = (pattern_analysis['peak_ratio'] > 0.5 or 
                         pattern_analysis['checkerboard_score'] > 50)
            
            anomaly_score = min(
                pattern_analysis['peak_ratio'] * 10 + 
                pattern_analysis['checkerboard_score'] * 0.5,
                100
            )
            
            # Store results
            self.results["frequency_analysis"]["suspicious"] = suspicious
            self.results["frequency_analysis"]["anomaly_score"] = float(anomaly_score)
            self.results["frequency_analysis"]["peak_ratio"] = pattern_analysis['peak_ratio']
            self.results["frequency_analysis"]["checkerboard_score"] = pattern_analysis['checkerboard_score']
            self.results["frequency_analysis"]["max_magnitude"] = pattern_analysis['max_magnitude']
            
            # Create visualization
            fft_vis = normalize_array(fft_magnitude)
            fft_image = Image.fromarray(fft_vis, mode='L')
            
            # Resize if needed
            width, height = fft_image.size
            if width > 1800:
                fft_image.thumbnail([1800, 1800], Image.Resampling.LANCZOS)
            
            # Save FFT visualization
            img_str = image2str(fft_image)
            self.results["frequency_analysis"]["fft_image_id"] = save_file(img_str, content_type="image/jpeg")
            
            logger.info(f"[Task {task.id}]: Frequency analysis complete - Anomaly score: {anomaly_score:.2f}")
            
        except Exception as e:
            logger.exception(f"[Task {task.id}]: Error in frequency analysis: {e}")
        
        return self.results
