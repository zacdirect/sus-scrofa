# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
import numpy as np

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import str2image, image2str
from lib.db import save_file
from lib.forensics.filters import extract_noise, get_luminance, normalize_array
from lib.forensics.statistics import calculate_block_variance, detect_outliers

try:
    from PIL import Image, ImageOps
    IS_PIL = True
except ImportError:
    IS_PIL = False

logger = logging.getLogger(__name__)


class NoiseAnalysisProcessing(BaseAnalyzerModule):
    """Analyzes noise patterns to detect manipulation."""

    name = "Noise Pattern Analysis"
    description = "Extracts and analyzes noise patterns to detect inconsistencies indicating manipulation or AI generation."
    order = 25

    def check_deps(self):
        return IS_PIL

    def create_variance_map(self, variances, positions, image_shape, block_size=32):
        """Create heatmap visualization of noise variance."""
        height, width = image_shape[:2]
        variance_map = np.zeros((height, width))
        
        for i, (x, y) in enumerate(positions):
            variance_map[y:y+block_size, x:x+block_size] = variances[i]
        
        # Normalize to 0-255
        variance_map = normalize_array(variance_map)
        
        return variance_map

    def run(self, task):
        try:
            # Load image
            pil_image = str2image(task.get_file_data)
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Handle grayscale
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            
            # Work with luminance channel
            luminance = get_luminance(image_array)
            
            # Extract noise
            noise = extract_noise(luminance, sigma=2)
            
            # Analyze local variance
            variances, positions = calculate_block_variance(noise, block_size=32)
            
            # Detect inconsistencies
            anomalies, inconsistency_score = detect_outliers(variances, threshold_sigma=2.0)
            
            # Store results
            self.results["noise_analysis"]["inconsistency_score"] = float(inconsistency_score)
            self.results["noise_analysis"]["suspicious"] = inconsistency_score > 15.0
            self.results["noise_analysis"]["mean_variance"] = float(np.mean(variances)) if len(variances) > 0 else 0
            self.results["noise_analysis"]["std_variance"] = float(np.std(variances)) if len(variances) > 0 else 0
            self.results["noise_analysis"]["anomaly_count"] = len(anomalies)
            
            # Create and save variance map visualization
            variance_map = self.create_variance_map(variances, positions, image_array.shape)
            
            # Convert to RGB for visualization (using a heatmap colormap)
            variance_image = Image.fromarray(variance_map, mode='L')
            variance_image = ImageOps.colorize(variance_image, 
                                              black='blue', 
                                              white='red',
                                              mid='green')
            
            # Resize if too large
            width, height = variance_image.size
            if width > 1800:
                variance_image.thumbnail([1800, 1800], Image.Resampling.LANCZOS)
            
            # Save variance map
            img_str = image2str(variance_image)
            self.results["noise_analysis"]["variance_map_id"] = save_file(img_str, content_type="image/jpeg")
            
            logger.info(f"[Task {task.id}]: Noise analysis complete - Inconsistency: {inconsistency_score:.2f}%")
            
        except Exception as e:
            logger.exception(f"[Task {task.id}]: Error in noise analysis: {e}")
        
        return self.results
