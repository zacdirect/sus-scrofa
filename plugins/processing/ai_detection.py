# Ghiro - Copyright (C) 2013-2026 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
import numpy as np

from lib.analyzer.base import BaseProcessingModule
from lib.utils import str2image
from lib.forensics.filters import get_luminance
from lib.forensics.statistics import calculate_entropy

try:
    from PIL import Image
    IS_PIL = True
except ImportError:
    IS_PIL = False

logger = logging.getLogger(__name__)


class AIArtifactDetection(BaseProcessingModule):
    """Detects artifacts specific to AI-generated images."""

    name = "AI Generation Detection"
    description = "Identifies characteristics of AI-generated content using deterministic analysis."
    order = 30

    def check_deps(self):
        return IS_PIL

    def check_common_ai_dimensions(self, width, height):
        """Check if image has common AI generation dimensions."""
        common_sizes = [
            (512, 512), (768, 768), (1024, 1024),
            (512, 768), (768, 512), (1024, 768), (768, 1024),
            (1024, 1536), (1536, 1024), (2048, 2048)
        ]
        return (width, height) in common_sizes or (height, width) in common_sizes

    def analyze_gradient_smoothness(self, image_array):
        """Analyze gradient distribution - AI images tend to have smoother gradients."""
        # Calculate gradients
        dy = np.diff(image_array, axis=0)
        dx = np.diff(image_array, axis=1)
        
        # Calculate gradient magnitudes
        grad_magnitude = np.sqrt(dy[:, :-1]**2 + dx[:-1, :]**2)
        
        # Calculate statistics
        mean_gradient = np.mean(grad_magnitude)
        std_gradient = np.std(grad_magnitude)
        
        # AI images often have lower variance in gradients
        coefficient_of_variation = std_gradient / (mean_gradient + 1e-10)
        
        # Calculate entropy of gradient histogram
        grad_entropy = calculate_entropy(grad_magnitude.astype(np.uint8))
        
        # Smoother = more likely AI
        # Natural photos typically have CV > 1.5 and entropy > 6
        smoothness_score = 0
        if coefficient_of_variation < 1.2:
            smoothness_score += 30
        if grad_entropy < 5.5:
            smoothness_score += 25
        
        return {
            'coefficient_of_variation': float(coefficient_of_variation),
            'gradient_entropy': float(grad_entropy),
            'smoothness_score': smoothness_score
        }

    def analyze_noise_uniformity(self, image_array):
        """Check for unnaturally uniform noise - characteristic of AI."""
        from lib.forensics.filters import extract_noise
        
        noise = extract_noise(image_array, sigma=1.5)
        
        # Calculate local variance of noise in blocks
        block_size = 64
        height, width = noise.shape
        local_vars = []
        
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = noise[y:y+block_size, x:x+block_size]
                local_vars.append(np.var(block))
        
        if len(local_vars) == 0:
            return {'uniformity_score': 0}
        
        # AI noise is often too uniform (low variance of variances)
        variance_of_variances = np.var(local_vars)
        mean_variance = np.mean(local_vars)
        
        # Coefficient of variation of local variances
        uniformity = mean_variance / (variance_of_variances + 1e-10)
        
        # Higher uniformity = more suspicious
        # Natural photos have more variation in local noise
        uniformity_score = min(uniformity * 20, 100)
        
        return {
            'uniformity_score': float(uniformity_score)
        }

    def check_metadata_indicators(self):
        """Check metadata for AI-related indicators."""
        score = 0
        indicators = []
        
        metadata = self.data.get('metadata', {})
        exif = metadata.get('Exif', {})
        image_meta = exif.get('Image', {})
        
        # Check for missing camera information
        if not image_meta.get('Make'):
            score += 20
            indicators.append("Missing camera manufacturer")
        
        if not image_meta.get('Model'):
            score += 20
            indicators.append("Missing camera model")
        
        # Check software tags for AI tools
        software = image_meta.get('Software', '')
        if software:
            ai_keywords = ['stable diffusion', 'dall-e', 'midjourney', 'neural', 
                          'ai', 'generated', 'diffusion', 'gan']
            if any(keyword in software.lower() for keyword in ai_keywords):
                score += 80
                indicators.append(f"AI software detected: {software}")
        
        # Check for missing lens/GPS data but high quality
        if not exif.get('ExifIFD', {}).get('LensModel') and not metadata.get('GPS'):
            score += 10
            indicators.append("Missing lens and GPS data")
        
        return score, indicators

    def run(self, task):
        try:
            # Load image
            pil_image = str2image(task.get_file_data)
            width, height = pil_image.size
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Work with luminance
            luminance = get_luminance(image_array)
            
            # Initialize AI probability score
            ai_probability = 0
            evidence = []
            
            # Check 1: Common AI dimensions
            if self.check_common_ai_dimensions(width, height):
                ai_probability += 15
                evidence.append(f"Common AI dimensions: {width}x{height}")
            
            # Check 2: Gradient smoothness
            gradient_analysis = self.analyze_gradient_smoothness(luminance)
            ai_probability += gradient_analysis['smoothness_score']
            if gradient_analysis['smoothness_score'] > 30:
                evidence.append(f"Unnaturally smooth gradients (CV: {gradient_analysis['coefficient_of_variation']:.2f})")
            
            # Check 3: Noise uniformity
            noise_analysis = self.analyze_noise_uniformity(luminance)
            noise_score = min(noise_analysis['uniformity_score'], 30)
            ai_probability += noise_score
            if noise_score > 20:
                evidence.append("Suspiciously uniform noise pattern")
            
            # Check 4: Metadata indicators
            metadata_score, metadata_indicators = self.check_metadata_indicators()
            ai_probability += metadata_score
            evidence.extend(metadata_indicators)
            
            # Cap at 100%
            ai_probability = min(ai_probability, 100)
            
            # Store results
            self.results["ai_detection"]["ai_probability"] = float(ai_probability)
            self.results["ai_detection"]["likely_ai"] = ai_probability > 60
            self.results["ai_detection"]["evidence"] = evidence
            self.results["ai_detection"]["gradient_cv"] = gradient_analysis['coefficient_of_variation']
            self.results["ai_detection"]["gradient_entropy"] = gradient_analysis['gradient_entropy']
            self.results["ai_detection"]["noise_uniformity"] = noise_analysis['uniformity_score']
            self.results["ai_detection"]["metadata_score"] = metadata_score
            
            logger.info(f"[Task {task.id}]: AI detection complete - Probability: {ai_probability:.2f}%")
            
        except Exception as e:
            logger.exception(f"[Task {task.id}]: Error in AI detection: {e}")
        
        return self.results
