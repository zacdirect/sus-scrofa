# Ghiro - Copyright (C) 2013-2026 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

import unittest
import os
import numpy as np
from PIL import Image

from lib.forensics.filters import extract_noise, get_luminance, normalize_array
from lib.forensics.statistics import calculate_block_variance, detect_outliers, calculate_entropy
from lib.forensics.confidence import calculate_manipulation_confidence


class ForensicsLibraryTestCase(unittest.TestCase):
    """Test forensics library functions."""

    def setUp(self):
        """Create test image data."""
        # Create a simple test image
        self.test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.test_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    def test_extract_noise(self):
        """Test noise extraction."""
        noise = extract_noise(self.test_gray, sigma=2)
        self.assertEqual(noise.shape, self.test_gray.shape)
        self.assertTrue(isinstance(noise, np.ndarray))

    def test_get_luminance(self):
        """Test luminance extraction."""
        # Test RGB image
        luminance = get_luminance(self.test_image)
        self.assertEqual(luminance.shape, (100, 100))
        
        # Test grayscale image
        luminance_gray = get_luminance(self.test_gray)
        self.assertEqual(luminance_gray.shape, self.test_gray.shape)

    def test_normalize_array(self):
        """Test array normalization."""
        arr = np.array([[0, 50, 100], [150, 200, 255]])
        normalized = normalize_array(arr)
        self.assertEqual(normalized.min(), 0)
        self.assertEqual(normalized.max(), 255)
        self.assertEqual(normalized.dtype, np.uint8)

    def test_calculate_block_variance(self):
        """Test block variance calculation."""
        variances, positions = calculate_block_variance(self.test_gray, block_size=32)
        self.assertTrue(len(variances) > 0)
        self.assertEqual(len(variances), len(positions))

    def test_detect_outliers(self):
        """Test outlier detection."""
        values = np.array([1, 2, 3, 4, 5, 100, 101, 102])
        outliers, score = detect_outliers(values, threshold_sigma=2.0)
        self.assertTrue(len(outliers) > 0)
        self.assertTrue(0 <= score <= 100)

    def test_calculate_entropy(self):
        """Test entropy calculation."""
        entropy = calculate_entropy(self.test_gray)
        self.assertTrue(entropy > 0)
        self.assertTrue(isinstance(entropy, float))

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Test with no detection results
        results = {}
        confidence = calculate_manipulation_confidence(results)
        self.assertIn('manipulation_detected', confidence)
        self.assertIn('confidence_score', confidence)
        
        # Test with ELA results
        results = {
            'ela': {'max_difference': 50}
        }
        confidence = calculate_manipulation_confidence(results)
        self.assertTrue(0 <= confidence['confidence_score'] <= 1.0)


class EnhancedModulesTestCase(unittest.TestCase):
    """Test enhanced processing modules."""

    def test_modules_importable(self):
        """Test that all new modules can be imported."""
        try:
            from plugins.processing import noise_analysis
            from plugins.processing import frequency_analysis
            from plugins.processing import ai_detection
            from plugins.processing import confidence_scoring
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")

    def test_module_dependencies(self):
        """Test module dependency checking."""
        from plugins.processing.noise_analysis import NoiseAnalysisProcessing
        from plugins.processing.frequency_analysis import FrequencyAnalysisProcessing
        from plugins.processing.ai_detection import AIArtifactDetection
        from plugins.processing.confidence_scoring import ConfidenceScoringProcessing
        
        # Check dependencies
        noise_module = NoiseAnalysisProcessing()
        self.assertTrue(noise_module.check_deps())
        
        freq_module = FrequencyAnalysisProcessing()
        self.assertTrue(freq_module.check_deps())
        
        ai_module = AIArtifactDetection()
        self.assertTrue(ai_module.check_deps())
        
        conf_module = ConfidenceScoringProcessing()
        self.assertTrue(conf_module.check_deps())


if __name__ == '__main__':
    unittest.main()
