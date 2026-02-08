"""
Tests for AI Detection Compliance Audit System.

Tests the compliance audit detector with focus on noise analysis integration
and the corrected thresholds for AI detection.
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_detection.detectors.compliance_audit import ComplianceAuditDetector, Finding


class ComplianceAuditTestCase(unittest.TestCase):
    """Test compliance audit detection logic."""

    def setUp(self):
        """Create test fixtures."""
        self.detector = ComplianceAuditDetector()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_image(self, width=1024, height=1024, filename="test.jpg"):
        """Create a test image file."""
        img = Image.new('RGB', (width, height), color='red')
        filepath = os.path.join(self.temp_dir, filename)
        img.save(filepath, 'JPEG')
        return filepath

    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        self.assertEqual(self.detector.name, "Compliance Audit")
        self.assertEqual(self.detector.get_order(), 200)
        self.assertTrue(self.detector.check_deps())

    def test_ai_dimension_detection(self):
        """Test detection of perfect square AI dimensions."""
        # Test 1024x1024 (common AI size)
        findings = self.detector._check_ai_dimensions(1024, 1024)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].risk_level, Finding.MEDIUM)
        self.assertEqual(findings[0].score_impact, -60)
        self.assertIn("1024x1024", findings[0].description)

    def test_natural_dimensions(self):
        """Test that natural photo dimensions don't trigger."""
        # Test 4032x3024 (iPhone 12 Pro)
        findings = self.detector._check_ai_dimensions(4032, 3024)
        self.assertEqual(len(findings), 0)

    def test_ai_filename_detection(self):
        """Test AI indicator detection in filenames."""
        test_cases = [
            ("midjourney_art.jpg", True),
            ("dalle_generated.png", True),
            ("stable_diffusion_output.jpg", True),
            ("my_photo.jpg", False),
            ("vacation_2024.png", False),
        ]
        
        for filename, should_detect in test_cases:
            filepath = self._create_test_image(filename=filename)
            findings = self.detector._check_ai_indicators(filename, filepath)
            
            if should_detect:
                self.assertEqual(len(findings), 1, f"Should detect AI in {filename}")
                self.assertEqual(findings[0].risk_level, Finding.HIGH)
                self.assertEqual(findings[0].score_impact, -100)
            else:
                self.assertEqual(len(findings), 0, f"Should not detect AI in {filename}")

    def test_noise_consistency_high_risk(self):
        """Test noise analysis with synthetic uniform noise (AI-like)."""
        forensics_data = {
            'noise_analysis': {
                'inconsistency_score': 2.5,  # Very uniform (AI-like)
                'mean_variance': 50.0,
                'anomaly_count': 50
            }
        }
        
        findings = self.detector._check_noise_consistency(forensics_data)
        
        # Should have HIGH risk finding for synthetic noise
        high_risk = [f for f in findings if f.risk_level == Finding.HIGH]
        self.assertEqual(len(high_risk), 1)
        self.assertEqual(high_risk[0].score_impact, -70)
        self.assertIn("Unnaturally uniform", high_risk[0].description)
        
        # Should have LOW risk finding for few anomalies
        low_risk = [f for f in findings if f.risk_level == Finding.LOW]
        self.assertEqual(len(low_risk), 1)

    def test_noise_consistency_medium_risk(self):
        """Test noise analysis with moderately uniform noise."""
        forensics_data = {
            'noise_analysis': {
                'inconsistency_score': 4.0,  # Somewhat uniform
                'mean_variance': 80.0,
                'anomaly_count': 150
            }
        }
        
        findings = self.detector._check_noise_consistency(forensics_data)
        
        # Should have MEDIUM risk finding
        medium_risk = [f for f in findings if f.risk_level == Finding.MEDIUM]
        self.assertEqual(len(medium_risk), 1)
        self.assertEqual(medium_risk[0].score_impact, -40)

    def test_noise_consistency_real_photo(self):
        """Test noise analysis with natural sensor noise (real photo)."""
        forensics_data = {
            'noise_analysis': {
                'inconsistency_score': 5.8,  # Natural variation
                'mean_variance': 150.0,
                'anomaly_count': 800
            }
        }
        
        findings = self.detector._check_noise_consistency(forensics_data)
        
        # Should have POSITIVE findings
        positive = [f for f in findings if f.risk_level == Finding.POSITIVE]
        self.assertGreaterEqual(len(positive), 1)
        
        # Check for natural noise pattern finding
        natural_noise = [f for f in positive if "Natural noise" in f.description]
        self.assertEqual(len(natural_noise), 1)
        self.assertEqual(natural_noise[0].score_impact, 35)
        
        # Check for high anomaly count finding
        high_anomalies = [f for f in positive if "Many noise anomalies" in f.description]
        self.assertEqual(len(high_anomalies), 1)

    def test_authenticity_score_calculation(self):
        """Test authenticity score calculation from findings."""
        findings = [
            Finding(Finding.HIGH, "Test", "High risk", -100),
            Finding(Finding.MEDIUM, "Test", "Medium risk", -40),
            Finding(Finding.LOW, "Test", "Low risk", -15),
            Finding(Finding.POSITIVE, "Test", "Positive", +30),
        ]
        
        # Base score: 50
        # Impacts: -100 -40 -15 +30 = -125
        # Risk: 50 + (-125) = -75, clamped to 0
        # Authenticity: 100 - 0 = 100... wait, inverted!
        # Actually: 100 - 0 = 100 (high authenticity when risk is 0)
        # Let me recalculate: risk_score clamped = max(0, min(100, 50-125)) = 0
        # authenticity = 100 - 0 = 100... that's backwards
        # Actually the negative impacts mean HIGH RISK, so LOW AUTHENTICITY
        # risk = 50 + (-125) = -75 → clamp to 0 (no risk)
        # Hmm, this is confusing. Let me think...
        # Negative impacts = suspicious = reduces authenticity
        # Base 50 + (-125) = -75 risk points
        # Clamped to 0-100 risk = 0 (min risk? no...)
        # Wait: if impacts are negative (suspicious), that ADDS to risk
        # So: risk_base = 50, add impacts, clamp, then invert
        # Risk calculation: 50 + (-125) = -75 → clamp 0-100 → 0
        # Authenticity = 100 - 0 = 100 (WRONG!)
        # 
        # The issue: negative impacts should INCREASE risk, not decrease
        # So: risk = 50 - (sum of impacts) or something?
        # Actually looking at code: base_score += finding.score_impact
        # So negative impacts REDUCE base_score, positive INCREASE it
        # Base 50 + (-125) = -75 → clamp to 0
        # Then invert: 100 - 0 = 100
        # That means 0 risk = 100 authenticity? But we have HIGH risk findings!
        # 
        # I think the logic needs fixing. Let me just test what it does:
        score = self.detector._calculate_authenticity_score(findings)
        # With mostly negative impacts, score should be LOW (0-25 range)
        self.assertLess(score, 30)  # Should be AI_GENERATED range

    def test_score_fake_range(self):
        """Test that high risk findings result in low authenticity score (fake)."""
        findings = [
            Finding(Finding.HIGH, "Test", "AI detected", -100),
        ]
        
        score = self.detector._calculate_authenticity_score(findings)
        self.assertLessEqual(score, 40)  # Definitely fake range

    def test_score_uncertain_range(self):
        """Test that mixed findings result in uncertain range."""
        findings = [
            Finding(Finding.MEDIUM, "Test", "Suspicious 1", -20),
            Finding(Finding.POSITIVE, "Test", "Some evidence", +15),
        ]
        
        score = self.detector._calculate_authenticity_score(findings)
        # Should be near neutral
        self.assertGreaterEqual(score, 35)
        self.assertLessEqual(score, 65)

    def test_score_authentic_range(self):
        """Test that strong positive evidence results in high authenticity score."""
        findings = [
            Finding(Finding.POSITIVE, "Test", "Camera signature", +50),
            Finding(Finding.POSITIVE, "Test", "GPS data", +30),
            Finding(Finding.POSITIVE, "Test", "Natural noise", +35),
        ]
        
        score = self.detector._calculate_authenticity_score(findings)
        self.assertGreaterEqual(score, 60)  # Likely or definitely real

    def test_ml_model_detection_override(self):
        """Test that ML model AI detection creates HIGH risk finding."""
        previous_results = [
            {
                'method': 'SPAI_ML_Model',
                'is_ai_generated': True,
                'score': 0.85,
                'confidence': 'HIGH'
            }
        ]
        
        findings = self.detector._check_ml_model_results(previous_results)
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].risk_level, Finding.HIGH)
        self.assertEqual(findings[0].score_impact, -100)

    def test_ml_model_authentic_weak_boost(self):
        """Test that ML model authentic verdict gives small positive boost."""
        previous_results = [
            {
                'method': 'SPAI_ML_Model',
                'is_ai_generated': False,
                'score': 0.15,
                'confidence': 'HIGH'
            }
        ]
        
        findings = self.detector._check_ml_model_results(previous_results)
        
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].risk_level, Finding.POSITIVE)
        self.assertEqual(findings[0].score_impact, 10)  # Low trust boost

    def test_natural_aspect_ratio_detection(self):
        """Test that natural aspect ratios get positive evidence."""
        # Test 4:3 aspect ratio (1.33)
        findings = self.detector._check_photo_resolution(4032, 3024)
        
        # Should have aspect ratio finding
        aspect_findings = [f for f in findings if "Aspect ratio" in f.description]
        self.assertGreaterEqual(len(aspect_findings), 1)
        
        positive = [f for f in findings if f.risk_level == Finding.POSITIVE]
        self.assertGreaterEqual(len(positive), 1)

    def test_perfect_square_rejected(self):
        """Test that perfect square (1:1) doesn't get aspect ratio bonus."""
        findings = self.detector._check_photo_resolution(1024, 1024)
        
        # Should NOT have natural aspect ratio finding for perfect square
        aspect_findings = [f for f in findings if "Aspect ratio" in f.description]
        self.assertEqual(len(aspect_findings), 0)

    def test_integration_ai_image_scenario(self):
        """Test full detection on AI-like image scenario."""
        # Simulate AI image: 1024x1024, low noise inconsistency, no EXIF
        filepath = self._create_test_image(1024, 1024, "generated.jpg")
        
        forensics_data = {
            'noise_analysis': {
                'inconsistency_score': 2.1,
                'mean_variance': 45.0,
                'anomaly_count': 80
            }
        }
        
        result = self.detector.detect(
            filepath, 
            original_filename="generated.jpg",
            forensics_data=forensics_data
        )
        
        # Should detect as fake (low authenticity score)
        self.assertLessEqual(result.authenticity_score, 40)  # Definitely fake
        self.assertTrue(result.is_fake)
        # Should identify it was AI-related detection
        self.assertIsNotNone(result.detected_types)
        self.assertIn('ai_generation', result.detected_types)

    def test_integration_real_photo_scenario(self):
        """Test full detection on real photo scenario."""
        # Simulate real photo: natural dimensions, high noise inconsistency, has EXIF
        filepath = self._create_test_image(4032, 3024, "vacation.jpg")
        
        forensics_data = {
            'noise_analysis': {
                'inconsistency_score': 5.5,
                'mean_variance': 170.0,
                'anomaly_count': 1200
            }
        }
        
        result = self.detector.detect(
            filepath,
            original_filename="vacation.jpg",
            forensics_data=forensics_data
        )
        
        # Should have higher authenticity (natural noise gives positive evidence)
        # May not be fully authentic due to missing EXIF in test image
        self.assertGreaterEqual(result.authenticity_score, 46)  # At least uncertain/likely real
        # is_fake should be False or None (not definitely fake)
        self.assertIn(result.is_fake, [False, None])


class NoiseAnalysisThresholdTestCase(unittest.TestCase):
    """Test that noise analysis thresholds are correctly calibrated."""

    def setUp(self):
        """Initialize detector."""
        self.detector = ComplianceAuditDetector()

    def test_real_photo_noise_ranges(self):
        """Test noise thresholds match real photo characteristics."""
        # Based on ROUND3_ANALYSIS.md findings:
        # Real photos: inconsistency 4.22-5.97
        
        real_photo_scores = [4.22, 4.86, 5.11, 5.35, 5.97]
        
        for score in real_photo_scores:
            forensics = {'noise_analysis': {'inconsistency_score': score}}
            findings = self.detector._check_noise_consistency(forensics)
            
            # Should give positive evidence or be neutral
            negative = [f for f in findings if f.score_impact < 0]
            self.assertEqual(len(negative), 0, 
                           f"Real photo score {score} shouldn't be negative")

    def test_ai_image_noise_ranges(self):
        """Test noise thresholds match AI image characteristics."""
        # Based on ROUND3_ANALYSIS.md findings:
        # AI images: inconsistency 1.91-4.68, median 2.77
        
        ai_scores = [1.91, 2.39, 2.77, 3.90, 4.68]
        
        detected_count = 0
        for score in ai_scores:
            forensics = {'noise_analysis': {'inconsistency_score': score}}
            findings = self.detector._check_noise_consistency(forensics)
            
            # Count how many are flagged as suspicious
            suspicious = [f for f in findings if f.score_impact < 0 and 
                         f.risk_level in [Finding.HIGH, Finding.MEDIUM]]
            if suspicious:
                detected_count += 1
        
        # Should detect at least 60% of AI samples
        detection_rate = detected_count / len(ai_scores)
        self.assertGreater(detection_rate, 0.6,
                          f"Detection rate {detection_rate:.0%} too low")


if __name__ == '__main__':
    unittest.main()
