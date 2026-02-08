"""
Tests for Compliance Audit System (lib/analyzer/auditor.py).

Tests the engine-level auditor module that reads accumulated plugin results
and produces the authoritative authenticity score, AI probability, and
manipulation probability.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.analyzer.auditor import (
    Finding,
    audit,
    _check_ai_dimensions,
    _check_ai_indicators,
    _check_noise_consistency,
    _check_ml_model_results,
    _check_photo_resolution,
    _check_legitimate_camera,
    _check_gps_data,
    _check_camera_settings,
    _check_opencv_findings,
    _calculate_authenticity_score,
)


class ComplianceAuditTestCase(unittest.TestCase):
    """Test compliance audit detection logic."""

    def test_ai_dimension_detection(self):
        """Test detection of perfect square AI dimensions."""
        findings = _check_ai_dimensions(1024, 1024)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].risk_level, Finding.MEDIUM)
        self.assertEqual(findings[0].score_impact, -60)
        self.assertIn("1024x1024", findings[0].description)

    def test_natural_dimensions(self):
        """Test that natural photo dimensions don't trigger."""
        findings = _check_ai_dimensions(4032, 3024)
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
            # New API: pass exif_data dict (empty — no software tag)
            findings = _check_ai_indicators(filename, {})

            if should_detect:
                self.assertEqual(len(findings), 1, f"Should detect AI in {filename}")
                self.assertEqual(findings[0].risk_level, Finding.HIGH)
                self.assertEqual(findings[0].score_impact, -100)
            else:
                self.assertEqual(len(findings), 0, f"Should not detect AI in {filename}")

    def test_noise_consistency_high_risk(self):
        """Test noise analysis with synthetic uniform noise (AI-like)."""
        noise_data = {
            'inconsistency_score': 2.5,
            'mean_variance': 50.0,
            'anomaly_count': 50,
        }

        findings = _check_noise_consistency(noise_data)

        high_risk = [f for f in findings if f.risk_level == Finding.HIGH]
        self.assertEqual(len(high_risk), 1)
        self.assertEqual(high_risk[0].score_impact, -70)
        self.assertIn("Unnaturally uniform", high_risk[0].description)

        low_risk = [f for f in findings if f.risk_level == Finding.LOW]
        self.assertEqual(len(low_risk), 1)

    def test_noise_consistency_medium_risk(self):
        """Test noise analysis with moderately uniform noise."""
        noise_data = {
            'inconsistency_score': 4.0,
            'mean_variance': 80.0,
            'anomaly_count': 150,
        }

        findings = _check_noise_consistency(noise_data)

        medium_risk = [f for f in findings if f.risk_level == Finding.MEDIUM]
        self.assertEqual(len(medium_risk), 1)
        self.assertEqual(medium_risk[0].score_impact, -40)

    def test_noise_consistency_real_photo(self):
        """Test noise analysis with natural sensor noise (real photo)."""
        noise_data = {
            'inconsistency_score': 5.8,
            'mean_variance': 150.0,
            'anomaly_count': 800,
        }

        findings = _check_noise_consistency(noise_data)

        positive = [f for f in findings if f.risk_level == Finding.POSITIVE]
        self.assertGreaterEqual(len(positive), 1)

        natural_noise = [f for f in positive if "Natural noise" in f.description]
        self.assertEqual(len(natural_noise), 1)
        self.assertEqual(natural_noise[0].score_impact, 35)

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

        score = _calculate_authenticity_score(findings)
        self.assertLess(score, 30)

    def test_score_fake_range(self):
        """Test that high risk findings result in low authenticity score."""
        findings = [
            Finding(Finding.HIGH, "Test", "AI detected", -100),
        ]

        score = _calculate_authenticity_score(findings)
        self.assertLessEqual(score, 40)

    def test_score_uncertain_range(self):
        """Test that mixed findings result in uncertain range."""
        findings = [
            Finding(Finding.MEDIUM, "Test", "Suspicious 1", -20),
            Finding(Finding.POSITIVE, "Test", "Some evidence", +15),
        ]

        score = _calculate_authenticity_score(findings)
        self.assertGreaterEqual(score, 35)
        self.assertLessEqual(score, 65)

    def test_score_authentic_range(self):
        """Test that strong positive evidence results in high authenticity."""
        findings = [
            Finding(Finding.POSITIVE, "Test", "Camera signature", +50),
            Finding(Finding.POSITIVE, "Test", "GPS data", +30),
            Finding(Finding.POSITIVE, "Test", "Natural noise", +35),
        ]

        score = _calculate_authenticity_score(findings)
        self.assertGreaterEqual(score, 60)

    def test_ml_model_detection_override(self):
        """Test that ML model AI detection creates HIGH risk finding."""
        detection_layers = [
            {
                'method': 'ml_model',
                'verdict': 'AI',
                'score': 0.85,
                'confidence': 'HIGH',
                'evidence': 'SDXL detector: 85% artificial',
            }
        ]

        findings = _check_ml_model_results(detection_layers)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].risk_level, Finding.HIGH)
        self.assertEqual(findings[0].score_impact, -100)

    def test_ml_model_authentic_weak_boost(self):
        """Test that ML model authentic verdict gives small positive boost."""
        detection_layers = [
            {
                'method': 'ml_model',
                'verdict': 'Real',
                'score': 0.15,
                'confidence': 'HIGH',
                'evidence': 'SDXL detector: 15% artificial',
            }
        ]

        findings = _check_ml_model_results(detection_layers)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].risk_level, Finding.POSITIVE)
        self.assertEqual(findings[0].score_impact, 10)

    def test_natural_aspect_ratio_detection(self):
        """Test that natural aspect ratios get positive evidence."""
        findings = _check_photo_resolution(4032, 3024)

        aspect_findings = [f for f in findings if "Aspect ratio" in f.description]
        self.assertGreaterEqual(len(aspect_findings), 1)

        positive = [f for f in findings if f.risk_level == Finding.POSITIVE]
        self.assertGreaterEqual(len(positive), 1)

    def test_perfect_square_rejected(self):
        """Test that perfect square (1:1) doesn't get aspect ratio bonus."""
        findings = _check_photo_resolution(1024, 1024)

        aspect_findings = [f for f in findings if "Aspect ratio" in f.description]
        self.assertEqual(len(aspect_findings), 0)

    def test_integration_ai_image_scenario(self):
        """Test full audit on AI-like image scenario."""
        results = {
            'file_name': 'generated.jpg',
            'metadata': {
                'dimensions': [1024, 1024],
                'Exif': {},
            },
            'noise_analysis': {
                'inconsistency_score': 2.1,
                'mean_variance': 45.0,
                'anomaly_count': 80,
            },
            'ai_detection': {
                'enabled': True,
                'detection_layers': [],
                'methods_run': [],
            },
        }

        audit_result = audit(results)

        self.assertLessEqual(audit_result['authenticity_score'], 40)
        self.assertIn('ai_generation', audit_result.get('detected_types', []))

    def test_integration_real_photo_scenario(self):
        """Test full audit on real photo scenario."""
        results = {
            'file_name': 'vacation.jpg',
            'metadata': {
                'dimensions': [4032, 3024],
                'Exif': {
                    'Image': {'Make': 'Apple', 'Model': 'iPhone 12'},
                },
                'gps': {'lat': 40.7, 'lon': -74.0},
            },
            'noise_analysis': {
                'inconsistency_score': 5.5,
                'mean_variance': 170.0,
                'anomaly_count': 1200,
            },
            'ai_detection': {
                'enabled': True,
                'detection_layers': [],
                'methods_run': [],
            },
        }

        audit_result = audit(results)

        self.assertGreaterEqual(audit_result['authenticity_score'], 46)


class NoiseAnalysisThresholdTestCase(unittest.TestCase):
    """Test that noise analysis thresholds are correctly calibrated."""

    def test_real_photo_noise_ranges(self):
        """Test noise thresholds match real photo characteristics."""
        real_photo_scores = [4.22, 4.86, 5.11, 5.35, 5.97]

        for score in real_photo_scores:
            findings = _check_noise_consistency({'inconsistency_score': score})

            negative = [f for f in findings if f.score_impact < 0]
            self.assertEqual(len(negative), 0,
                             f"Real photo score {score} shouldn't be negative")

    def test_ai_image_noise_ranges(self):
        """Test noise thresholds match AI image characteristics."""
        ai_scores = [1.91, 2.39, 2.77, 3.90, 4.68]

        detected_count = 0
        for score in ai_scores:
            findings = _check_noise_consistency({'inconsistency_score': score})

            suspicious = [f for f in findings if f.score_impact < 0 and
                          f.risk_level in [Finding.HIGH, Finding.MEDIUM]]
            if suspicious:
                detected_count += 1

        detection_rate = detected_count / len(ai_scores)
        self.assertGreater(detection_rate, 0.6,
                           f"Detection rate {detection_rate:.0%} too low")


class ScoreCappingTestCase(unittest.TestCase):
    """Test that scores never hit 100/0 when contradicting evidence exists."""

    def test_high_risk_caps_ceiling(self):
        """Score never reaches 100 when a HIGH risk finding exists."""
        findings = [
            Finding(Finding.POSITIVE, "Camera", "Verified camera", +50),
            Finding(Finding.POSITIVE, "GPS", "GPS present", +30),
            Finding(Finding.POSITIVE, "Settings", "Camera settings", +25),
            Finding(Finding.POSITIVE, "Resolution", "Standard resolution", +20),
            Finding(Finding.POSITIVE, "Aspect", "Natural aspect ratio", +10),
            Finding(Finding.POSITIVE, "Noise", "Natural noise", +20),
            Finding(Finding.POSITIVE, "Anomalies", "Many anomalies", +15),
            Finding(Finding.HIGH, "ML Model", "ML says AI", -100),
        ]
        score = _calculate_authenticity_score(findings)
        self.assertLess(score, 100, "Score must not be 100 with a HIGH risk finding")
        self.assertEqual(score, 90, "Ceiling should be 90 with one HIGH finding")

    def test_two_high_findings_lower_ceiling_further(self):
        """Two HIGH risk findings cap the ceiling at 80."""
        findings = [
            Finding(Finding.POSITIVE, "Camera", "Verified camera", +50),
            Finding(Finding.POSITIVE, "GPS", "GPS present", +30),
            Finding(Finding.POSITIVE, "Settings", "Camera settings", +25),
            Finding(Finding.HIGH, "ML Model", "ML model 1 says AI", -100),
            Finding(Finding.HIGH, "ML Model", "ML model 2 says AI", -80),
        ]
        score = _calculate_authenticity_score(findings)
        self.assertEqual(score, 15, "Floor should be 15 with 3 positive findings")

    def test_medium_risk_caps_ceiling(self):
        """MEDIUM risk findings lower the ceiling by 5 each."""
        findings = [
            Finding(Finding.POSITIVE, "Camera", "Verified camera", +50),
            Finding(Finding.POSITIVE, "GPS", "GPS present", +30),
            Finding(Finding.POSITIVE, "Noise", "Natural noise", +20),
            Finding(Finding.MEDIUM, "Forensic", "Moderate manipulation", -35),
            Finding(Finding.MEDIUM, "Forensic", "JPEG artifacts", -25),
        ]
        score = _calculate_authenticity_score(findings)
        self.assertLessEqual(score, 90, "Ceiling should account for MEDIUM findings")

    def test_positive_evidence_raises_floor(self):
        """Positive findings prevent the score from hitting 0."""
        findings = [
            Finding(Finding.POSITIVE, "Camera", "Verified camera", +50),
            Finding(Finding.POSITIVE, "GPS", "GPS present", +30),
            Finding(Finding.HIGH, "AI", "Filename says midjourney", -100),
            Finding(Finding.HIGH, "ML", "ML confirms AI", -100),
        ]
        score = _calculate_authenticity_score(findings)
        self.assertEqual(score, 10, "Floor should be 10 with 2 positive findings")

    def test_no_findings_gives_neutral_50(self):
        """No findings at all gives exactly 50 (uncertain)."""
        score = _calculate_authenticity_score([])
        self.assertEqual(score, 50)

    def test_only_positive_findings_still_caps_at_100(self):
        """Pure positive findings with zero risk CAN reach 100."""
        findings = [
            Finding(Finding.POSITIVE, "Camera", "Verified camera", +50),
            Finding(Finding.POSITIVE, "GPS", "GPS present", +30),
        ]
        score = _calculate_authenticity_score(findings)
        self.assertEqual(score, 100, "Pure positive with no risk should reach 100")

    def test_floor_ceiling_bounds(self):
        """Ceiling never drops below 55, floor never rises above 45."""
        findings = [
            Finding(Finding.HIGH, "Test", "Risk 1", -100),
            Finding(Finding.HIGH, "Test", "Risk 2", -100),
            Finding(Finding.HIGH, "Test", "Risk 3", -100),
            Finding(Finding.HIGH, "Test", "Risk 4", -100),
            Finding(Finding.HIGH, "Test", "Risk 5", -100),
            Finding(Finding.HIGH, "Test", "Risk 6", -100),
            Finding(Finding.POSITIVE, "Test", "Pos 1", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 2", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 3", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 4", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 5", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 6", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 7", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 8", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 9", +50),
            Finding(Finding.POSITIVE, "Test", "Pos 10", +50),
        ]
        score = _calculate_authenticity_score(findings)
        self.assertEqual(score, 45)

    def test_real_world_pixel8_with_sdxl_false_positive(self):
        """
        Real-world test: Pixel 8 Pro photo (image 10) where SDXL detector
        gives a 99.5% AI false positive but all other evidence is authentic.

        Before the capping fix, this scored 100/100 (overconfident).
        After: should score 90/100 (HIGH finding caps ceiling at 90).
        """
        findings = [
            Finding(Finding.POSITIVE, "Legitimate Camera",
                    "Verified camera signature: Google Pixel 8 Pro", +50),
            Finding(Finding.POSITIVE, "GPS Data",
                    "GPS location data present (rare in AI-generated images)", +30),
            Finding(Finding.POSITIVE, "Camera Settings",
                    "Realistic camera settings present (4/4 key settings)", +25),
            Finding(Finding.POSITIVE, "Photo Resolution",
                    "Standard camera resolution: 3840x2160 (16:9 aspect ratio)", +20),
            Finding(Finding.POSITIVE, "Natural Aspect Ratio",
                    "Aspect ratio 1.78:1 is typical for photos (not perfect square)", +10),
            Finding(Finding.HIGH, "ML Model Detection",
                    "ML model detected AI generation: 99.5% probability "
                    "(SDXL detector: 99.5% artificial, 0.5% human)", -100),
            Finding(Finding.POSITIVE, "Moderate Noise Variation",
                    "Moderate noise variation (5.11) - likely authentic", +20),
            Finding(Finding.POSITIVE, "High Noise Variation",
                    "Many noise anomalies (1619) - typical of real sensor", +15),
        ]

        score = _calculate_authenticity_score(findings)
        self.assertEqual(score, 90)
        self.assertLess(score, 100, "Must not be 100 with a HIGH risk ML finding")
        self.assertGreater(score, 80, "Strong positive evidence should keep it high")


if __name__ == '__main__':
    unittest.main()
