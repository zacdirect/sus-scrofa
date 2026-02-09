"""
Tests for Compliance Audit System (lib/analyzer/auditor.py).

Tests the engine-level auditor module that reads accumulated plugin results
and produces the authoritative authenticity score, AI probability, and
manipulation probability.

Scoring model under test:
    Finding(level, category, description, is_positive=False)
    Point table: LOW=5, MEDIUM=15, HIGH=50
    Base score: 50, positive adds, negative subtracts, clamped 0–100.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.analyzer.auditor import (
    Finding,
    POINTS,
    audit,
    _check_ai_dimensions,
    _check_ai_indicators,
    _check_noise_consistency,
    _check_frequency_analysis,
    _check_ml_model_results,
    _check_photo_resolution,
    _check_legitimate_camera,
    _check_gps_data,
    _check_camera_settings,
    _check_opencv_findings,
    _check_missing_metadata,
    _check_minimal_exif,
    _check_convergent_evidence,
    _calculate_authenticity_score,
)


class FindingAPITestCase(unittest.TestCase):
    """Test the Finding class API (6-point scale)."""

    def test_finding_defaults_to_negative(self):
        f = Finding(Finding.LOW, "Test", "test description")
        self.assertFalse(f.is_positive)

    def test_finding_positive(self):
        f = Finding(Finding.MEDIUM, "Test", "positive", is_positive=True)
        self.assertTrue(f.is_positive)

    def test_finding_has_no_score_impact(self):
        f = Finding(Finding.HIGH, "Test", "desc")
        self.assertFalse(hasattr(f, 'score_impact'))

    def test_finding_has_no_risk_level(self):
        f = Finding(Finding.HIGH, "Test", "desc")
        self.assertFalse(hasattr(f, 'risk_level'))

    def test_point_table(self):
        self.assertEqual(POINTS[Finding.LOW], 5)
        self.assertEqual(POINTS[Finding.MEDIUM], 15)
        self.assertEqual(POINTS[Finding.HIGH], 50)


class ComplianceAuditTestCase(unittest.TestCase):
    """Test compliance audit detection logic."""

    def test_ai_dimension_detection(self):
        """Square power-of-2 dimensions → LOW negative."""
        findings = _check_ai_dimensions(1024, 1024)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.LOW)
        self.assertFalse(findings[0].is_positive)
        self.assertIn("1024", findings[0].description)

    def test_natural_dimensions(self):
        """Natural photo dimensions → no findings."""
        findings = _check_ai_dimensions(4032, 3024)
        self.assertEqual(len(findings), 0)

    def test_ai_filename_detection(self):
        """AI indicator keywords → HIGH negative."""
        test_cases = [
            ("midjourney_art.jpg", True),
            ("dalle_generated.png", True),
            ("stable_diffusion_output.jpg", True),
            ("my_photo.jpg", False),
            ("vacation_2024.png", False),
        ]

        for filename, should_detect in test_cases:
            findings = _check_ai_indicators(filename, {})

            if should_detect:
                self.assertEqual(len(findings), 1, f"Should detect AI in {filename}")
                self.assertEqual(findings[0].level, Finding.HIGH)
                self.assertFalse(findings[0].is_positive)
            else:
                self.assertEqual(len(findings), 0, f"Should not detect AI in {filename}")

    def test_missing_metadata_is_low(self):
        """Missing EXIF → LOW negative (routine stripping)."""
        findings = _check_missing_metadata({})
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.LOW)
        self.assertFalse(findings[0].is_positive)

    def test_minimal_exif_is_low(self):
        """Few EXIF tags → LOW negative."""
        exif = {'Image': {'Software': 'test'}}  # 1 tag < 10
        findings = _check_minimal_exif(exif)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.LOW)

    def test_legitimate_camera_is_medium_positive(self):
        """Verified camera → MEDIUM positive."""
        exif = {'Image': {'Make': 'Apple', 'Model': 'iPhone 12'}}
        findings = _check_legitimate_camera(exif)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.MEDIUM)
        self.assertTrue(findings[0].is_positive)

    def test_gps_data_is_medium_positive(self):
        """GPS present → MEDIUM positive."""
        findings = _check_gps_data({'gps': {'lat': 40.7, 'lon': -74.0}})
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.MEDIUM)
        self.assertTrue(findings[0].is_positive)

    def test_camera_settings_is_medium_positive(self):
        """Realistic camera settings → MEDIUM positive."""
        exif = {'Photo': {
            'ISOSpeedRatings': 100,
            'ExposureTime': '1/125',
            'FNumber': 2.8,
            'FocalLength': 50,
        }}
        findings = _check_camera_settings(exif)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.MEDIUM)
        self.assertTrue(findings[0].is_positive)


class NoiseAnalysisTestCase(unittest.TestCase):
    """Test noise analysis thresholds."""

    def test_very_uniform_noise_is_medium(self):
        """Inconsistency < 2.0 → MEDIUM negative (not HIGH)."""
        findings = _check_noise_consistency({'inconsistency_score': 1.5})
        neg = [f for f in findings if not f.is_positive]
        self.assertEqual(len(neg), 1)
        self.assertEqual(neg[0].level, Finding.MEDIUM)

    def test_low_noise_is_low(self):
        """Inconsistency 2.0–3.5 → LOW negative."""
        findings = _check_noise_consistency({'inconsistency_score': 2.8})
        neg = [f for f in findings if not f.is_positive]
        self.assertEqual(len(neg), 1)
        self.assertEqual(neg[0].level, Finding.LOW)

    def test_natural_noise_is_medium_positive(self):
        """Inconsistency > 5.5 → MEDIUM positive."""
        findings = _check_noise_consistency({'inconsistency_score': 6.0})
        pos = [f for f in findings if f.is_positive]
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0].level, Finding.MEDIUM)

    def test_moderate_noise_is_low_positive(self):
        """Inconsistency 4.5–5.5 → LOW positive."""
        findings = _check_noise_consistency({'inconsistency_score': 5.0})
        pos = [f for f in findings if f.is_positive]
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0].level, Finding.LOW)

    def test_high_anomaly_count_positive(self):
        """Anomaly count > 500 → LOW positive."""
        findings = _check_noise_consistency({
            'inconsistency_score': 5.0,
            'anomaly_count': 800,
        })
        anomaly_f = [f for f in findings if 'anomal' in f.description.lower()
                     and f.is_positive]
        self.assertEqual(len(anomaly_f), 1)
        self.assertEqual(anomaly_f[0].level, Finding.LOW)

    def test_low_anomaly_count_negative(self):
        """Anomaly count < 100 → LOW negative."""
        findings = _check_noise_consistency({
            'inconsistency_score': 5.0,
            'anomaly_count': 50,
        })
        anomaly_f = [f for f in findings if 'anomal' in f.description.lower()
                     and not f.is_positive]
        self.assertEqual(len(anomaly_f), 1)
        self.assertEqual(anomaly_f[0].level, Finding.LOW)

    def test_real_photo_noise_not_penalized(self):
        """Real photo noise ranges should never produce negative findings."""
        real_photo_scores = [4.6, 4.9, 5.1, 5.4, 6.0]

        for score in real_photo_scores:
            findings = _check_noise_consistency({'inconsistency_score': score})
            negative = [f for f in findings if not f.is_positive]
            self.assertEqual(len(negative), 0,
                             f"Real photo score {score} shouldn't be negative")

    def test_ai_image_noise_detected(self):
        """AI image noise scores should produce negative findings."""
        ai_scores = [1.5, 2.0, 2.5, 3.0, 3.4]

        detected_count = 0
        for score in ai_scores:
            findings = _check_noise_consistency({'inconsistency_score': score})
            negative = [f for f in findings if not f.is_positive]
            if negative:
                detected_count += 1

        detection_rate = detected_count / len(ai_scores)
        self.assertGreater(detection_rate, 0.8,
                           f"Detection rate {detection_rate:.0%} too low for AI scores")


class FrequencyAnalysisTestCase(unittest.TestCase):
    """Test frequency/checkerboard analysis."""

    def test_high_checkerboard_is_medium(self):
        """Checkerboard >= 95% → MEDIUM negative."""
        findings = _check_frequency_analysis({'checkerboard_score': 98.0})
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.MEDIUM)
        self.assertFalse(findings[0].is_positive)

    def test_low_checkerboard_is_low_positive(self):
        """Checkerboard < 20% → LOW positive."""
        findings = _check_frequency_analysis({'checkerboard_score': 10.0})
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.LOW)
        self.assertTrue(findings[0].is_positive)

    def test_moderate_checkerboard_no_finding(self):
        """Checkerboard 20–95% → no finding (ambiguous)."""
        findings = _check_frequency_analysis({'checkerboard_score': 50.0})
        self.assertEqual(len(findings), 0)


class MLModelTestCase(unittest.TestCase):
    """Test ML model findings."""

    def test_ml_high_confidence_ai_is_high(self):
        """ML model >80% AI → HIGH negative."""
        layers = [{'method': 'ml_model', 'verdict': 'AI',
                   'score': 0.92, 'confidence': 'HIGH', 'evidence': 'SDXL'}]
        findings = _check_ml_model_results(layers)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.HIGH)
        self.assertFalse(findings[0].is_positive)

    def test_ml_moderate_ai_is_medium(self):
        """ML model 50–80% AI → MEDIUM negative."""
        layers = [{'method': 'ml_model', 'verdict': 'AI',
                   'score': 0.65, 'confidence': 'MEDIUM', 'evidence': 'SDXL'}]
        findings = _check_ml_model_results(layers)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.MEDIUM)
        self.assertFalse(findings[0].is_positive)

    def test_ml_authentic_is_medium_positive(self):
        """ML model says Real with high confidence → MEDIUM positive."""
        layers = [{'method': 'ml_model', 'verdict': 'Real',
                   'score': 0.15, 'confidence': 'HIGH', 'evidence': 'SDXL'}]
        findings = _check_ml_model_results(layers)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Finding.MEDIUM)
        self.assertTrue(findings[0].is_positive)


class PhotoResolutionTestCase(unittest.TestCase):
    """Test photo resolution and aspect ratio checks."""

    def test_natural_aspect_ratio_detected(self):
        """Natural aspect ratio → LOW positive."""
        findings = _check_photo_resolution(4032, 3024)
        aspect = [f for f in findings if "Aspect ratio" in f.description]
        self.assertGreaterEqual(len(aspect), 1)
        self.assertTrue(aspect[0].is_positive)

    def test_perfect_square_no_aspect_bonus(self):
        """Perfect square (1:1) → no aspect ratio bonus."""
        findings = _check_photo_resolution(1024, 1024)
        aspect = [f for f in findings if "Aspect ratio" in f.description]
        self.assertEqual(len(aspect), 0)


class ConvergentEvidenceTestCase(unittest.TestCase):
    """Test cross-detector convergence logic."""

    def test_three_negative_categories_triggers(self):
        """3+ independent negative categories → MEDIUM negative."""
        findings = [
            Finding(Finding.LOW, "Missing Metadata", "no exif"),
            Finding(Finding.LOW, "AI Dimensions", "1024×1024"),
            Finding(Finding.LOW, "Suspicious Noise", "low noise"),
        ]
        extra = _check_convergent_evidence(findings, {})
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0].level, Finding.MEDIUM)
        self.assertFalse(extra[0].is_positive)
        self.assertIn("Convergent", extra[0].category)

    def test_three_positive_categories_triggers(self):
        """3+ independent positive categories → MEDIUM positive."""
        findings = [
            Finding(Finding.MEDIUM, "Legitimate Camera", "Apple", is_positive=True),
            Finding(Finding.MEDIUM, "GPS Data", "present", is_positive=True),
            Finding(Finding.MEDIUM, "Camera Settings", "4/4", is_positive=True),
        ]
        extra = _check_convergent_evidence(findings, {})
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0].level, Finding.MEDIUM)
        self.assertTrue(extra[0].is_positive)

    def test_two_categories_does_not_trigger(self):
        """2 categories → no convergent finding."""
        findings = [
            Finding(Finding.LOW, "Missing Metadata", "no exif"),
            Finding(Finding.LOW, "AI Dimensions", "1024×1024"),
        ]
        extra = _check_convergent_evidence(findings, {})
        self.assertEqual(len(extra), 0)


class AuthenticityScoreTestCase(unittest.TestCase):
    """Test the fixed-point-table scoring."""

    def test_no_findings_gives_50(self):
        """No findings → base score of 50."""
        self.assertEqual(_calculate_authenticity_score([]), 50)

    def test_single_low_negative(self):
        """One LOW negative → 50 - 5 = 45."""
        findings = [Finding(Finding.LOW, "Test", "desc")]
        self.assertEqual(_calculate_authenticity_score(findings), 45)

    def test_single_medium_negative(self):
        """One MEDIUM negative → 50 - 15 = 35."""
        findings = [Finding(Finding.MEDIUM, "Test", "desc")]
        self.assertEqual(_calculate_authenticity_score(findings), 35)

    def test_single_high_negative(self):
        """One HIGH negative → raw 0, ceiling 95, floor 0 → 0."""
        findings = [Finding(Finding.HIGH, "Test", "desc")]
        self.assertEqual(_calculate_authenticity_score(findings), 0)

    def test_single_low_positive(self):
        """One LOW positive → 50 + 5 = 55."""
        findings = [Finding(Finding.LOW, "Test", "desc", is_positive=True)]
        self.assertEqual(_calculate_authenticity_score(findings), 55)

    def test_single_medium_positive(self):
        """One MEDIUM positive → 50 + 15 = 65."""
        findings = [Finding(Finding.MEDIUM, "Test", "desc", is_positive=True)]
        self.assertEqual(_calculate_authenticity_score(findings), 65)

    def test_single_high_positive(self):
        """One HIGH positive → raw 100, floor 5, ceiling 100 → 100."""
        findings = [Finding(Finding.HIGH, "Test", "desc", is_positive=True)]
        self.assertEqual(_calculate_authenticity_score(findings), 100)

    def test_clamped_at_zero(self):
        """Score never goes below 0."""
        findings = [
            Finding(Finding.HIGH, "Test", "desc"),
            Finding(Finding.HIGH, "Test", "desc2"),
        ]
        self.assertEqual(_calculate_authenticity_score(findings), 0)

    def test_clamped_at_100(self):
        """Score never exceeds 100."""
        findings = [
            Finding(Finding.HIGH, "Test", "desc", is_positive=True),
            Finding(Finding.HIGH, "Test", "desc2", is_positive=True),
        ]
        self.assertEqual(_calculate_authenticity_score(findings), 100)

    def test_mixed_findings(self):
        """Mixed positive/negative → 50 + 15 - 5 = 60."""
        findings = [
            Finding(Finding.MEDIUM, "Camera", "camera", is_positive=True),
            Finding(Finding.LOW, "Metadata", "missing"),
        ]
        self.assertEqual(_calculate_authenticity_score(findings), 60)

    def test_ai_image_scoring(self):
        """Typical AI image: missing metadata + AI dims + low noise.

        50 - 5 (LOW) - 5 (LOW) - 15 (MEDIUM noise) = 25
        """
        findings = [
            Finding(Finding.LOW, "Missing Metadata", "no exif"),
            Finding(Finding.LOW, "AI Dimensions", "1024×1024"),
            Finding(Finding.MEDIUM, "Synthetic Noise", "very uniform"),
        ]
        self.assertEqual(_calculate_authenticity_score(findings), 25)

    def test_real_photo_scoring(self):
        """Real photo: camera + GPS + settings + noise.

        50 + 15 + 15 + 15 + 15 = 100 (capped)
        """
        findings = [
            Finding(Finding.MEDIUM, "Legitimate Camera", "Apple", is_positive=True),
            Finding(Finding.MEDIUM, "GPS Data", "present", is_positive=True),
            Finding(Finding.MEDIUM, "Camera Settings", "4/4", is_positive=True),
            Finding(Finding.MEDIUM, "Natural Noise", "high variation", is_positive=True),
        ]
        self.assertEqual(_calculate_authenticity_score(findings), 100)

    def test_high_negative_with_positive_offset(self):
        """HIGH neg + MEDIUM pos: 50 - 50 + 15 = 15."""
        findings = [
            Finding(Finding.HIGH, "AI", "keyword detected"),
            Finding(Finding.MEDIUM, "Camera", "verified", is_positive=True),
        ]
        self.assertEqual(_calculate_authenticity_score(findings), 15)

    def test_one_high_neg_caps_ceiling_at_95(self):
        """1 HIGH negative → ceiling = 95."""
        findings = [
            Finding(Finding.HIGH, "ML", "AI detected"),
            Finding(Finding.MEDIUM, "A", "a", is_positive=True),
            Finding(Finding.MEDIUM, "B", "b", is_positive=True),
            Finding(Finding.MEDIUM, "C", "c", is_positive=True),
            Finding(Finding.MEDIUM, "D", "d", is_positive=True),
            Finding(Finding.MEDIUM, "E", "e", is_positive=True),
            Finding(Finding.MEDIUM, "F", "f", is_positive=True),
            Finding(Finding.MEDIUM, "G", "g", is_positive=True),
            Finding(Finding.MEDIUM, "H", "h", is_positive=True),
        ]
        # Raw: 50 - 50 + 8*15 = 120 → ceiling 95
        self.assertEqual(_calculate_authenticity_score(findings), 95)

    def test_two_high_neg_caps_ceiling_at_90(self):
        """2 HIGH negatives → ceiling = 90."""
        findings = [
            Finding(Finding.HIGH, "ML1", "model 1"),
            Finding(Finding.HIGH, "ML2", "model 2"),
            Finding(Finding.MEDIUM, "A", "a", is_positive=True),
            Finding(Finding.MEDIUM, "B", "b", is_positive=True),
            Finding(Finding.MEDIUM, "C", "c", is_positive=True),
            Finding(Finding.MEDIUM, "D", "d", is_positive=True),
            Finding(Finding.MEDIUM, "E", "e", is_positive=True),
            Finding(Finding.MEDIUM, "F", "f", is_positive=True),
            Finding(Finding.MEDIUM, "G", "g", is_positive=True),
            Finding(Finding.MEDIUM, "H", "h", is_positive=True),
            Finding(Finding.MEDIUM, "I", "i", is_positive=True),
        ]
        # Raw: 50 - 100 + 9*15 = 85, ceiling 90 → 85
        self.assertEqual(_calculate_authenticity_score(findings), 85)

    def test_ten_high_neg_caps_ceiling_at_50(self):
        """10 HIGH negatives → ceiling = 50. Impossible to score well."""
        findings = [Finding(Finding.HIGH, f"T{i}", f"neg {i}") for i in range(10)]
        # Add a pile of positive MEDIUM findings
        findings += [Finding(Finding.MEDIUM, f"P{i}", f"pos {i}",
                             is_positive=True) for i in range(20)]
        # Raw: 50 - 10*50 + 20*15 = 50 - 500 + 300 = -150, ceiling 50, floor 0
        self.assertEqual(_calculate_authenticity_score(findings), 0)

    def test_ten_high_neg_ceiling_clamps_raw_positive(self):
        """10 HIGH negatives → ceiling 50, even with massive positives."""
        findings = [Finding(Finding.HIGH, f"T{i}", f"neg {i}") for i in range(10)]
        findings += [Finding(Finding.HIGH, f"P{i}", f"pos {i}",
                             is_positive=True) for i in range(15)]
        # Raw: 50 - 10*50 + 15*50 = 50 + 250 = 300
        # Ceiling: max(50, 100 - 10*5) = 50
        # Floor: min(50, 15*5) = 50
        # Clamped: max(50, min(50, 300)) = 50
        self.assertEqual(_calculate_authenticity_score(findings), 50)

    def test_one_high_pos_raises_floor_to_5(self):
        """1 HIGH positive → floor = 5."""
        findings = [
            Finding(Finding.HIGH, "Camera", "verified", is_positive=True),
            Finding(Finding.MEDIUM, "A", "a"),
            Finding(Finding.MEDIUM, "B", "b"),
            Finding(Finding.MEDIUM, "C", "c"),
            Finding(Finding.MEDIUM, "D", "d"),
            Finding(Finding.MEDIUM, "E", "e"),
            Finding(Finding.MEDIUM, "F", "f"),
            Finding(Finding.MEDIUM, "G", "g"),
            Finding(Finding.MEDIUM, "H", "h"),
        ]
        # Raw: 50 + 50 - 8*15 = -20 → floor 5
        self.assertEqual(_calculate_authenticity_score(findings), 5)

    def test_high_on_both_sides(self):
        """HIGH on both sides: floor=5, ceiling=95."""
        findings = [
            Finding(Finding.HIGH, "Camera", "verified", is_positive=True),
            Finding(Finding.HIGH, "AI", "keyword detected"),
        ]
        # Raw: 50 + 50 - 50 = 50, within [5, 95]
        self.assertEqual(_calculate_authenticity_score(findings), 50)

    def test_ceiling_never_below_50(self):
        """Ceiling bottoms out at 50 even with 20 HIGH negatives."""
        findings = [Finding(Finding.HIGH, f"T{i}", f"n{i}") for i in range(20)]
        # ceiling = max(50, 100 - 20*5) = max(50, 0) = 50
        # Raw: 50 - 20*50 = -950 → max(0, min(50, -950)) = 0
        self.assertEqual(_calculate_authenticity_score(findings), 0)

    def test_floor_never_above_50(self):
        """Floor tops out at 50 even with 20 HIGH positives."""
        findings = [Finding(Finding.HIGH, f"T{i}", f"p{i}",
                            is_positive=True) for i in range(20)]
        # floor = min(50, 20*5) = min(50, 100) = 50
        # Raw: 50 + 20*50 = 1050 → max(50, min(100, 1050)) = 100
        self.assertEqual(_calculate_authenticity_score(findings), 100)


class IntegrationTestCase(unittest.TestCase):
    """Full audit() integration tests."""

    def test_ai_image_full_audit(self):
        """AI-like image: missing EXIF, square dims, very low noise → low score."""
        results = {
            'file_name': 'generated.jpg',
            'metadata': {
                'dimensions': [1024, 1024],
                'Exif': {},
            },
            'noise_analysis': {
                'inconsistency_score': 1.5,
                'anomaly_count': 40,
            },
        }

        audit_result = audit(results)

        # Should be in the "suspicious" range
        self.assertLessEqual(audit_result['authenticity_score'], 35)
        self.assertGreater(audit_result['findings_count'], 0)

    def test_real_photo_full_audit(self):
        """Real photo: camera EXIF, GPS, settings, natural noise → high score."""
        results = {
            'file_name': 'vacation.jpg',
            'metadata': {
                'dimensions': [4032, 3024],
                'Exif': {
                    'Image': {'Make': 'Apple', 'Model': 'iPhone 12'},
                    'Photo': {
                        'ISOSpeedRatings': 100,
                        'ExposureTime': '1/125',
                        'FNumber': 2.8,
                        'FocalLength': 26,
                    },
                },
                'gps': {'lat': 40.7, 'lon': -74.0},
            },
            'noise_analysis': {
                'inconsistency_score': 5.8,
                'anomaly_count': 1200,
            },
        }

        audit_result = audit(results)

        # Strong positive evidence → high authenticity
        self.assertGreaterEqual(audit_result['authenticity_score'], 70)

    def test_ambiguous_image(self):
        """Ambiguous image: no strong signals → near 50."""
        results = {
            'file_name': 'image.jpg',
            'metadata': {
                'dimensions': [800, 600],
                'Exif': {
                    'Image': {'Software': 'GIMP 2.10'},
                },
            },
        }

        audit_result = audit(results)

        # Should be in the uncertain range
        self.assertGreaterEqual(audit_result['authenticity_score'], 30)
        self.assertLessEqual(audit_result['authenticity_score'], 70)

    def test_findings_summary_uses_new_keys(self):
        """Verify findings_summary has the 6-point-scale keys."""
        results = {
            'file_name': 'test.jpg',
            'metadata': {'dimensions': [1024, 1024], 'Exif': {}},
        }

        audit_result = audit(results)
        summary = audit_result['findings_summary']

        # Should have 6 keys for the 6-point scale
        expected_keys = {'high_neg', 'medium_neg', 'low_neg',
                         'high_pos', 'medium_pos', 'low_pos'}
        self.assertEqual(set(summary.keys()), expected_keys)

    def test_pixel8_with_ml_false_positive(self):
        """Real Pixel 8 photo where ML gives a false positive.

        Positive: camera (+15), GPS (+15), settings (+15),
                  resolution (+5), aspect ratio (+5), noise (+15), anomalies (+5)
                  convergent positive (+15) = +90
        Negative: ML model HIGH (-50)
        Net: 50 + 90 - 50 = 90
        """
        results = {
            'file_name': 'pixel8_photo.jpg',
            'metadata': {
                'dimensions': [3840, 2160],
                'Exif': {
                    'Image': {'Make': 'Google', 'Model': 'Pixel 8 Pro'},
                    'Photo': {
                        'ISOSpeedRatings': 200,
                        'ExposureTime': '1/60',
                        'FNumber': 1.8,
                        'FocalLength': 6.9,
                    },
                },
                'gps': {'lat': 37.4, 'lon': -122.0},
            },
            'noise_analysis': {
                'inconsistency_score': 5.8,
                'anomaly_count': 1600,
            },
            'ai_detection': {
                'enabled': True,
                'detection_layers': [{
                    'method': 'ml_model',
                    'verdict': 'AI',
                    'score': 0.995,
                    'confidence': 'HIGH',
                    'evidence': 'SDXL detector',
                }],
            },
        }

        audit_result = audit(results)

        # With strong positive evidence from camera, GPS, settings,
        # the ML false positive shouldn't tank the score below 50
        self.assertGreater(audit_result['authenticity_score'], 50,
                           "Strong metadata + ML false positive should stay above 50")
        self.assertLess(audit_result['authenticity_score'], 100,
                        "Must not be 100 with an ML HIGH finding")


if __name__ == '__main__':
    unittest.main()
