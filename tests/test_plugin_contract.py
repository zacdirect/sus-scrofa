# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Tests for the plugin data contract and auditor integration.

Ensures all plugins that produce audit_findings follow the standard format.
"""

import unittest
from lib.analyzer.plugin_contract import (
    validate_audit_findings,
    create_finding,
    get_audit_findings,
    VALID_LEVELS,
)


class PluginContractTests(unittest.TestCase):
    """Test the plugin contract validation and helpers."""

    def test_create_finding_valid(self):
        """Valid finding creation should succeed."""
        finding = create_finding(
            level='MEDIUM',
            category='Test Category',
            description='Test description',
            is_positive=True,
            confidence=0.85
        )
        
        self.assertEqual(finding['level'], 'MEDIUM')
        self.assertEqual(finding['category'], 'Test Category')
        self.assertEqual(finding['description'], 'Test description')
        self.assertTrue(finding['is_positive'])
        self.assertEqual(finding['confidence'], 0.85)

    def test_create_finding_invalid_level(self):
        """Invalid level should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_finding('INVALID', 'Cat', 'Desc', False)
        
        self.assertIn('must be one of', str(ctx.exception))

    def test_create_finding_invalid_confidence(self):
        """Invalid confidence should raise ValueError."""
        with self.assertRaises(ValueError):
            create_finding('LOW', 'Cat', 'Desc', False, confidence=1.5)
        
        with self.assertRaises(ValueError):
            create_finding('LOW', 'Cat', 'Desc', False, confidence=-0.1)

    def test_validate_audit_findings_valid(self):
        """Valid findings list should pass validation."""
        findings = [
            {
                'level': 'HIGH',
                'category': 'AI Detection',
                'description': 'Strong AI signature detected',
                'is_positive': False,
            },
            {
                'level': 'LOW',
                'category': 'Metadata',
                'description': 'GPS data present',
                'is_positive': True,
                'confidence': 0.95,
            }
        ]
        
        is_valid, errors = validate_audit_findings(findings, 'test_plugin')
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_audit_findings_missing_fields(self):
        """Missing required fields should fail validation."""
        findings = [
            {
                'level': 'MEDIUM',
                'category': 'Test',
                # missing description and is_positive
            }
        ]
        
        is_valid, errors = validate_audit_findings(findings, 'test_plugin')
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('description' in e for e in errors))
        self.assertTrue(any('is_positive' in e for e in errors))

    def test_validate_audit_findings_invalid_level(self):
        """Invalid level value should fail validation."""
        findings = [
            {
                'level': 'CRITICAL',  # Invalid
                'category': 'Test',
                'description': 'Test',
                'is_positive': False,
            }
        ]
        
        is_valid, errors = validate_audit_findings(findings, 'test_plugin')
        
        self.assertFalse(is_valid)
        self.assertTrue(any('Invalid level' in e for e in errors))

    def test_validate_audit_findings_wrong_types(self):
        """Wrong field types should fail validation."""
        findings = [
            {
                'level': 'LOW',
                'category': 123,  # Should be string
                'description': 'Test',
                'is_positive': 'yes',  # Should be bool
            }
        ]
        
        is_valid, errors = validate_audit_findings(findings, 'test_plugin')
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_get_audit_findings_valid(self):
        """get_audit_findings should extract and validate findings."""
        results = {
            'test_plugin': {
                'audit_findings': [
                    create_finding('LOW', 'Cat1', 'Desc1', False),
                    create_finding('HIGH', 'Cat2', 'Desc2', True, 0.92),
                ]
            }
        }
        
        findings = get_audit_findings(results, 'test_plugin')
        
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0]['level'], 'LOW')
        self.assertEqual(findings[1]['level'], 'HIGH')

    def test_get_audit_findings_invalid(self):
        """get_audit_findings should return empty list for invalid findings."""
        results = {
            'test_plugin': {
                'audit_findings': [
                    {'level': 'INVALID', 'description': 'test'}  # Missing fields
                ]
            }
        }
        
        findings = get_audit_findings(results, 'test_plugin')
        
        # Invalid findings should be rejected
        self.assertEqual(len(findings), 0)

    def test_get_audit_findings_missing_plugin(self):
        """get_audit_findings should handle missing plugin gracefully."""
        results = {}
        
        findings = get_audit_findings(results, 'nonexistent')
        
        self.assertEqual(len(findings), 0)


class PhotoholmesContractComplianceTests(unittest.TestCase):
    """Test that photoholmes plugin follows the contract."""

    def test_photoholmes_produces_valid_findings(self):
        """Photoholmes should produce valid audit_findings structure."""
        # Simulate photoholmes results
        from plugins.ai_ml.photoholmes_detection import PhotoholmesDetector
        
        detector = PhotoholmesDetector()
        
        # Test the _create_audit_findings method directly
        mock_methods = {
            'dq': {
                'detection_score': 0.05,
                'forgery_detected': False,
                'method': 'DQ',
            },
            'zero': {
                'detection_score': 0.02,
                'forgery_detected': False,
                'method': 'ZERO',
            },
            'noisesniffer': {
                'detection_score': 0.85,
                'forgery_detected': True,
                'method': 'Noisesniffer',
            },
        }
        
        findings = detector._create_audit_findings(
            methods_results=mock_methods,
            methods_run=3,
            forgery_count=1,
            avg_score=0.31,
            max_score=0.85
        )
        
        # Should produce findings
        self.assertGreater(len(findings), 0)
        
        # All findings should be valid
        is_valid, errors = validate_audit_findings(findings, 'photoholmes')
        self.assertTrue(is_valid, f"Validation errors: {errors}")
        
        # Should have per-method and possibly consensus findings
        categories = [f['category'] for f in findings]
        self.assertTrue(any('Photoholmes' in c for c in categories))


if __name__ == '__main__':
    unittest.main()
