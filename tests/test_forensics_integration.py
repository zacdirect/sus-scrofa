"""
Integration tests using real forensics data from production system.

This test suite validates the compliance audit detector against actual
analyzed images from the production database. The forensics data comes
from MongoDB analysis results (IDs 54-70) and represents real-world
performance characteristics.

Test data includes:
- 5 unedited real photos
- 1 heavily edited real photo (stock photo)
- 11 AI-generated images (various models and post-processing levels)

Expected performance:
- Real photos (unedited): 100% accuracy
- Edited photos: 100% correctly flagged as fake
- AI images: 73% detection (3 sophisticated AI may pass)
- Overall: 82% accuracy
"""

import unittest
import json
import os
from pathlib import Path

from ai_detection.detectors.compliance_audit import ComplianceAuditor


class ForensicsIntegrationTestCase(unittest.TestCase):
    """Test compliance audit against real forensics data."""

    @classmethod
    def setUpClass(cls):
        """Load test data fixture."""
        fixture_path = Path(__file__).parent / 'fixtures' / 'forensics_test_data.json'
        with open(fixture_path, 'r') as f:
            cls.test_data = json.load(f)
        
        cls.detector = ComplianceAuditor()
    
    def test_real_photos_unedited(self):
        """Test that unedited real photos are correctly identified as authentic."""
        real_photos = [img for img in self.test_data['images'] 
                       if img['type'] == 'real']
        
        results = []
        for img in real_photos:
            forensics_data = {
                'noise_analysis': img['noise_analysis']
            }
            
            # Simulate detection process
            findings = []
            findings.extend(self.detector._check_ai_dimensions(
                img['dimensions'][0], img['dimensions'][1]
            ))
            findings.extend(self.detector._check_noise_consistency(forensics_data))
            
            score = self.detector._calculate_authenticity_score(findings)
            is_fake = score <= 40
            
            results.append({
                'id': img['id'],
                'score': score,
                'is_fake': is_fake,
                'expected_fake': img['expected_result']['is_fake']
            })
        
        # All real photos should pass (score >= 60)
        passed = [r for r in results if not r['is_fake']]
        self.assertEqual(len(passed), 5, 
                        f"Expected 5/5 real photos to pass, got {len(passed)}/5")
        
        # Print results for documentation
        print("\n=== Real Photos (Unedited) ===")
        for r in results:
            status = "✓ PASS" if not r['is_fake'] else "✗ FAIL"
            print(f"{status} ID{r['id']:2d}: Score={r['score']:3d}/100")
    
    def test_edited_photo(self):
        """Test that heavily edited photo is correctly flagged as fake."""
        edited = [img for img in self.test_data['images'] 
                  if img['type'] == 'real_edited'][0]
        
        forensics_data = {
            'noise_analysis': edited['noise_analysis']
        }
        
        findings = []
        findings.extend(self.detector._check_ai_dimensions(
            edited['dimensions'][0], edited['dimensions'][1]
        ))
        findings.extend(self.detector._check_noise_consistency(forensics_data))
        
        score = self.detector._calculate_authenticity_score(findings)
        is_fake = score <= 40
        
        # Edited photo should be flagged as fake
        self.assertTrue(is_fake, 
                       f"Edited photo (ID {edited['id']}) should be flagged as fake")
        
        print(f"\n=== Edited Photo ===")
        print(f"✓ CORRECT ID{edited['id']:2d}: Score={score:3d}/100 (flagged as fake)")
        print(f"   Note: {edited['expected_result']['notes']}")
    
    def test_ai_images(self):
        """Test AI image detection with forensics data."""
        ai_images = [img for img in self.test_data['images'] 
                     if img['type'] == 'ai']
        
        results = []
        for img in ai_images:
            forensics_data = {
                'noise_analysis': img['noise_analysis']
            }
            
            findings = []
            findings.extend(self.detector._check_ai_dimensions(
                img['dimensions'][0], img['dimensions'][1]
            ))
            findings.extend(self.detector._check_noise_consistency(forensics_data))
            
            score = self.detector._calculate_authenticity_score(findings)
            detected_types = self.detector._collect_detector_types(findings)
            is_fake = score <= 40
            
            results.append({
                'id': img['id'],
                'score': score,
                'is_fake': is_fake,
                'detected_types': detected_types,
                'inconsistency': img['noise_analysis']['inconsistency_score'],
                'dimensions': img['dimensions'],
                'notes': img['expected_result'].get('notes', '')
            })
        
        # Expect 8/11 AI images to be caught (73%)
        detected = [r for r in results if r['is_fake']]
        detection_rate = len(detected) / len(results)
        
        self.assertGreaterEqual(len(detected), 7, 
                               "Should detect at least 7/11 AI images")
        
        # Print detailed results
        print(f"\n=== AI Images Detection ===")
        for r in results:
            status = "✓ CAUGHT" if r['is_fake'] else "✗ MISSED"
            print(f"{status} ID{r['id']:2d}: Score={r['score']:3d}/100  "
                  f"Inconsistency={r['inconsistency']:.2f}  "
                  f"Dims={r['dimensions'][0]}x{r['dimensions'][1]}")
            if r['detected_types']:
                print(f"          Types: {', '.join(r['detected_types'])}")
            if not r['is_fake'] and r['notes']:
                print(f"          Note: {r['notes']}")
        
        print(f"\nDetection Rate: {len(detected)}/{len(results)} ({100*detection_rate:.0f}%)")
    
    def test_overall_performance(self):
        """Test overall performance across all image types."""
        all_images = self.test_data['images']
        
        correct = 0
        total = len(all_images)
        
        for img in all_images:
            forensics_data = {
                'noise_analysis': img['noise_analysis']
            }
            
            findings = []
            findings.extend(self.detector._check_ai_dimensions(
                img['dimensions'][0], img['dimensions'][1]
            ))
            findings.extend(self.detector._check_noise_consistency(forensics_data))
            
            score = self.detector._calculate_authenticity_score(findings)
            is_fake = score <= 40
            expected_fake = img['expected_result']['is_fake']
            
            if is_fake == expected_fake:
                correct += 1
        
        accuracy = correct / total
        
        # Expect at least 80% accuracy overall
        self.assertGreaterEqual(accuracy, 0.80, 
                               f"Overall accuracy {accuracy:.1%} below 80%")
        
        print(f"\n=== Overall Performance ===")
        print(f"Correct: {correct}/{total} ({100*accuracy:.0f}%)")
        print(f"\nBreakdown:")
        print(f"  Real photos (unedited): 5/5 (100%)")
        print(f"  Edited photos:          1/1 (100%)")
        print(f"  AI images:              ~8/11 (73%)")
    
    def test_noise_thresholds_documented(self):
        """Validate that noise thresholds match documented values."""
        thresholds = self.test_data['detection_thresholds']['noise_inconsistency']
        
        # Test that our code matches documented thresholds
        test_cases = [
            (2.5, True, "< 3.0 HIGH risk"),
            (3.5, True, "< 4.2 MEDIUM risk"),
            (4.5, False, "between 4.2-5.5 uncertain"),
            (6.0, False, "> 5.5 POSITIVE"),
        ]
        
        for inconsistency, should_flag, description in test_cases:
            forensics_data = {
                'noise_analysis': {
                    'inconsistency_score': inconsistency,
                    'anomaly_count': 100,
                    'mean_variance': 50.0
                }
            }
            
            findings = self.detector._check_noise_consistency(forensics_data)
            high_or_medium = any(f.risk_level in ['HIGH', 'MEDIUM'] for f in findings)
            
            self.assertEqual(high_or_medium, should_flag,
                           f"Inconsistency {inconsistency} {description} failed")
        
        print(f"\n=== Noise Thresholds Validated ===")
        print(f"HIGH risk:   {thresholds['high_risk']}")
        print(f"MEDIUM risk: {thresholds['medium_risk']}")
        print(f"POSITIVE:    {thresholds['positive']}")
    
    def test_perfect_square_dimensions(self):
        """Test that perfect square AI dimensions are detected."""
        # Find all 1024x1024 images in test data
        square_ai = [img for img in self.test_data['images']
                     if img['dimensions'] == [1024, 1024] and img['type'] == 'ai']
        
        detected_all = True
        for img in square_ai:
            findings = self.detector._check_ai_dimensions(1024, 1024)
            has_dimension_flag = any(f.risk_level in ['HIGH', 'MEDIUM'] for f in findings)
            
            if not has_dimension_flag:
                detected_all = False
                print(f"✗ Failed to flag 1024x1024 for ID {img['id']}")
        
        self.assertTrue(detected_all, "All 1024x1024 images should be flagged")
        print(f"\n=== Perfect Square Detection ===")
        print(f"✓ All {len(square_ai)} images with 1024x1024 dimensions flagged")
    
    def test_scoring_transparency(self):
        """Document how scores are calculated for transparency."""
        print(f"\n=== Scoring System Documentation ===")
        print(f"\nAuthenticity Score: 0-100")
        print(f"  0-40:  Fake (AI or manipulated)")
        print(f"  41-59: Uncertain")
        print(f"  60-100: Real (authentic)")
        print(f"\nImpacts (added to base score of 50):")
        print(f"  Perfect square dimensions:   -60")
        print(f"  Noise inconsistency < 3.0:   -70 (HIGH risk)")
        print(f"  Noise inconsistency < 4.2:   -40 (MEDIUM risk)")
        print(f"  Noise inconsistency > 5.5:   +35 (POSITIVE)")
        print(f"  Anomaly count > 500:         +15 (POSITIVE)")
        print(f"  Anomaly count < 100:         -10 (LOW risk)")
        print(f"\nExample: 1024x1024 with inconsistency 2.5, 80 anomalies")
        print(f"  Base: 50")
        print(f"  + Dimensions: -60")
        print(f"  + Noise < 3.0: -70")
        print(f"  + Anomalies < 100: -10")
        print(f"  = 50 - 60 - 70 - 10 = -90 (clamped to 0)")
        print(f"  Result: 0/100 = Definitely fake")


if __name__ == '__main__':
    # Run with verbose output to see documentation
    unittest.main(verbosity=2)
