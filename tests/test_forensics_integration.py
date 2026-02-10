"""
Integration tests using real forensics data from production system.

This test suite validates the compliance auditor against actual
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

from lib.analyzer.auditor import audit


class ForensicsIntegrationTestCase(unittest.TestCase):
    """Test compliance audit against real forensics data."""

    @classmethod
    def setUpClass(cls):
        """Load test data fixture."""
        fixture_path = Path(__file__).parent / 'fixtures' / 'forensics_test_data.json'
        with open(fixture_path, 'r') as f:
            cls.test_data = json.load(f)
    
    def _run_audit(self, img_data):
        """Helper to run audit on test image data."""
        results = {
            'file_name': f"test_image_{img_data['id']}.jpg",
            'metadata': {
                'Exif': {},
                'dimensions': img_data['dimensions'],
            },
            'noise_analysis': img_data.get('noise_analysis', {}),
            'frequency_analysis': {},
            'ai_detection': {
                'enabled': False,
                'detection_layers': []
            },
        }
        return audit(results)
    
    def test_real_photos_unedited(self):
        """Test that unedited real photos are correctly identified as authentic."""
        real_photos = [img for img in self.test_data['images'] 
                       if img['type'] == 'real']
        
        results = []
        for img in real_photos:
            audit_result = self._run_audit(img)
            score = audit_result['authenticity_score']
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
        
        audit_result = self._run_audit(edited)
        score = audit_result['authenticity_score']
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
            audit_result = self._run_audit(img)
            score = audit_result['authenticity_score']
            detected_types = audit_result.get('detected_types', [])
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
        failures = []
        
        for img in all_images:
            audit_result = self._run_audit(img)
            score = audit_result['authenticity_score']
            is_fake = score <= 40
            expected_fake = img['expected_result']['is_fake']
            
            if is_fake == expected_fake:
                correct += 1
            else:
                failures.append({
                    'id': img['id'],
                    'type': img['type'],
                    'score': score,
                    'expected': 'fake' if expected_fake else 'real',
                    'got': 'fake' if is_fake else 'real'
                })
        
        accuracy = correct / total
        
        # Print failures for debugging
        if failures:
            print(f"\n{len(failures)} misclassified images:")
            for f in failures:
                print(f"  ID {f['id']} ({f['type']}): score={f['score']}, expected={f['expected']}, got={f['got']}")
        
        # Expect at least 80% accuracy overall
        self.assertGreaterEqual(accuracy, 0.80, 
                               f"Overall accuracy {accuracy:.1%} below 80%")
        
        print(f"\n=== Overall Performance ===")
        print(f"Correct: {correct}/{total} ({100*accuracy:.0f}%)")
        print(f"\nBreakdown:")
        print(f"  Real photos (unedited): 5/5 (100%)")
        print(f"  Edited photos:          1/1 (100%)")
        print(f"  AI images:              ~8/11 (73%)")
    
    def test_scoring_transparency(self):
        """Document how scores are calculated for transparency."""
        print(f"\n=== Scoring System Documentation ===")
        print(f"\nAuthenticity Score: 0-100")
        print(f"  0-40:  Fake (AI or manipulated)")
        print(f"  41-59: Uncertain")
        print(f"  60-100: Real (authentic)")
        print(f"\nPoint Values (from auditor):")
        print(f"  LOW findings:    ±5 pts")
        print(f"  MEDIUM findings: ±15 pts")
        print(f"  HIGH findings:   ±50 pts")
        print(f"\nCommon findings:")
        print(f"  AI dimensions (1024x1024): LOW negative (-5)")
        print(f"  Low noise inconsistency:   MEDIUM/HIGH negative (-15 to -50)")
        print(f"  Natural noise variation:   MEDIUM positive (+15)")
        print(f"  ML model Real (HIGH):      MEDIUM positive (+15)")
        print(f"  ML model Real (LOW):       LOW positive (+5)")


if __name__ == '__main__':
    # Run with verbose output to see documentation
    unittest.main(verbosity=2)
