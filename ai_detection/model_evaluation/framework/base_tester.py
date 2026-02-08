"""
Base class for testing AI detection models.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """Result for a single image."""
    file_path: str
    file_name: str
    category: str  # 'fake', 'edited', 'real'
    expected_ai: bool
    detected_ai: Optional[bool]
    score: Optional[float]
    confidence: str
    evidence: str
    inference_time: float
    error: Optional[str] = None


@dataclass
class CategoryMetrics:
    """Metrics for a single category."""
    category: str
    total: int
    correct: int
    accuracy: float
    avg_score: float
    avg_inference_time: float
    errors: int


@dataclass
class OverallMetrics:
    """Overall evaluation metrics."""
    total_images: int
    correct: int
    incorrect: int
    errors: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    avg_inference_time: float
    total_time: float


class ModelTester:
    """
    Base class for testing AI detection models.
    
    Provides standardized evaluation across different models.
    """
    
    def __init__(
        self,
        model_name: str,
        test_data_dir: Path,
        output_path: Optional[Path] = None
    ):
        """
        Initialize model tester.
        
        Args:
            model_name: Name of the model being tested
            test_data_dir: Path to test data directory
            output_path: Path to save results (optional)
        """
        self.model_name = model_name
        self.test_data_dir = Path(test_data_dir)
        self.output_path = Path(output_path) if output_path else None
        
        # Validate test data structure
        self._validate_test_data()
        
        self.results: List[ImageResult] = []
        
    def _validate_test_data(self):
        """Validate test data directory structure."""
        required_dirs = ['fake', 'real']
        missing = []
        
        for dir_name in required_dirs:
            dir_path = self.test_data_dir / dir_name
            if not dir_path.exists():
                missing.append(dir_name)
        
        if missing:
            raise ValueError(
                f"Missing required directories: {', '.join(missing)}\n"
                f"Expected structure: {self.test_data_dir}/{{fake,edited,real}}/"
            )
    
    def run_evaluation(
        self,
        detector,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run full evaluation on test dataset.
        
        Args:
            detector: Detector instance with detect() method
            verbose: Print progress
            
        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()
        self.results = []
        
        # Test each category
        categories = {
            'fake': True,   # Expected: AI-generated
            'edited': False,  # Expected: Real (human-edited)
            'real': False,  # Expected: Real
        }
        
        for category, expected_ai in categories.items():
            category_dir = self.test_data_dir / category
            
            if not category_dir.exists():
                if verbose:
                    logger.warning(f"Skipping {category}/ - directory not found")
                continue
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Testing {category.upper()} images (Expected: {'AI' if expected_ai else 'Real'})")
                print(f"{'='*70}")
            
            # Find all image files
            image_files = self._find_images(category_dir)
            
            if not image_files:
                logger.warning(f"No images found in {category_dir}")
                continue
            
            if verbose:
                print(f"Found {len(image_files)} images")
            
            # Test each image
            for img_path in sorted(image_files):
                result = self._test_image(
                    detector=detector,
                    image_path=img_path,
                    category=category,
                    expected_ai=expected_ai
                )
                self.results.append(result)
                
                if verbose:
                    status = "✓ CORRECT" if (result.detected_ai == expected_ai) else "✗ WRONG"
                    if result.error:
                        status = "⚠ ERROR"
                    
                    print(
                        f"  {status:12} | "
                        f"Score: {result.score*100 if result.score is not None else 'N/A':>6}% | "
                        f"{result.inference_time:.3f}s | "
                        f"{result.file_name}"
                    )
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(total_time)
        
        # Prepare output
        output = {
            'model_name': self.model_name,
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time': total_time,
            'results': [asdict(r) for r in self.results],
            'metrics': metrics
        }
        
        # Save results
        if self.output_path:
            self._save_results(output)
        
        if verbose:
            self._print_summary(metrics)
        
        return output
    
    def _test_image(
        self,
        detector,
        image_path: Path,
        category: str,
        expected_ai: bool
    ) -> ImageResult:
        """Test a single image."""
        start_time = time.time()
        
        try:
            # Run detection
            result = detector.detect(str(image_path))
            
            inference_time = time.time() - start_time
            
            return ImageResult(
                file_path=str(image_path),
                file_name=image_path.name,
                category=category,
                expected_ai=expected_ai,
                detected_ai=result.is_ai_generated,
                score=result.score,
                confidence=result.confidence.value if result.confidence else 'NONE',
                evidence=result.evidence or '',
                inference_time=inference_time,
                error=None
            )
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"Error testing {image_path.name}: {e}")
            
            return ImageResult(
                file_path=str(image_path),
                file_name=image_path.name,
                category=category,
                expected_ai=expected_ai,
                detected_ai=None,
                score=None,
                confidence='NONE',
                evidence='',
                inference_time=inference_time,
                error=str(e)
            )
    
    def _find_images(self, directory: Path) -> List[Path]:
        """Find all image files in directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        images = []
        
        for ext in extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        
        return sorted(images)
    
    def _calculate_metrics(self, total_time: float) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        # Filter out errors for accuracy calculation
        valid_results = [r for r in self.results if r.detected_ai is not None]
        
        if not valid_results:
            return {
                'error': 'No valid results',
                'total_images': len(self.results),
                'errors': len(self.results)
            }
        
        # Overall metrics
        correct = sum(1 for r in valid_results if r.detected_ai == r.expected_ai)
        incorrect = len(valid_results) - correct
        errors = len(self.results) - len(valid_results)
        
        # Confusion matrix
        true_positives = sum(1 for r in valid_results if r.detected_ai and r.expected_ai)
        false_positives = sum(1 for r in valid_results if r.detected_ai and not r.expected_ai)
        true_negatives = sum(1 for r in valid_results if not r.detected_ai and not r.expected_ai)
        false_negatives = sum(1 for r in valid_results if not r.detected_ai and r.expected_ai)
        
        # Calculate rates
        accuracy = correct / len(valid_results) if valid_results else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0.0
        
        avg_time = sum(r.inference_time for r in self.results) / len(self.results)
        
        # Category-specific metrics
        categories = {}
        for category in set(r.category for r in self.results):
            cat_results = [r for r in valid_results if r.category == category]
            if cat_results:
                cat_correct = sum(1 for r in cat_results if r.detected_ai == r.expected_ai)
                cat_scores = [r.score for r in cat_results if r.score is not None]
                
                categories[category] = {
                    'total': len([r for r in self.results if r.category == category]),
                    'correct': cat_correct,
                    'accuracy': cat_correct / len(cat_results),
                    'avg_score': sum(cat_scores) / len(cat_scores) if cat_scores else 0.0,
                    'avg_inference_time': sum(r.inference_time for r in cat_results) / len(cat_results),
                    'errors': len([r for r in self.results if r.category == category and r.error])
                }
        
        return {
            'overall': {
                'total_images': len(self.results),
                'correct': correct,
                'incorrect': incorrect,
                'errors': errors,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'avg_inference_time': avg_time,
                'total_time': total_time
            },
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            },
            'by_category': categories
        }
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        overall = metrics['overall']
        print(f"\nOverall Performance:")
        print(f"  Total Images: {overall['total_images']}")
        print(f"  Correct:      {overall['correct']}")
        print(f"  Incorrect:    {overall['incorrect']}")
        print(f"  Errors:       {overall['errors']}")
        print(f"  Accuracy:     {overall['accuracy']*100:.2f}%")
        print(f"  Precision:    {overall['precision']*100:.2f}%")
        print(f"  Recall:       {overall['recall']*100:.2f}%")
        print(f"  F1 Score:     {overall['f1_score']*100:.2f}%")
        print(f"  FP Rate:      {overall['false_positive_rate']*100:.2f}%")
        print(f"  FN Rate:      {overall['false_negative_rate']*100:.2f}%")
        print(f"\nPerformance:")
        print(f"  Avg Time:     {overall['avg_inference_time']:.3f}s per image")
        print(f"  Total Time:   {overall['total_time']:.2f}s")
        
        print(f"\nBy Category:")
        for category, cat_metrics in metrics['by_category'].items():
            print(f"  {category.upper()}:")
            print(f"    Accuracy: {cat_metrics['accuracy']*100:.2f}% ({cat_metrics['correct']}/{cat_metrics['total']})")
            print(f"    Avg Score: {cat_metrics['avg_score']*100:.2f}%")
            print(f"    Avg Time: {cat_metrics['avg_inference_time']:.3f}s")
            if cat_metrics['errors'] > 0:
                print(f"    Errors: {cat_metrics['errors']}")
        
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"                Predicted AI    Predicted Real")
        print(f"  Actual AI     {cm['true_positives']:>12}    {cm['false_negatives']:>14}")
        print(f"  Actual Real   {cm['false_positives']:>12}    {cm['true_negatives']:>14}")
    
    def _save_results(self, output: Dict[str, Any]):
        """Save results to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {self.output_path}")
