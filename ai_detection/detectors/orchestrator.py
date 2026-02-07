"""
Multi-layer AI detection orchestrator.

Runs multiple detection methods in sequence and combines results
with a decision logic that prioritizes fast, certain methods.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

from .base import BaseDetector, DetectionResult, ConfidenceLevel
from .metadata import MetadataDetector
from .spai_detector import SPAIDetector


logger = logging.getLogger(__name__)


class MultiLayerDetector:
    """
    Orchestrates multiple AI detection methods.
    
    Strategy:
    1. Run fast, certain methods first (metadata)
    2. If high confidence result found, stop early
    3. Otherwise continue to slower methods (ML models)
    4. Combine results with weighted decision logic
    """
    
    def __init__(self, enable_ml: bool = True):
        """
        Initialize detector with available methods.
        
        Args:
            enable_ml: Whether to enable ML model detection (slower)
        """
        self.detectors: List[BaseDetector] = []
        
        # Register detectors in priority order
        self._register_detector(MetadataDetector())
        
        if enable_ml:
            self._register_detector(SPAIDetector())
    
    def _register_detector(self, detector: BaseDetector):
        """Register a detector if its dependencies are available."""
        if detector.check_deps():
            self.detectors.append(detector)
            logger.info(f"Registered detector: {detector.name}")
        else:
            logger.warning(f"Detector {detector.name} disabled - dependencies not met")
    
    def detect(self, image_path: str, early_stop: bool = True, original_filename: str = None) -> Dict:
        """
        Run multi-layer detection on an image.
        
        Args:
            image_path: Path to image file (may be temporary file)
            early_stop: Stop at first high-confidence result
            original_filename: Original filename (for pattern matching when using temp files)
            
        Returns:
            Dictionary with combined results:
            - overall_verdict: Final decision (True/False/None)
            - overall_confidence: Combined confidence level
            - overall_score: Combined score (0.0-1.0)
            - evidence: Human-readable explanation
            - layer_results: Individual detector results
        """
        if not Path(image_path).exists():
            return self._error_result("Image file not found")
        
        # Sort detectors by execution order
        sorted_detectors = sorted(self.detectors, key=lambda d: d.get_order())
        
        layer_results = []
        
        for detector in sorted_detectors:
            try:
                # Pass original filename to metadata detector
                if detector.name == 'MetadataDetector' and original_filename:
                    result = detector.detect(image_path, original_filename=original_filename)
                else:
                    result = detector.detect(image_path)
                layer_results.append(result)
                
                logger.debug(f"{detector.name}: {result.evidence}")
                
                # Early stopping: if we have high confidence, stop
                if early_stop and result.confidence in [ConfidenceLevel.CERTAIN, ConfidenceLevel.HIGH]:
                    logger.info(f"Early stop: {detector.name} provided {result.confidence.name} confidence")
                    break
                    
            except Exception as e:
                logger.error(f"Error running {detector.name}: {e}")
                continue
        
        # Combine results
        return self._combine_results(layer_results)
    
    def _combine_results(self, results: List[DetectionResult]) -> Dict:
        """
        Combine results from multiple detectors into final verdict.
        
        Decision logic:
        1. If any CERTAIN verdict, use that
        2. If any HIGH confidence, use that
        3. Otherwise, weighted average of scores
        """
        if not results:
            return self._error_result("No detection methods available")
        
        # Check for certain/high confidence results
        for conf_level in [ConfidenceLevel.CERTAIN, ConfidenceLevel.HIGH]:
            certain_results = [r for r in results if r.confidence == conf_level]
            if certain_results:
                # Use first certain/high confidence result
                primary = certain_results[0]
                return {
                    'overall_verdict': primary.is_ai_generated,
                    'overall_confidence': primary.confidence.name,
                    'overall_score': primary.score,
                    'evidence': primary.evidence,
                    'detection_method': primary.method.value,
                    'layer_results': [r.to_dict() for r in results],
                    'enabled': True
                }
        
        # No high confidence results - combine scores
        scored_results = [r for r in results if r.score > 0]
        
        if not scored_results:
            # No useful results
            return {
                'overall_verdict': None,
                'overall_confidence': 'NONE',
                'overall_score': 0.0,
                'evidence': 'Unable to determine - no strong signals detected',
                'detection_method': 'combined',
                'layer_results': [r.to_dict() for r in results],
                'enabled': True
            }
        
        # Weighted average (give more weight to higher confidence results)
        weights = {
            ConfidenceLevel.CERTAIN: 1.0,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.NONE: 0.0
        }
        
        total_weight = sum(weights[r.confidence] for r in scored_results)
        if total_weight == 0:
            weighted_score = sum(r.score for r in scored_results) / len(scored_results)
        else:
            weighted_score = sum(r.score * weights[r.confidence] for r in scored_results) / total_weight
        
        # Determine overall confidence from weighted score
        if weighted_score > 0.7:
            overall_conf = ConfidenceLevel.MEDIUM
        elif weighted_score > 0.3:
            overall_conf = ConfidenceLevel.LOW
        else:
            overall_conf = ConfidenceLevel.NONE
        
        verdict = weighted_score > 0.5 if weighted_score != 0.5 else None
        
        # Build evidence summary
        evidence_parts = [f"{r.method.value}: {r.evidence}" for r in scored_results]
        evidence = " | ".join(evidence_parts)
        
        return {
            'overall_verdict': verdict,
            'overall_confidence': overall_conf.name,
            'overall_score': float(weighted_score),
            'evidence': f"Combined analysis: {evidence}",
            'detection_method': 'multi-layer',
            'layer_results': [r.to_dict() for r in results],
            'enabled': True
        }
    
    def _error_result(self, error_msg: str) -> Dict:
        """Return error result."""
        return {
            'overall_verdict': None,
            'overall_confidence': 'NONE',
            'overall_score': 0.0,
            'evidence': error_msg,
            'detection_method': 'none',
            'layer_results': [],
            'enabled': False,
            'error': error_msg
        }
