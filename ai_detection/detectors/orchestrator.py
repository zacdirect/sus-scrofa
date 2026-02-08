"""
Multi-layer AI detection orchestrator.

Runs multiple detection methods in sequence and combines results
with a decision logic that prioritizes fast, certain methods.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

from .base import BaseDetector, DetectionResult, ConfidenceLevel, ResultStore
from .metadata import MetadataDetector
from .compliance_audit import ComplianceAuditor
from .spai_detector import SPAIDetector
from .sdxl_detector import SDXLDetector


logger = logging.getLogger(__name__)


class MultiLayerDetector:
    """
    Orchestrates multiple AI detection methods.
    
    Architecture:
    - Orchestrator: Runs detectors in operationally efficient order (fast → slow)
    - Auditor: Separate component that reviews results after each detector
    
    Workflow:
    1. Run detector
    2. Send result to auditor for analysis
    3. Auditor decides: continue or stop early?
    4. After all detectors (or early stop), ask auditor to summarize
    
    The auditor is NOT a detector - it's a decision-maker and aggregator.
    """
    
    def __init__(self, enable_ml: bool = True):
        """
        Initialize orchestrator with detectors and auditor.
        
        Args:
            enable_ml: Whether to enable ML model detection (slower)
        """
        self.detectors: List[BaseDetector] = []
        self.auditor = ComplianceAuditor()  # The gatekeeper
        
        # Register detectors in priority order (fast → slow)
        self._register_detector(MetadataDetector())
        
        if enable_ml:
            self._register_detector(SDXLDetector())
            self._register_detector(SPAIDetector())
        
        logger.info(f"Orchestrator initialized with {len(self.detectors)} detectors")
    
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
        
        # Shared store for this analysis run.
        # Detectors can optionally read from it; the auditor will.
        store = ResultStore()
        
        # Sort detectors by execution order
        sorted_detectors = sorted(self.detectors, key=lambda d: d.get_order())
        
        for detector in sorted_detectors:
            try:
                # Run detector — pass context so it *can* see prior results
                if detector.name == 'MetadataDetector' and original_filename:
                    result = detector.detect(image_path, original_filename=original_filename, context=store)
                else:
                    result = detector.detect(image_path, context=store)
                
                # Record into the shared store
                store.record(detector.name, result)
                logger.debug(f"{detector.name}: {result.evidence}")
                
                # Consult auditor: have we seen enough?
                if early_stop:
                    should_stop = self.auditor.should_stop_early(store.get_all())
                    if should_stop:
                        logger.info(f"Auditor decision: stop early after {detector.name}")
                        break
                    
            except Exception as e:
                logger.error(f"Error running {detector.name}: {e}")
                continue
        
        # Ask auditor to summarize all findings — it reads from the store
        return self._get_audit_summary(image_path, store)
    
    def _get_audit_summary(self, image_path: str, store: ResultStore) -> Dict:
        """
        Ask auditor to provide final summary of all findings.
        
        The auditor reads detector results from the shared ResultStore
        rather than receiving them as a parameter — it's self-contained.
        
        Args:
            image_path: Path to the analyzed image
            store: Shared ResultStore containing all detector results
            
        Returns:
            Dictionary with final verdict and all supporting data
        """
        if not store:
            return self._error_result("No detection methods available")
        
        try:
            # Auditor reads from the store on its own
            audit_result = self.auditor.detect(image_path, context=store)
            
            return {
                'overall_verdict': audit_result.is_fake,
                'overall_confidence': 'AUDIT',
                'overall_score': audit_result.authenticity_score / 100.0,
                'authenticity_score': audit_result.authenticity_score,
                'evidence': audit_result.evidence,
                'detection_method': audit_result.method.value,
                'detected_types': audit_result.detected_types or [],
                'audit_metadata': audit_result.metadata or {},  # Three-bucket probabilities
                'layer_results': store.get_all_serialized(),
                'enabled': True
            }
        except Exception as e:
            logger.error(f"Error getting audit summary: {e}")
            return self._error_result(f"Audit failed: {e}")
    
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
