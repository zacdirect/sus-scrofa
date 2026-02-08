"""
Base classes for AI detection methods.

Each detector implements a specific detection method and returns
standardized results that can be combined.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List
from abc import ABC, abstractmethod


class DetectionMethod(Enum):
    """Type of detection method used."""
    METADATA = "metadata"
    STATISTICAL = "statistical"
    ML_MODEL = "ml_model"
    HEURISTIC = "heuristic"


class ConfidenceLevel(Enum):
    """
    Confidence level for detection methods.
    
    Used by legacy detectors (metadata, SPAI) to indicate how certain
    the detection method is about its result (not the same as authenticity_score).
    """
    CERTAIN = "certain"  # 95%+ confidence (e.g., known AI filename pattern)
    HIGH = "high"  # 75-95% confidence
    MEDIUM = "medium"  # 50-75% confidence
    LOW = "low"  # 25-50% confidence
    NONE = "none"  # Unable to determine


@dataclass
class DetectionResult:
    """
    Standardized result from a detection method.
    
    Two usage patterns:
    1. Individual detectors (metadata, SPAI): Use confidence + score + is_ai_generated
    2. Compliance audit (aggregator): Use authenticity_score + detected_types
    
    Attributes:
        method: Detection method used
        evidence: Human-readable evidence/reasoning
        
        # For individual detectors:
        confidence: How confident the detector is (CERTAIN/HIGH/MEDIUM/LOW/NONE)
        score: 0.0-1.0 AI likelihood score
        is_ai_generated: Boolean verdict (or None if uncertain)
        
        # For compliance audit (aggregator only):
        authenticity_score: 0-100 (0=definitely fake, 100=definitely real)
        detected_types: List of what was detected ["ai_generation", "noise_analysis", etc.]
        
        metadata: Additional method-specific metadata
    """
    method: DetectionMethod
    evidence: str
    
    # Individual detector fields
    confidence: Optional[ConfidenceLevel] = None
    score: Optional[float] = None
    is_ai_generated: Optional[bool] = None
    
    # Compliance audit fields (aggregator)
    authenticity_score: Optional[int] = None  # 0-100
    detected_types: Optional[List[str]] = None
    
    metadata: Dict = None
    
    @property
    def is_fake(self) -> Optional[bool]:
        """Boolean verdict derived from authenticity_score (audit) or is_ai_generated (detector)."""
        if self.authenticity_score is not None:
            # Compliance audit result
            if self.authenticity_score <= 40:
                return True  # Definitely or likely fake
            elif self.authenticity_score >= 60:
                return False  # Likely or definitely real
            else:
                return None  # Uncertain (40-60 range)
        else:
            # Individual detector result
            return self.is_ai_generated
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/display."""
        result = {
            'method': self.method.value,
            'evidence': self.evidence,
            'metadata': self.metadata or {}
        }
        
        # Include fields that are present
        if self.authenticity_score is not None:
            result['authenticity_score'] = self.authenticity_score
            result['is_fake'] = self.is_fake
            result['detected_types'] = self.detected_types or []
        
        if self.confidence is not None:
            result['confidence'] = self.confidence.name
        if self.score is not None:
            result['score'] = self.score
        if self.is_ai_generated is not None:
            result['is_ai_generated'] = self.is_ai_generated
            
        return result


class BaseDetector(ABC):
    """
    Base class for all AI detection methods.
    
    Each detector should:
    1. Be fast and efficient
    2. Return standardized DetectionResult
    3. Handle errors gracefully
    4. Provide clear evidence for decisions
    """
    
    def __init__(self):
        self.enabled = True
        self.name = self.__class__.__name__
    
    @abstractmethod
    def detect(self, image_path: str) -> DetectionResult:
        """
        Analyze an image and return detection result.
        
        Args:
            image_path: Path to image file
            
        Returns:
            DetectionResult with verdict and evidence
        """
        pass
    
    def check_deps(self) -> bool:
        """
        Check if detector dependencies are available.
        
        Returns:
            True if detector can run, False otherwise
        """
        return True
    
    def get_order(self) -> int:
        """
        Get execution order (lower runs first).
        
        Fast metadata checks should run before slow ML models.
        
        Returns:
            Order value (0-100, default 50)
        """
        return 50
