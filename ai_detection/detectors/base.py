"""
Base classes for AI detection methods.

Each detector implements a specific detection method and returns
standardized results that can be combined.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List
from abc import ABC, abstractmethod


class ConfidenceLevel(Enum):
    """Confidence level in detection result."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CERTAIN = 4


class DetectionMethod(Enum):
    """Type of detection method used."""
    METADATA = "metadata"
    STATISTICAL = "statistical"
    ML_MODEL = "ml_model"
    HEURISTIC = "heuristic"


@dataclass
class DetectionResult:
    """
    Standardized result from a detection method.
    
    Attributes:
        method: Detection method used
        is_ai_generated: Boolean verdict (True = AI, False = Real, None = Unknown)
        confidence: Confidence level in the verdict
        score: Numeric score (0.0-1.0, where 1.0 = definitely AI)
        evidence: Human-readable evidence/reasoning
        metadata: Additional method-specific metadata
    """
    method: DetectionMethod
    is_ai_generated: Optional[bool]
    confidence: ConfidenceLevel
    score: float
    evidence: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/display."""
        return {
            'method': self.method.value,
            'is_ai_generated': self.is_ai_generated,
            'confidence': self.confidence.name,
            'score': float(self.score),
            'evidence': self.evidence,
            'metadata': self.metadata or {}
        }


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
