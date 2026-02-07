"""
AI Detection - Multi-Layer Detector Framework

This module provides a flexible framework for AI-generated image detection
using multiple complementary detection methods.
"""

from .metadata import MetadataDetector
from .base import BaseDetector, DetectionResult

__all__ = ['MetadataDetector', 'BaseDetector', 'DetectionResult']
