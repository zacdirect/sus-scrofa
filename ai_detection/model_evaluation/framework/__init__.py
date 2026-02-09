"""
AI Detection Model Evaluation Framework

This package provides tools for evaluating AI detection models systematically.
"""

__version__ = "1.0.0"

from .base_tester import ModelTester
from .metrics import Metrics, calculate_metrics
from .comparison import ModelComparison

__all__ = [
    'ModelTester',
    'Metrics',
    'calculate_metrics',
    'ModelComparison',
]
