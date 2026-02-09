"""
SPAI Models Module

Extracted from the SPAI repository:
https://github.com/mever-team/spai

Includes:
- Vision Transformer (ViT) backbone
- Spectral Image Detection (SID) models  
- Frequency filtering utilities
- Position embedding utilities
- Alternative backbones (CLIP, DINOv2)

Contains model architecture definitions for SPAI inference.
"""

from .build import build_cls_model

__all__ = ["build_cls_model"]
