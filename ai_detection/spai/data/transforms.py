"""
Image Transformations for SPAI Inference

Preprocessing pipeline for input images.
"""

import torch
import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_inference_transform(config):
    """
    Build image transformation pipeline for SPAI inference.
    
    Args:
        config: YACS configuration object
    
    Returns:
        torchvision.transforms.Compose object
    """
    img_size = config.DATA.IMG_SIZE
    interpolation = _str_to_pil_interpolation(config.DATA.INTERPOLATION)
    
    transform = T.Compose([
        T.Resize(img_size, interpolation=interpolation),
        T.CenterCrop(img_size) if config.TEST.CROP else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    
    return transform


def _str_to_pil_interpolation(method):
    """Convert interpolation method string to PIL constant."""
    from PIL import Image
    
    methods = {
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'nearest': Image.NEAREST,
        'lanczos': Image.LANCZOS,
    }
    
    return methods.get(method.lower(), Image.BICUBIC)
