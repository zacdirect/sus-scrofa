"""
Image Transformations for SPAI Inference

Preprocessing pipeline for input images.

CRITICAL: The MFViT model expects pixels in [0, 1] range. It applies its own
internal normalization (backbone_norm) AFTER FFT frequency decomposition.
Do NOT apply ImageNet normalization here — that was the root cause of the
double-normalization bug that killed all detections.

Reference: sid.py MFViT.forward() → FFT → clamp(0,1) → self.backbone_norm(x)
Reference: spai.yaml REQUIRED_NORMALIZATION = "positive_0_1"
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def build_inference_transform(config):
    """
    Build image transformation pipeline for SPAI inference.

    For arbitrary resolution mode (ORIGINAL_RESOLUTION=True):
        - Keep original image dimensions
        - Pad to minimum IMG_SIZE (224) if either dimension is smaller
        - Convert to tensor [0, 1] range — NO ImageNet normalization

    For fixed resolution mode (ORIGINAL_RESOLUTION=False):
        - Resize + center crop to IMG_SIZE
        - Convert to tensor [0, 1] range — NO ImageNet normalization

    Args:
        config: YACS configuration object

    Returns:
        callable transform
    """
    img_size = config.DATA.IMG_SIZE
    original_resolution = getattr(config.TEST, 'ORIGINAL_RESOLUTION', True)

    if original_resolution:
        # Arbitrary resolution: preserve original size, pad small images
        return _ArbitraryResolutionTransform(min_size=img_size)
    else:
        # Fixed resolution: resize + crop to img_size
        interpolation = _str_to_pil_interpolation(config.DATA.INTERPOLATION)
        return T.Compose([
            T.Resize(img_size, interpolation=interpolation),
            T.CenterCrop(img_size) if config.TEST.CROP else T.Lambda(lambda x: x),
            T.ToTensor(),
            # NO T.Normalize — model normalizes internally after FFT
        ])


class _ArbitraryResolutionTransform:
    """
    Transform for arbitrary resolution inference.

    Converts PIL image to [0,1] tensor. If either dimension is smaller than
    min_size, pads with zeros (black) to reach min_size. This ensures the
    PatchBasedMFViT gets at least one full 224x224 patch.
    """

    def __init__(self, min_size=224):
        self.min_size = min_size

    def __call__(self, image):
        tensor = TF.to_tensor(image)  # [C, H, W] in [0, 1]

        _, h, w = tensor.shape
        pad_h = max(0, self.min_size - h)
        pad_w = max(0, self.min_size - w)

        if pad_h > 0 or pad_w > 0:
            # Pad right and bottom with zeros
            tensor = TF.pad(tensor, [0, 0, pad_w, pad_h], fill=0)

        return tensor


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
