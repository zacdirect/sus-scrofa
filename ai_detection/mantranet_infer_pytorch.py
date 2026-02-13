#!/usr/bin/env python3
"""
ManTraNet Inference Script (PyTorch)

Runs ManTraNet forgery detection using PyTorch implementation.
Outputs JSON with mask analysis results.

Usage:
    python mantranet_infer_pytorch.py <model_path> <image_path>

Output JSON format:
{
    "success": true,
    "analysis": {
        "manipulated_percentage": 8.5,
        "region_count": 3,
        "max_confidence": 0.85
    },
    "mask_bytes": "<base64 encoded PNG>",
    "timing": {"total_time": 5.2}
}
"""

import sys
import json
import time
import base64
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
from PIL import Image

# Add mantranet directory to path
sys.path.insert(0, str(Path(__file__).parent / "mantranet"))

from mantranet import MantraNet


def load_model(weight_path: Path, device: str = 'cpu'):
    """Load pretrained MantraNet model."""
    model = MantraNet(device=device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, max_size: int = 2048):
    """
    Load and preprocess image for ManTraNet.

    Args:
        image_path: Path to image
        max_size: Maximum dimension (width or height). Images larger than this
                  will be resized to improve inference speed.

    Returns:
        torch.Tensor: Image tensor (1, 3, H, W)
        np.ndarray: Original image for visualization
    """
    im = Image.open(image_path)

    # Convert to RGB if needed
    if im.mode != 'RGB':
        im = im.convert('RGB')

    # Resize large images for faster inference
    width, height = im.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        im = im.resize((new_width, new_height), Image.LANCZOS)
        print(f"Resized image from {width}x{height} to {new_width}x{new_height} for faster inference",
              file=sys.stderr)

    original = np.array(im)

    # Convert to tensor and rearrange dimensions
    im_tensor = torch.from_numpy(original).float()
    im_tensor = im_tensor.unsqueeze(0)  # Add batch dimension: (1, H, W, 3)
    im_tensor = im_tensor.permute(0, 3, 1, 2)  # Rearrange to (1, 3, H, W)

    return im_tensor, original


def analyze_mask(mask: np.ndarray, threshold: float = 0.2, min_region_size: int = 100):
    """
    Analyze forgery mask to extract statistics.

    Args:
        mask: 2D numpy array with values 0-1
        threshold: Confidence threshold for forgery detection
        min_region_size: Minimum region size in pixels to consider (filters noise)

    Returns:
        Dict with analysis results
    """
    from scipy import ndimage

    # Binary mask at threshold
    binary_mask = (mask > threshold).astype(np.uint8)

    # Find connected regions
    labeled, num_regions = ndimage.label(binary_mask)

    # Filter out small regions (likely noise from JPEG compression, sensor artifacts)
    filtered_mask = binary_mask.copy()
    if min_region_size > 0:
        for region_id in range(1, num_regions + 1):
            region = (labeled == region_id)
            region_size = np.sum(region)
            if region_size < min_region_size:
                # Remove small noisy regions
                filtered_mask[region] = 0

        # Recalculate regions after filtering
        labeled, num_regions = ndimage.label(filtered_mask)

    # Calculate manipulated percentage using filtered mask
    total_pixels = mask.size
    manipulated_pixels = np.sum(filtered_mask)
    manipulated_pct = (manipulated_pixels / total_pixels) * 100.0

    # Max confidence in mask
    max_confidence = float(np.max(mask))

    return {
        "manipulated_percentage": float(manipulated_pct),
        "region_count": int(num_regions),
        "max_confidence": max_confidence,
        "filtered_mask": filtered_mask  # Return for visualization
    }


def create_overlay(original: np.ndarray, mask: np.ndarray, threshold: float = 0.2, alpha: float = 0.5):
    """
    Create overlay image showing manipulated regions highlighted on original.

    Args:
        original: Original RGB image
        mask: Forgery confidence mask (0-1)
        threshold: Confidence threshold
        alpha: Transparency of red overlay (0-1)

    Returns:
        bytes: PNG image data
    """
    # Create binary mask
    binary_mask = (mask > threshold).astype(bool)

    # Create overlay: original image with red highlight on manipulated regions
    overlay = original.copy()
    overlay[binary_mask] = (
        overlay[binary_mask] * (1 - alpha) +
        np.array([255, 0, 0]) * alpha
    ).astype(np.uint8)

    # Convert to PIL and save
    img = Image.fromarray(overlay)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def mask_to_png_bytes(mask: np.ndarray, threshold: float = 0.2, binary: bool = True):
    """
    Convert forgery mask to PNG bytes.

    Args:
        mask: 2D numpy array with values 0-1
        threshold: Confidence threshold for binary mode
        binary: If True, create black/white mask. If False, use jet colormap.

    Returns:
        bytes: PNG image data
    """
    if binary:
        # Simple binary mask: black = pristine, white = manipulated
        # This is clearer for highlighting regions of concern
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        img = Image.fromarray(binary_mask, mode='L')
    else:
        # Colormap visualization (jet: blue=0, red=1)
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap('jet')
        colored = cmap(mask)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(colored_rgb)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def run_inference(model_path: Path, image_path: Path):
    """
    Run ManTraNet inference and return analysis results.

    Returns:
        Dict with success, analysis, mask_bytes, and timing
    """
    start_time = time.time()

    try:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}", file=sys.stderr)

        # Load model
        t0 = time.time()
        model = load_model(model_path, device=device)
        print(f"Model loaded in {time.time()-t0:.2f}s", file=sys.stderr)

        # Load and preprocess image
        t0 = time.time()
        image_tensor, original = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        print(f"Image preprocessed in {time.time()-t0:.2f}s", file=sys.stderr)

        # Run inference
        t0 = time.time()
        with torch.no_grad():
            output = model(image_tensor)
        print(f"Inference completed in {time.time()-t0:.2f}s", file=sys.stderr)

        # Convert output to numpy (H, W)
        mask = output[0, 0].cpu().numpy()
        print(f"Mask shape: {mask.shape}, Original shape: {original.shape}", file=sys.stderr)

        # Ensure mask and original have matching dimensions
        if mask.shape[:2] != original.shape[:2]:
            print(f"WARNING: Shape mismatch! Resizing mask to match original", file=sys.stderr)
            from scipy import ndimage
            zoom_factors = (original.shape[0] / mask.shape[0],
                          original.shape[1] / mask.shape[1])
            mask = ndimage.zoom(mask, zoom_factors, order=1)

        # Analyze mask with higher threshold to reduce false positives
        # threshold=0.4: Require 40% confidence to flag manipulation (reduces sensor noise)
        # min_region_size=200: Require 200px coherent regions (filters JPEG artifacts)
        print("Analyzing mask...", file=sys.stderr)
        analysis = analyze_mask(mask, threshold=0.4, min_region_size=200)
        print(f"Analysis complete: {analysis['manipulated_percentage']:.2f}% manipulated", file=sys.stderr)

        # Get filtered binary mask (removes small noisy regions)
        # Extract it before it goes into JSON serialization
        filtered_mask_binary = analysis.pop('filtered_mask', None)
        if filtered_mask_binary is None:
            filtered_mask_binary = (mask > 0.4).astype(np.uint8)

        # Create mask visualization: Apply filtering to original confidence values
        # This preserves the confidence information while removing small noise regions
        filtered_mask_float = mask * filtered_mask_binary.astype(float)
        print(f"Filtered mask shape: {filtered_mask_float.shape}", file=sys.stderr)

        # Convert mask to PNG bytes (binary black/white for clarity)
        print("Creating mask PNG...", file=sys.stderr)
        mask_bytes = mask_to_png_bytes(filtered_mask_float, threshold=0.01, binary=True)
        mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')

        # Also create overlay version (mask on original image)
        print("Creating overlay...", file=sys.stderr)
        overlay_bytes = create_overlay(original, filtered_mask_float, threshold=0.01)
        overlay_b64 = base64.b64encode(overlay_bytes).decode('utf-8')

        total_time = time.time() - start_time

        return {
            "success": True,
            "analysis": analysis,
            "mask_bytes": mask_b64,  # Binary black/white mask
            "overlay_bytes": overlay_b64,  # Original image with red highlights
            "timing": {
                "total_time": total_time
            }
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR: {error_trace}", file=sys.stderr)
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace,
            "timing": {
                "total_time": time.time() - start_time
            }
        }


def main():
    if len(sys.argv) != 3:
        print(json.dumps({
            "success": False,
            "error": "Usage: mantranet_infer_pytorch.py <model_path> <image_path>"
        }))
        sys.exit(1)

    model_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])

    if not model_path.exists():
        print(json.dumps({
            "success": False,
            "error": f"Model not found: {model_path}"
        }))
        sys.exit(1)

    if not image_path.exists():
        print(json.dumps({
            "success": False,
            "error": f"Image not found: {image_path}"
        }))
        sys.exit(1)

    result = run_inference(model_path, image_path)
    print(json.dumps(result))

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
