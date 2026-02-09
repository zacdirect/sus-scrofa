# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Image filtering utilities for forensic analysis.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def extract_noise(image_array, sigma=2):
    """
    Extract noise pattern using high-pass filter.
    
    Args:
        image_array: NumPy array of image data
        sigma: Gaussian blur sigma value
        
    Returns:
        NumPy array containing noise pattern
    """
    denoised = gaussian_filter(image_array.astype(float), sigma=sigma)
    noise = image_array.astype(float) - denoised
    return noise


def get_luminance(image_array):
    """
    Extract luminance channel from RGB image.
    
    Args:
        image_array: NumPy array of image data (H, W, C)
        
    Returns:
        NumPy array of luminance channel (H, W)
    """
    if len(image_array.shape) == 2:
        return image_array
    
    if image_array.shape[2] >= 3:
        # ITU-R BT.601 conversion
        luminance = (0.299 * image_array[:, :, 0] + 
                    0.587 * image_array[:, :, 1] + 
                    0.114 * image_array[:, :, 2])
        return luminance
    
    return image_array[:, :, 0]


def normalize_array(arr, min_val=0, max_val=255):
    """
    Normalize array to specified range.
    
    Args:
        arr: NumPy array
        min_val: Minimum output value
        max_val: Maximum output value
        
    Returns:
        Normalized NumPy array
    """
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.full_like(arr, min_val)
    
    normalized = (arr - arr_min) / (arr_max - arr_min)
    normalized = normalized * (max_val - min_val) + min_val
    return normalized.astype(np.uint8)
