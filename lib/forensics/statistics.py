# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Statistical analysis utilities for forensic detection.
"""

import numpy as np


def calculate_block_variance(data, block_size=32, stride=None):
    """
    Calculate variance in local blocks.
    
    Args:
        data: 2D NumPy array
        block_size: Size of analysis blocks
        stride: Step size between blocks (default: block_size // 2)
        
    Returns:
        Tuple of (variances array, positions list)
    """
    if stride is None:
        stride = block_size // 2
    
    height, width = data.shape
    variances = []
    positions = []
    
    for y in range(0, height - block_size, stride):
        for x in range(0, width - block_size, stride):
            block = data[y:y+block_size, x:x+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                var = np.var(block)
                variances.append(var)
                positions.append((x, y))
    
    return np.array(variances), positions


def detect_outliers(values, threshold_sigma=2.0):
    """
    Detect outliers using standard deviation threshold.
    
    Args:
        values: Array of values
        threshold_sigma: Number of standard deviations for threshold
        
    Returns:
        Tuple of (outlier_indices, outlier_score)
    """
    if len(values) == 0:
        return np.array([]), 0.0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if std_val == 0:
        return np.array([]), 0.0
    
    # Detect outliers on both sides (above and below mean)
    upper_threshold = mean_val + threshold_sigma * std_val
    lower_threshold = mean_val - threshold_sigma * std_val
    outliers = np.where((values > upper_threshold) | (values < lower_threshold))[0]
    
    outlier_score = len(outliers) / len(values) * 100
    
    return outliers, outlier_score


def calculate_entropy(data, bins=256):
    """
    Calculate Shannon entropy of data.
    
    Args:
        data: NumPy array
        bins: Number of histogram bins
        
    Returns:
        Entropy value (float)
    """
    hist, _ = np.histogram(data.flatten(), bins=bins, range=(0, 256))
    hist = hist[hist > 0]  # Remove zero bins
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return entropy
