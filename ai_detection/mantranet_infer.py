#!/usr/bin/env python3
"""
Standalone ManTraNet inference script.
Uses modern TensorFlow 2.20 + Keras 3.x implementation.

Based on: https://github.com/ISICV/ManTraNet
"""

import sys
import json
import time
import base64
from pathlib import Path
from io import BytesIO

# Suppress TensorFlow warnings BEFORE importing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: mantranet_infer.py <model_path> <image_path>"}))
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    try:
        import numpy as np
        import tensorflow as tf
        from PIL import Image
        
        # Additional TF logging suppression
        if hasattr(tf, 'get_logger'):
            tf.get_logger().setLevel('ERROR')
        
        # Add ManTraNet src directory to path
        mantranet_src = Path(__file__).parent / "mantranet" / "src"
        if str(mantranet_src) not in sys.path:
            sys.path.insert(0, str(mantranet_src))
        
        start_time = time.time()
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        original_h, original_w = img_array.shape[:2]
        
        # Preprocess: normalize to [-1, 1]
        x = np.expand_dims(img_array.astype('float32') / 255.0 * 2 - 1, axis=0)
        
        # Load model using modern implementation
        load_start = time.time()
        try:
            from modern_mantranet import load_mantranet_pretrained
            
            model_dir = Path(model_path).parent
            model_index = 4  # ManTraNet_Ptrain4.h5
            model = load_mantranet_pretrained(str(model_dir), model_index)
            load_time = time.time() - load_start
        except Exception as e:
            print(json.dumps({"success": False, "error": f"Model loading failed: {str(e)}"}))
            sys.exit(1)
        
        # Run inference
        infer_start = time.time()
        mask = model.predict(x, verbose=0)[0, ..., 0]
        infer_time = time.time() - infer_start
        
        # Resize mask back to original dimensions if needed
        if mask.shape != (original_h, original_w):
            mask_img = Image.fromarray((mask * 255).astype('uint8'))
            mask_img = mask_img.resize((original_w, original_h), Image.BILINEAR)
            mask = np.array(mask_img).astype('float32') / 255.0
        
        # Analyze mask
        analysis = analyze_mask(mask)
        
        # Generate visualization
        mask_bytes = create_mask_visualization(mask)
        
        # Output JSON result
        output = {
            "success": True,
            "analysis": analysis,
            "mask_bytes": base64.b64encode(mask_bytes).decode('ascii'),
            "timing": {
                "load_time": round(load_time, 2),
                "inference_time": round(infer_time, 2),
                "total_time": round(time.time() - start_time, 2)
            },
            "image_shape": [original_h, original_w]
        }
        print(json.dumps(output))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


def analyze_mask(mask):
    """
    Analyze forgery mask to extract statistics.
    
    Args:
        mask: 2D numpy array with values in [0, 1]
        
    Returns:
        Dict with manipulated_percentage, region_count, max_confidence
    """
    import numpy as np
    from scipy import ndimage
    
    # Binary threshold at 0.5
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Calculate manipulated percentage
    total_pixels = mask.size
    manipulated_pixels = np.sum(binary_mask)
    manipulated_pct = (manipulated_pixels / total_pixels) * 100.0
    
    # Find connected components (manipulated regions)
    labeled, region_count = ndimage.label(binary_mask)
    
    # Get max confidence in mask
    max_confidence = float(np.max(mask))
    
    return {
        "manipulated_percentage": round(manipulated_pct, 2),
        "region_count": int(region_count),
        "max_confidence": round(max_confidence, 3),
        "mean_confidence": round(float(np.mean(mask[binary_mask > 0])), 3) if manipulated_pixels > 0 else 0.0
    }


def create_mask_visualization(mask):
    """
    Create a grayscale visualization of the forgery mask.
    
    Uses simple grayscale where black=pristine, white=manipulated.
    This matches the original ManTraNet paper visualization style.
    
    Args:
        mask: 2D numpy array with values in [0, 1]
        
    Returns:
        PNG image bytes
    """
    import numpy as np
    from PIL import Image
    
    # Convert [0, 1] mask directly to grayscale [0, 255]
    # Higher values (closer to 1) = more likely manipulated = brighter/whiter
    # Lower values (closer to 0) = pristine = darker/blacker
    grayscale = (mask * 255).astype(np.uint8)
    
    # Create PIL Image from grayscale array
    img = Image.fromarray(grayscale, mode='L')
    
    # Save to bytes
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


if __name__ == "__main__":
    main()
