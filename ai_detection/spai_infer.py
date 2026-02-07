#!/usr/bin/env python3
"""
Standalone SPAI inference script.
Run from ai_detection venv to perform AI detection on an image.
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from spai.inference import SPAIDetector

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: spai_infer.py <weights_path> <image_path>"}))
        sys.exit(1)
    
    weights_path = sys.argv[1]
    image_path = sys.argv[2]
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        detector = SPAIDetector(
            weights_path=weights_path,
            device=device
        )
        
        result = detector.predict(image_path)
        
        # Output JSON result
        output = {
            "success": True,
            "score": float(result['score']),
            "logit": float(result['logit']),
            "confidence": float(result['confidence']),
            "is_ai_generated": bool(result['is_ai_generated']),
            "device": device
        }
        print(json.dumps(output))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
