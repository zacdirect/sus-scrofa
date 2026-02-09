#!/usr/bin/env python3
"""
Standalone SDXL inference script.

Called by SDXLDetector via subprocess inside the ai_detection venv.
Outputs a single JSON line to stdout.

Usage:
    python sdxl_infer.py <image_path>

Output (JSON):
    {"success": true, "artificial_score": 0.99, "human_score": 0.01,
     "inference_time_s": 1.23}
"""

import json
import sys
import time
from pathlib import Path

MODEL_ID = "Organika/sdxl-detector"
CACHE_DIR = Path(__file__).parent / "models" / "Organika-sdxl-detector"


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: sdxl_infer.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(json.dumps({"success": False, "error": f"Image not found: {image_path}"}))
        sys.exit(1)

    try:
        import os

        # Prefer offline if cache exists (avoids network latency)
        if CACHE_DIR.exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from transformers import pipeline as hf_pipeline
        from PIL import Image

        # Build pipeline (lazy-loaded — fast on subsequent calls if kept warm)
        pipe = hf_pipeline(
            "image-classification",
            model=MODEL_ID,
            device="cpu",
            model_kwargs={"cache_dir": str(CACHE_DIR)},
        )

        # Run inference
        img = Image.open(image_path).convert("RGB")
        start = time.time()
        outputs = pipe(img)
        elapsed = time.time() - start

        scores = {item["label"]: item["score"] for item in outputs}

        print(
            json.dumps(
                {
                    "success": True,
                    "artificial_score": scores.get("artificial", 0.0),
                    "human_score": scores.get("human", 0.0),
                    "inference_time_s": round(elapsed, 3),
                }
            )
        )

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
