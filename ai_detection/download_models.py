#!/usr/bin/env python3
"""Download HuggingFace models to a local cache directory.

Usage:
    python download_models.py <model_id> <cache_dir>

Example:
    python download_models.py Organika/sdxl-detector models/Organika-sdxl-detector
"""

import sys
import os


def download_model(model_id: str, cache_dir: str) -> bool:
    """Download a HuggingFace image classification model and its processor."""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return False

    os.makedirs(cache_dir, exist_ok=True)

    print(f"  Downloading model: {model_id}")
    print(f"  Cache directory:   {cache_dir}")
    print()

    try:
        print("  Fetching model weights and config...")
        model = AutoModelForImageClassification.from_pretrained(model_id)
        print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
        
        # Save model directly to cache directory
        print(f"  Saving model to {cache_dir}...")
        model.save_pretrained(cache_dir)
        print("  ✓ Model saved")
    except Exception as e:
        print(f"  ✗ Model download failed: {e}")
        return False

    try:
        print("  Fetching image processor...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        print("  ✓ Image processor loaded")
        
        # Save processor to cache directory
        processor.save_pretrained(cache_dir)
        print("  ✓ Processor saved")
    except Exception as e:
        print(f"  ⚠ Image processor download issue: {e}")
        print("    (This is OK — the pipeline will handle preprocessing)")

    # Quick sanity check: can we load from the saved location?
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "image-classification",
            model=cache_dir,
            device="cpu",
        )
        labels = list(pipe.model.config.id2label.values())
        print(f"  ✓ Pipeline OK — labels: {labels}")
    except Exception as e:
        print(f"  ⚠ Pipeline sanity check failed (model may still work): {e}")

    print()
    print(f"  Done. Model saved in {cache_dir}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    model_id = sys.argv[1]
    cache_dir = sys.argv[2]

    success = download_model(model_id, cache_dir)
    sys.exit(0 if success else 1)
