#!/usr/bin/env python3
"""
Download pretrained models required by the Phase 1c research plugin.

This script pre-populates the same caches that torchvision and OpenCV
use at runtime, so the first analysis doesn't stall on network I/O.

Models downloaded:
    1. FasterRCNN_ResNet50_FPN_V2 (COCO)  — ~168 MB  → torch hub cache
    2. KeypointRCNN_ResNet50_FPN  (COCO)  — ~227 MB  → torch hub cache
    3. YuNet face detector (ONNX)         — ~228 KB  → ~/.cache/opencv_models/

Usage:
    python scripts/download_research_models.py           # download all
    python scripts/download_research_models.py --verify  # check only
    python scripts/download_research_models.py --clean   # remove cached models
"""

import argparse
import os
import sys

# ── YuNet constants (must match image_content_analysis.py) ────────
YUNET_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "opencv_models")
YUNET_MODEL_PATH = os.path.join(YUNET_MODEL_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# ANSI colours (disabled if not a tty)
if sys.stdout.isatty():
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    RESET = "\033[0m"
else:
    GREEN = YELLOW = RED = DIM = RESET = ""


def _sizeof_fmt(num):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


# ── Torchvision models ────────────────────────────────────────────

def _get_torch_hub_dir():
    """Return the torch hub checkpoints directory."""
    import torch.hub
    return os.path.join(torch.hub.get_dir(), "checkpoints")


def _torchvision_model_info():
    """Return (name, url, expected_filename) for each torchvision model."""
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        KeypointRCNN_ResNet50_FPN_Weights,
    )
    w_faster = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    w_kp = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    return [
        ("FasterRCNN_ResNet50_FPN_V2 (COCO)", w_faster.url),
        ("KeypointRCNN_ResNet50_FPN (COCO)", w_kp.url),
    ]


def download_torchvision_models():
    """Download torchvision pretrained weights via torch.hub (idempotent)."""
    import torch.hub

    models = _torchvision_model_info()
    hub_dir = _get_torch_hub_dir()

    for name, url in models:
        filename = os.path.basename(url)
        local_path = os.path.join(hub_dir, filename)

        if os.path.isfile(local_path):
            size = os.path.getsize(local_path)
            print(f"  {GREEN}✓{RESET} {name}  {DIM}({_sizeof_fmt(size)}, cached){RESET}")
        else:
            print(f"  {YELLOW}↓{RESET} {name}  {DIM}downloading…{RESET}")
            # torch.hub.load_state_dict_from_url handles progress bar,
            # caching, and hash verification automatically.
            torch.hub.load_state_dict_from_url(url, map_location="cpu")
            size = os.path.getsize(local_path)
            print(f"  {GREEN}✓{RESET} {name}  {DIM}({_sizeof_fmt(size)}){RESET}")


def verify_torchvision_models():
    """Check that torchvision weights exist in cache."""
    models = _torchvision_model_info()
    hub_dir = _get_torch_hub_dir()
    ok = True
    for name, url in models:
        filename = os.path.basename(url)
        local_path = os.path.join(hub_dir, filename)
        if os.path.isfile(local_path):
            size = os.path.getsize(local_path)
            print(f"  {GREEN}✓{RESET} {name}  {DIM}({_sizeof_fmt(size)}){RESET}")
        else:
            print(f"  {RED}✗{RESET} {name}  {DIM}(not downloaded){RESET}")
            ok = False
    return ok


def clean_torchvision_models():
    """Remove cached torchvision weights."""
    models = _torchvision_model_info()
    hub_dir = _get_torch_hub_dir()
    for name, url in models:
        filename = os.path.basename(url)
        local_path = os.path.join(hub_dir, filename)
        if os.path.isfile(local_path):
            os.remove(local_path)
            print(f"  {YELLOW}✗{RESET} Removed {name}")
        else:
            print(f"  {DIM}  (not present) {name}{RESET}")


# ── YuNet ONNX model ─────────────────────────────────────────────

def download_yunet():
    """Download YuNet ONNX face detector (idempotent)."""
    if os.path.isfile(YUNET_MODEL_PATH):
        size = os.path.getsize(YUNET_MODEL_PATH)
        print(f"  {GREEN}✓{RESET} YuNet face detector (ONNX)  {DIM}({_sizeof_fmt(size)}, cached){RESET}")
        return

    print(f"  {YELLOW}↓{RESET} YuNet face detector (ONNX)  {DIM}downloading…{RESET}")
    os.makedirs(YUNET_MODEL_DIR, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
    size = os.path.getsize(YUNET_MODEL_PATH)
    print(f"  {GREEN}✓{RESET} YuNet face detector (ONNX)  {DIM}({_sizeof_fmt(size)}){RESET}")


def verify_yunet():
    """Check that YuNet ONNX model exists."""
    if os.path.isfile(YUNET_MODEL_PATH):
        size = os.path.getsize(YUNET_MODEL_PATH)
        print(f"  {GREEN}✓{RESET} YuNet face detector (ONNX)  {DIM}({_sizeof_fmt(size)}){RESET}")
        return True
    print(f"  {RED}✗{RESET} YuNet face detector (ONNX)  {DIM}(not downloaded){RESET}")
    return False


def clean_yunet():
    """Remove cached YuNet model."""
    if os.path.isfile(YUNET_MODEL_PATH):
        os.remove(YUNET_MODEL_PATH)
        print(f"  {YELLOW}✗{RESET} Removed YuNet face detector")
    else:
        print(f"  {DIM}  (not present) YuNet face detector{RESET}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained models for the Phase 1c research plugin.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", action="store_true",
                       help="Check that all models are cached (don't download)")
    group.add_argument("--clean", action="store_true",
                       help="Remove cached model files")
    args = parser.parse_args()

    if args.clean:
        print(f"\n{YELLOW}Removing cached research models…{RESET}\n")
        print(f"{YELLOW}Torchvision weights:{RESET}")
        try:
            clean_torchvision_models()
        except ImportError:
            print(f"  {DIM}  (torch not installed, skipping){RESET}")
        print(f"\n{YELLOW}YuNet ONNX:{RESET}")
        clean_yunet()
        print(f"\n{GREEN}✓ Clean complete{RESET}\n")
        return

    if args.verify:
        print(f"\n{GREEN}Verifying research model cache…{RESET}\n")
        ok = True
        print(f"{YELLOW}Torchvision weights:{RESET}")
        try:
            ok = verify_torchvision_models() and ok
        except ImportError:
            print(f"  {RED}✗{RESET} torch/torchvision not installed")
            ok = False
        print(f"\n{YELLOW}YuNet ONNX:{RESET}")
        ok = verify_yunet() and ok
        print()
        if ok:
            print(f"{GREEN}✓ All models cached and ready{RESET}\n")
        else:
            print(f"{RED}✗ Some models missing — run: make research-setup{RESET}\n")
            sys.exit(1)
        return

    # Default: download
    print(f"\n{GREEN}Downloading research plugin models…{RESET}\n")

    print(f"{YELLOW}Torchvision pretrained weights (~395 MB total):{RESET}")
    try:
        download_torchvision_models()
    except ImportError:
        print(f"  {RED}✗{RESET} torch/torchvision not installed")
        print(f"    Install with: make photoholmes-setup  (installs torch)")
        sys.exit(1)

    print(f"\n{YELLOW}YuNet face detector ONNX (~228 KB):{RESET}")
    download_yunet()

    print(f"\n{GREEN}✓ All research models downloaded and cached{RESET}\n")


if __name__ == "__main__":
    main()
