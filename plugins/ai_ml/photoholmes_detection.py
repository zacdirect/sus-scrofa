# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Photoholmes Forgery Detection — library-backed image forensics.

Integrates the photoholmes library (https://github.com/photoholmes/photoholmes)
for state-of-the-art image forgery detection. Uses the library's factory pattern
and preprocessing pipelines directly — we don't reimplement any methods.

Available methods (license-safe for broad distribution):
  DQ             — Double JPEG quantization detection (Apache 2.0, CPU-only)
  Noisesniffer   — Noise inconsistency analysis (GPL-3.0, CPU-only)
  ZERO           — JPEG grid alignment forgery detection (AGPL-3.0, CPU-only)
  FOCAL          — Contrastive learning forgery localization (MIT, GPU, needs weights)
  PSCC-Net       — Progressive spatio-channel correlation (MIT, GPU, needs weights)
  EXIF as Language — CLIP-based EXIF consistency (MIT, GPU, needs weights)
  Adaptive CFA   — Demosaicing artifact analysis (Apache 2.0, GPU, needs weights)

Methods NOT included due to restrictive licenses:
  Splicebuster   — Research-only license
  TruFor          — Research-only license
  CAT-Net         — Research-only license

Design:
  - CPU-only methods (DQ, ZERO, Noisesniffer) run by default
  - GPU methods are opt-in and only load when weights are available
  - Each method produces a BenchmarkOutput with heatmap/mask/detection
  - Results are stored in a consistent format for confidence scoring
  - Heatmaps are stored as base64 PNG for template rendering
"""

import base64
import logging
import os
import tempfile
from io import BytesIO

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import str2temp_file

logger = logging.getLogger(__name__)

# Lazy-check for photoholmes availability
try:
    from photoholmes.methods.factory import MethodFactory
    from photoholmes.methods.registry import MethodRegistry
    from photoholmes.utils.image import read_image, read_jpeg_data
    IS_PHOTOHOLMES = True
except ImportError:
    IS_PHOTOHOLMES = False

try:
    import torch
    import numpy as np
    IS_TORCH = True
except ImportError:
    IS_TORCH = False


# Method configuration: which methods to run by default
# Each entry: (registry_name, display_name, requires_gpu, requires_weights, license)
METHOD_REGISTRY = [
    ("dq",           "DQ (Double Quantization)",      False, False, "Apache-2.0"),
    ("zero",         "ZERO (JPEG Grid Analysis)",     False, False, "AGPL-3.0"),
    ("noisesniffer", "Noisesniffer",                  False, False, "GPL-3.0"),
    # GPU methods — only run if weights are available
    ("focal",            "FOCAL",               True, True, "MIT"),
    ("psccnet",          "PSCC-Net",            True, True, "MIT"),
    ("exif_as_language", "EXIF as Language",    True, True, "MIT"),
    ("adaptive_cfa_net", "Adaptive CFA Net",   True, True, "Apache-2.0"),
]


def _heatmap_to_base64_png(heatmap_tensor):
    """Convert a heatmap tensor/ndarray to a base64-encoded PNG for template display."""
    try:
        import cv2

        if hasattr(heatmap_tensor, 'numpy'):
            heatmap = heatmap_tensor.numpy()
        else:
            heatmap = np.asarray(heatmap_tensor)

        # Normalize to 0-255
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            normalized = ((heatmap - hmin) / (hmax - hmin) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(heatmap, dtype=np.uint8)

        # Apply JET colormap for visualization
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        _, png_bytes = cv2.imencode('.png', colored)
        return base64.b64encode(png_bytes.tobytes()).decode('ascii')
    except Exception as e:
        logger.warning(f"Failed to convert heatmap to PNG: {e}")
        return None


def _mask_to_base64_png(mask_tensor):
    """Convert a binary mask tensor to a base64-encoded PNG."""
    try:
        import cv2

        if hasattr(mask_tensor, 'numpy'):
            mask = mask_tensor.numpy()
        else:
            mask = np.asarray(mask_tensor)

        # Binary mask → 0/255
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        _, png_bytes = cv2.imencode('.png', mask_uint8)
        return base64.b64encode(png_bytes.tobytes()).decode('ascii')
    except Exception as e:
        logger.warning(f"Failed to convert mask to PNG: {e}")
        return None


class PhotoholmesDetector(BaseAnalyzerModule):
    """Image forgery detection using the photoholmes library."""

    order = 25

    # Environment variable to enable GPU methods
    ENABLE_GPU_METHODS = os.environ.get("SUSSCROFA_PHOTOHOLMES_GPU", "0") == "1"

    # Path where model weights are stored (for GPU methods)
    WEIGHTS_DIR = os.environ.get(
        "SUSSCROFA_PHOTOHOLMES_WEIGHTS",
        os.path.expanduser("~/.susscrofa/photoholmes_weights")
    )

    def check_deps(self):
        return IS_PHOTOHOLMES and IS_TORCH

    def _get_enabled_methods(self):
        """Return list of methods that should run based on current environment."""
        enabled = []
        for reg_name, display_name, needs_gpu, needs_weights, license_id in METHOD_REGISTRY:
            # Skip GPU methods unless explicitly enabled
            if needs_gpu and not self.ENABLE_GPU_METHODS:
                continue

            # For weight-dependent methods, check if weights directory exists
            if needs_weights:
                weight_path = os.path.join(self.WEIGHTS_DIR, reg_name)
                if not os.path.isdir(weight_path):
                    logger.debug(f"Skipping {display_name}: weights not found at {weight_path}")
                    continue

            enabled.append((reg_name, display_name, license_id))

        return enabled

    def _save_image_to_temp(self, task):
        """Save the task's image data to a temporary file for photoholmes ingestion."""
        file_data = task.get_file_data
        # Detect JPEG for DCT methods
        is_jpeg = file_data[:2] == b'\xff\xd8'
        suffix = ".jpg" if is_jpeg else ".png"
        tmp = str2temp_file(file_data, suffix=suffix)
        tmp.close()
        return tmp.name, is_jpeg

    def _run_method(self, method_name, image_path, is_jpeg):
        """Run a single photoholmes method and return standardized results."""
        try:
            method, preprocessing = MethodFactory.load(method_name)
        except Exception as e:
            logger.warning(f"Failed to load method {method_name}: {e}")
            return None

        try:
            # Determine what inputs the preprocessing pipeline needs
            needed_inputs = set(preprocessing.inputs)

            kwargs = {}

            # All methods need the image
            if "image" in needed_inputs:
                kwargs["image"] = read_image(image_path)

            # DQ and CAT-Net need DCT coefficients
            if "dct_coefficients" in needed_inputs or "qtables" in needed_inputs:
                dct_coefficients, qtables = read_jpeg_data(
                    image_path,
                    suppress_not_jpeg_warning=True
                )
                if "dct_coefficients" in needed_inputs:
                    kwargs["dct_coefficients"] = dct_coefficients
                if "qtables" in needed_inputs:
                    kwargs["qtables"] = qtables

            # Run preprocessing
            processed = preprocessing(**kwargs)

            # Run method benchmark (standardized output)
            output = method.benchmark(**processed)

            return output

        except Exception as e:
            logger.warning(f"Error running method {method_name}: {e}")
            return None

    def _interpret_output(self, output, method_name):
        """Convert BenchmarkOutput to a structured result dict."""
        result = {
            "method": method_name,
            "heatmap": None,
            "mask": None,
            "detection_score": None,
            "forgery_detected": False,
        }

        # Detection score (0-1 where 1 = forged)
        if output.get("detection") is not None:
            det = output["detection"]
            if hasattr(det, 'item'):
                det = det.item()
            elif hasattr(det, 'numpy'):
                det = float(det.numpy().flatten()[0])
            result["detection_score"] = round(float(det), 4)
            result["forgery_detected"] = result["detection_score"] > 0.5

        # Heatmap (probability map)
        if output.get("heatmap") is not None:
            heatmap = output["heatmap"]
            # Compute summary statistics
            if hasattr(heatmap, 'numpy'):
                h_np = heatmap.numpy()
            else:
                h_np = np.asarray(heatmap)

            result["heatmap_stats"] = {
                "mean": round(float(h_np.mean()), 4),
                "max": round(float(h_np.max()), 4),
                "std": round(float(h_np.std()), 4),
                "suspicious_pixel_ratio": round(
                    float((h_np > 0.5).sum() / max(h_np.size, 1)), 4
                ),
            }
            # Store base64-encoded heatmap for display
            result["heatmap"] = _heatmap_to_base64_png(heatmap)

            # If no explicit detection score, derive one from heatmap
            if result["detection_score"] is None:
                # Use suspicious pixel ratio as a proxy
                spr = result["heatmap_stats"]["suspicious_pixel_ratio"]
                result["detection_score"] = round(min(spr * 2.0, 1.0), 4)
                result["forgery_detected"] = spr > 0.05

        # Binary mask
        if output.get("mask") is not None:
            mask = output["mask"]
            if hasattr(mask, 'numpy'):
                m_np = mask.numpy()
            else:
                m_np = np.asarray(mask)

            result["mask_stats"] = {
                "forged_pixel_ratio": round(
                    float((m_np > 0.5).sum() / max(m_np.size, 1)), 4
                ),
            }
            result["mask"] = _mask_to_base64_png(mask)

            # If still no detection score, derive from mask
            if result["detection_score"] is None:
                fpr = result["mask_stats"]["forged_pixel_ratio"]
                result["detection_score"] = round(min(fpr * 3.0, 1.0), 4)
                result["forgery_detected"] = fpr > 0.01

        return result

    def run(self, task):
        """Run all enabled photoholmes forgery detection methods."""
        enabled = self._get_enabled_methods()
        if not enabled:
            logger.info(f"[Task {task.id}]: No photoholmes methods available")
            self.results["photoholmes"]["enabled"] = False
            return self.results

        self.results["photoholmes"]["enabled"] = True
        self.results["photoholmes"]["methods"] = {}

        # Save image to temp file (photoholmes works with file paths)
        try:
            image_path, is_jpeg = self._save_image_to_temp(task)
        except Exception as e:
            logger.error(f"[Task {task.id}]: Failed to save temp file for photoholmes: {e}")
            self.results["photoholmes"]["error"] = str(e)
            return self.results

        methods_run = 0
        forgery_detected_count = 0
        detection_scores = []
        all_results = {}

        try:
            for reg_name, display_name, license_id in enabled:
                logger.info(f"[Task {task.id}]: Running photoholmes method: {display_name}")

                output = self._run_method(reg_name, image_path, is_jpeg)
                if output is None:
                    all_results[reg_name] = {
                        "method": display_name,
                        "error": "Method failed to produce output",
                        "license": license_id,
                    }
                    continue

                result = self._interpret_output(output, display_name)
                result["license"] = license_id
                all_results[reg_name] = result

                methods_run += 1
                if result.get("detection_score") is not None:
                    detection_scores.append(result["detection_score"])
                if result.get("forgery_detected"):
                    forgery_detected_count += 1

        finally:
            # Clean up temp file
            try:
                os.unlink(image_path)
            except OSError:
                pass

        # Store per-method results
        self.results["photoholmes"]["methods"] = all_results

        # Aggregate summary
        if detection_scores:
            avg_score = sum(detection_scores) / len(detection_scores)
            max_score = max(detection_scores)
        else:
            avg_score = 0.0
            max_score = 0.0

        self.results["photoholmes"]["summary"] = {
            "methods_run": methods_run,
            "methods_available": len(enabled),
            "forgery_detected_count": forgery_detected_count,
            "avg_detection_score": round(avg_score, 4),
            "max_detection_score": round(max_score, 4),
            "consensus_forgery": forgery_detected_count > (methods_run / 2) if methods_run > 0 else False,
        }

        if methods_run > 0:
            logger.info(
                f"[Task {task.id}]: Photoholmes complete — "
                f"{methods_run} methods, "
                f"{forgery_detected_count} detected forgery, "
                f"avg score={avg_score:.3f}, max={max_score:.3f}"
            )
        else:
            logger.warning(f"[Task {task.id}]: No photoholmes methods succeeded")

        return self.results
