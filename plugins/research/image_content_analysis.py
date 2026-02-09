# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Image Content Analysis — object detection, person identification, and
photorealism classification for forensic research.

This is a **research tier** plugin (Phase 1c). It runs after all static
and AI/ML plugins and is never skipped. Most of its output is research
data — only the photorealism/animated classification reports to the
auditor, and at LOW level only.

Pipeline:
    1. Photorealism Classification
       Uses texture statistics (LBP entropy, high-frequency energy,
       color distribution kurtosis) plus edge coherence to distinguish
       photorealistic images from drawings/illustrations/cartoons.
       → Reports to auditor: LOW positive (photorealistic)
                              LOW negative (non-photorealistic)

    2. Object Detection (COCO, 91 classes)
       Uses torchvision FasterRCNN-ResNet50-FPN-v2 pretrained on COCO.
       Detects all objects at ≥0.5 confidence.
       → Research data only: object inventory, bounding boxes, counts.

    3. Person Analysis
       When persons are detected, crops each person region and runs
       targeted analysis for visible attributes:
         - Approximate hair color (dominant hue in head region)
         - Visible tattoos (high-frequency ink-pattern detection)
         - Visible piercings (small bright-spot detection near face)
       → Research data only: per-person attribute annotations.

Configuration:
    ENABLE_RESEARCH_CONTENT_ANALYSIS = True/False in local_settings.py

Dependencies:
    torch, torchvision (already required for photoholmes)
    opencv-python (already required for ELA/noise/frequency)
    numpy, scipy (already required)
"""

import base64
import logging
import os
import tempfile
from io import BytesIO

import cv2
import numpy as np

from lib.analyzer.base import BaseAnalyzerModule
from lib.analyzer.plugin_contract import create_finding, validate_audit_findings
from lib.utils import str2temp_file

logger = logging.getLogger(__name__)

# ── Dependency checks ─────────────────────────────────────────────

try:
    import torch
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        keypointrcnn_resnet50_fpn,
        KeypointRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.transforms import functional as TF
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy import stats as sp_stats
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Lazy-loaded singleton models ──────────────────────────────────

_detector_model = None
_detector_weights = None
_keypoint_model = None
_keypoint_weights = None
_yunet_detector = None

# YuNet ONNX model cache path
_YUNET_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "opencv_models")
_YUNET_MODEL_PATH = os.path.join(_YUNET_MODEL_DIR, "face_detection_yunet_2023mar.onnx")
_YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _get_detector():
    """Load FasterRCNN once, reuse across calls."""
    global _detector_model, _detector_weights
    if _detector_model is None:
        logger.info("Loading FasterRCNN-ResNet50-FPN-v2 (COCO)…")
        _detector_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        _detector_model = fasterrcnn_resnet50_fpn_v2(
            weights=_detector_weights, box_score_thresh=0.5
        )
        _detector_model.eval()
        logger.info("FasterRCNN loaded (CPU)")
    return _detector_model, _detector_weights


def _get_keypoint_model():
    """Load KeypointRCNN once, reuse across calls.

    Returns both person bounding boxes AND 17 COCO keypoints per person:
        nose, left_eye, right_eye, left_ear, right_ear,
        left_shoulder, right_shoulder, left_elbow, right_elbow,
        left_wrist, right_wrist, left_hip, right_hip,
        left_knee, right_knee, left_ankle, right_ankle
    """
    global _keypoint_model, _keypoint_weights
    if _keypoint_model is None:
        logger.info("Loading KeypointRCNN-ResNet50-FPN (COCO)…")
        _keypoint_weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        _keypoint_model = keypointrcnn_resnet50_fpn(
            weights=_keypoint_weights, box_score_thresh=0.5
        )
        _keypoint_model.eval()
        logger.info("KeypointRCNN loaded (CPU)")
    return _keypoint_model, _keypoint_weights


def _get_yunet_detector():
    """Load YuNet face detector (ONNX) once, reuse across calls.

    YuNet provides precise face bounding boxes and 5 facial landmarks:
        right_eye, left_eye, nose_tip, right_mouth_corner, left_mouth_corner

    Auto-downloads the model on first use if not cached.
    Returns None if model unavailable.
    """
    global _yunet_detector
    if _yunet_detector is not None:
        return _yunet_detector

    # Download if not cached
    if not os.path.isfile(_YUNET_MODEL_PATH):
        logger.info("Downloading YuNet face detector ONNX model…")
        os.makedirs(_YUNET_MODEL_DIR, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(_YUNET_MODEL_URL, _YUNET_MODEL_PATH)
            logger.info("YuNet model saved to %s", _YUNET_MODEL_PATH)
        except Exception as e:
            logger.warning("Failed to download YuNet model: %s", e)
            return None

    try:
        # Create with dummy size — we resize per-image in _detect_faces_yunet()
        _yunet_detector = cv2.FaceDetectorYN.create(
            _YUNET_MODEL_PATH, "", (320, 320), 0.7, 0.3, 5000
        )
        logger.info("YuNet face detector loaded (ONNX)")
        return _yunet_detector
    except Exception as e:
        logger.warning("Failed to load YuNet model: %s", e)
        return None


# ── Photorealism classifier (no ML model — pure signal analysis) ──

def _classify_photorealism(image_bgr):
    """
    Classify whether an image looks photorealistic or animated/drawn.

    Uses four orthogonal signals:
      1. LBP (Local Binary Pattern) entropy — photos have high micro-texture
         entropy; drawings have smoother, more uniform textures.
      2. High-frequency energy — photos have rich high-freq detail from
         camera sensor noise and real-world textures; drawings/illustrations
         have less.
      3. Color distribution kurtosis — photos have mesokurtic or platykurtic
         color distributions; cell-shaded art has high kurtosis (flat colors
         with sharp transitions).
      4. Edge coherence — drawings have cleaner, longer edge segments;
         photos have fragmented, noisy edges.

    Returns:
        dict with keys:
            is_photorealistic: bool
            confidence: float (0.0–1.0)
            signals: dict of individual signal values
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Downsample for speed on large images (work at ~1024px max dim)
    scale = min(1.0, 1024.0 / max(h, w))
    if scale < 1.0:
        gray_s = cv2.resize(gray, None, fx=scale, fy=scale)
        bgr_s = cv2.resize(image_bgr, None, fx=scale, fy=scale)
    else:
        gray_s = gray
        bgr_s = image_bgr

    # ── Signal 1: LBP entropy ─────────────────────────────────────
    # Simplified LBP via comparison of each pixel to its 8 neighbors
    lbp_score = _lbp_entropy(gray_s)

    # ── Signal 2: High-frequency energy ───────────────────────────
    hf_energy = _high_freq_energy(gray_s)

    # ── Signal 3: Color kurtosis ──────────────────────────────────
    color_kurt = _color_kurtosis(bgr_s)

    # ── Signal 4: Edge coherence ──────────────────────────────────
    edge_coherence = _edge_coherence(gray_s)

    # ── Combine signals ──────────────────────────────────────────
    # Photos: high LBP entropy, high HF energy, low kurtosis, low edge coherence
    # Drawings: low LBP entropy, low HF energy, high kurtosis, high edge coherence
    #
    # Score each signal 0-1 where 1 = "looks like a photo"
    photo_lbp = min(lbp_score / 7.0, 1.0)          # Photos >6.5, drawings <5.5
    photo_hf = min(hf_energy / 0.15, 1.0)           # Photos >0.12, drawings <0.05
    photo_kurt = max(1.0 - color_kurt / 15.0, 0.0)  # Photos <5, drawings >10
    photo_edge = max(1.0 - edge_coherence / 0.6, 0.0)  # Photos <0.3, drawings >0.5

    # Weighted combination (LBP and HF are strongest discriminators)
    composite = (
        0.35 * photo_lbp +
        0.30 * photo_hf +
        0.20 * photo_kurt +
        0.15 * photo_edge
    )

    is_photo = composite >= 0.50
    # Confidence is how far from the boundary
    confidence = min(abs(composite - 0.50) * 2.0, 1.0)

    return {
        "is_photorealistic": is_photo,
        "confidence": round(confidence, 4),
        "composite_score": round(composite, 4),
        "signals": {
            "lbp_entropy": round(lbp_score, 4),
            "high_freq_energy": round(hf_energy, 6),
            "color_kurtosis": round(color_kurt, 4),
            "edge_coherence": round(edge_coherence, 4),
            "photo_lbp": round(photo_lbp, 4),
            "photo_hf": round(photo_hf, 4),
            "photo_kurt": round(photo_kurt, 4),
            "photo_edge": round(photo_edge, 4),
        },
    }


def _lbp_entropy(gray):
    """Compute Local Binary Pattern entropy (simplified 3×3 LBP)."""
    padded = np.pad(gray.astype(np.int16), 1, mode='edge')
    center = padded[1:-1, 1:-1]

    # 8 neighbors, build binary code
    code = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy:padded.shape[0] - 1 + dy,
                          1 + dx:padded.shape[1] - 1 + dx]
        code |= ((neighbor >= center).astype(np.uint8) << bit)

    # Shannon entropy of LBP histogram
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float64)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _high_freq_energy(gray):
    """Ratio of high-frequency energy (DCT-based)."""
    # Use Laplacian as a proxy for high-frequency content
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    hf_energy = np.mean(lap ** 2)
    total_energy = np.mean(gray.astype(np.float64) ** 2) + 1e-10
    return float(hf_energy / total_energy)


def _color_kurtosis(bgr):
    """Mean kurtosis across color channels (flat colors → high kurtosis)."""
    if not HAS_SCIPY:
        # Fallback: crude kurtosis from moments
        result = 0.0
        for c in range(3):
            ch = bgr[:, :, c].ravel().astype(np.float64)
            m = ch.mean()
            s = ch.std() + 1e-10
            result += float(np.mean(((ch - m) / s) ** 4) - 3.0)
        return abs(result / 3.0)

    kurts = []
    for c in range(3):
        k = sp_stats.kurtosis(bgr[:, :, c].ravel(), fisher=True)
        kurts.append(abs(k))
    return float(np.mean(kurts))


def _edge_coherence(gray):
    """
    Edge coherence: ratio of long connected edge runs to total edge pixels.
    Drawings have clean long strokes; photos have fragmented noisy edges.
    """
    edges = cv2.Canny(gray, 50, 150)
    total_edge_pixels = np.count_nonzero(edges)
    if total_edge_pixels < 100:
        return 0.5  # Ambiguous

    # Dilate to connect nearby fragments, then count connected components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    n_components, labels, comp_stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    if n_components <= 1:
        return 0.5

    # Area of largest connected edge component vs total
    # (skip component 0 = background)
    areas = comp_stats[1:, cv2.CC_STAT_AREA]
    largest = areas.max()
    coherence = largest / total_edge_pixels

    return float(min(coherence, 1.0))


# ── Person attribute analysis (Keypoint-guided) ──────────────────

# Body region definitions: pairs of keypoints that define limb segments.
# Each region gets a rectangular strip between the two keypoints,
# expanded laterally by a fraction of the segment length.
BODY_REGIONS = {
    "left_forearm":    ("left_elbow",    "left_wrist"),
    "right_forearm":   ("right_elbow",   "right_wrist"),
    "left_upper_arm":  ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder","right_elbow"),
    "left_thigh":      ("left_hip",      "left_knee"),
    "right_thigh":     ("right_hip",     "right_knee"),
    "left_calf":       ("left_knee",     "left_ankle"),
    "right_calf":      ("right_knee",    "right_ankle"),
}

# Piercing zone definitions.
# v3: Uses YuNet 5-point landmarks (precise) + body keypoints for ears.
# YuNet landmarks: right_eye, left_eye, nose_tip, right_mouth_corner, left_mouth_corner
# Body keypoints:  left_ear, right_ear (YuNet doesn't detect ears)
#
# Zone radius factors are relative to inter-eye distance (much more stable
# than the old _face_radius estimate from body keypoints).
PIERCING_ZONES = {
    "nose":           {"anchor": "nose_tip",           "radius_factor": 0.30},
    "left_ear":       {"anchor": "left_ear",           "radius_factor": 0.35},
    "right_ear":      {"anchor": "right_ear",          "radius_factor": 0.35},
    "lip":            {"anchor": "_lip_center",         "radius_factor": 0.28},
    "left_eyebrow":   {"anchor": "_left_brow",          "radius_factor": 0.18},
    "right_eyebrow":  {"anchor": "_right_brow",         "radius_factor": 0.18},
}

# Minimum keypoint score to consider a keypoint "visible"
KP_SCORE_THRESHOLD = 2.0


def _extract_keypoints(kps_array, kps_scores, kp_names, rescale=1.0):
    """Convert raw keypoint arrays into a named dict of visible keypoints.

    Returns:
        dict mapping keypoint name → (x, y) in original image coords,
        only for keypoints with score above threshold.
    """
    visible = {}
    for j, name in enumerate(kp_names):
        x, y, vis_flag = kps_array[j]
        score = float(kps_scores[j])
        if vis_flag > 0 and score > KP_SCORE_THRESHOLD:
            visible[name] = (float(x) * rescale, float(y) * rescale)
    return visible


def _detect_faces_yunet(image_bgr):
    """Run YuNet face detector on an image.

    Returns:
        list of dicts, each with:
            bbox: (x, y, w, h) face bounding box
            landmarks: dict with keys:
                right_eye, left_eye, nose_tip,
                right_mouth_corner, left_mouth_corner
            score: detection confidence
    Returns empty list if YuNet is unavailable.
    """
    detector = _get_yunet_detector()
    if detector is None:
        return []

    h, w = image_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, raw = detector.detect(image_bgr)

    if raw is None:
        return []

    faces = []
    for row in raw:
        # row: x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
        #      x_rmc, y_rmc, x_lmc, y_lmc, score
        faces.append({
            "bbox": (float(row[0]), float(row[1]),
                     float(row[2]), float(row[3])),
            "landmarks": {
                "right_eye":           (float(row[4]),  float(row[5])),
                "left_eye":            (float(row[6]),  float(row[7])),
                "nose_tip":            (float(row[8]),  float(row[9])),
                "right_mouth_corner":  (float(row[10]), float(row[11])),
                "left_mouth_corner":   (float(row[12]), float(row[13])),
            },
            "score": float(row[14]),
        })
    return faces


def _match_face_to_person(face, person_box, body_kps):
    """Check if a YuNet face detection belongs inside a person bounding box.

    Also merges body keypoints (ears) into the face landmark set.

    Returns:
        merged landmark dict (YuNet + body ears) or None if no match
    """
    fx, fy, fw, fh = face["bbox"]
    face_cx = fx + fw / 2
    face_cy = fy + fh / 2
    px1, py1, px2, py2 = person_box

    # Face center must be inside person bbox (with small margin)
    margin_x = (px2 - px1) * 0.05
    margin_y = (py2 - py1) * 0.05
    if not (px1 - margin_x <= face_cx <= px2 + margin_x and
            py1 - margin_y <= face_cy <= py2 + margin_y):
        return None

    # Start with YuNet landmarks (precise)
    merged = dict(face["landmarks"])

    # Add body-keypoint ears (YuNet doesn't detect ears)
    if "left_ear" in body_kps:
        merged["left_ear"] = body_kps["left_ear"]
    if "right_ear" in body_kps:
        merged["right_ear"] = body_kps["right_ear"]

    # Compute inter-eye distance for zone sizing
    re = np.array(merged["right_eye"])
    le = np.array(merged["left_eye"])
    merged["_inter_eye_dist"] = float(np.linalg.norm(le - re))

    return merged


def _derive_face_points_yunet(landmarks):
    """Derive additional face anchor points from YuNet + body keypoint landmarks.

    Adds synthetic points for piercing zone anchors:
      _lip_center  — midpoint between mouth corners, shifted slightly down
      _left_brow   — above left eye
      _right_brow  — above right eye

    The inter-eye distance (already in landmarks as _inter_eye_dist) is
    used as the universal scale reference.
    """
    derived = dict(landmarks)
    ied = landmarks.get("_inter_eye_dist", 30.0)

    # Eye midpoint and face vertical axis
    re = np.array(landmarks["right_eye"])
    le = np.array(landmarks["left_eye"])
    eye_mid = (re + le) / 2
    nose = np.array(landmarks["nose_tip"])

    # Downward direction: eye_mid → nose
    down = nose - eye_mid
    down_len = np.linalg.norm(down)
    if down_len > 1:
        down_unit = down / down_len
    else:
        down_unit = np.array([0.0, 1.0])

    # Lip center: midpoint of mouth corners, nudged slightly downward
    if "right_mouth_corner" in landmarks and "left_mouth_corner" in landmarks:
        rmc = np.array(landmarks["right_mouth_corner"])
        lmc = np.array(landmarks["left_mouth_corner"])
        mouth_mid = (rmc + lmc) / 2
        # Shift down by ~15% of inter-eye distance (below lip line → below-lip piercings)
        derived["_lip_center"] = tuple(mouth_mid + down_unit * ied * 0.15)
    elif down_len > 1:
        # Fallback: estimate from nose
        derived["_lip_center"] = tuple(nose + down_unit * ied * 0.8)

    # Eyebrow points: above each eye, perpendicular to face vertical
    brow_offset = down_unit * ied * 0.35  # 35% of inter-eye above eye
    derived["_left_brow"] = tuple(np.array(landmarks["left_eye"]) - brow_offset)
    derived["_right_brow"] = tuple(np.array(landmarks["right_eye"]) - brow_offset)

    return derived


def _get_limb_strip(image_bgr, pt_a, pt_b, width_factor=0.35):
    """Extract a rectangular strip along a limb segment.

    Creates an oriented bounding box between two keypoints,
    expanded laterally by width_factor × segment_length.

    Returns:
        (cropped_bgr, mask) or (None, None) if too small
    """
    a = np.array(pt_a, dtype=np.float32)
    b = np.array(pt_b, dtype=np.float32)
    length = np.linalg.norm(b - a)
    if length < 15:
        return None, None

    # Direction along limb and perpendicular
    direction = (b - a) / length
    perp = np.array([-direction[1], direction[0]])

    half_w = length * width_factor

    # Four corners of the oriented rectangle
    corners = np.array([
        a - perp * half_w,
        a + perp * half_w,
        b + perp * half_w,
        b - perp * half_w,
    ], dtype=np.float32)

    # Axis-aligned bounding box
    x_min = max(int(corners[:, 0].min()), 0)
    y_min = max(int(corners[:, 1].min()), 0)
    x_max = min(int(corners[:, 0].max()) + 1, image_bgr.shape[1])
    y_max = min(int(corners[:, 1].max()) + 1, image_bgr.shape[0])

    if x_max - x_min < 10 or y_max - y_min < 10:
        return None, None

    crop = image_bgr[y_min:y_max, x_min:x_max].copy()

    # Build a mask for just the oriented strip within the axis-aligned crop
    shifted_corners = corners - np.array([x_min, y_min])
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, shifted_corners.astype(np.int32), 255)

    return crop, mask


# Minimum person bounding-box height (in pixels) for face-level analysis.
# People shorter than this in the image are too distant for reliable
# piercing or hair-color detection. Tattoos on limbs can still be
# analysed at smaller sizes because they occupy larger pixel areas.
MIN_PERSON_HEIGHT_FOR_FACE = 120   # ~120px tall — roughly waist-up at 1080p


def _analyze_person_keypoints(image_bgr, person_box, keypoints,
                               img_h, img_w, yunet_faces=None):
    """
    Analyze a detected person using keypoint-guided body region extraction.

    Piercing detection requires a YuNet face match — if YuNet doesn't
    detect the face (too far, occluded, profile view) piercings are
    simply skipped rather than guessed from imprecise body keypoints.

    Persons whose bounding box is shorter than MIN_PERSON_HEIGHT_FOR_FACE
    skip face-level analysis entirely (piercings + hair) to avoid noise
    from distant/small figures.

    Args:
        image_bgr: Full image (BGR)
        person_box: (x1, y1, x2, y2) bounding box
        keypoints: dict of visible keypoint name → (x, y)
        img_h, img_w: image dimensions
        yunet_faces: list of YuNet face detections (from _detect_faces_yunet)

    Returns:
        dict of person attributes with location-specific findings
    """
    x1, y1, x2, y2 = [int(v) for v in person_box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_w), min(y2, img_h)

    person_w = x2 - x1
    person_h = y2 - y1

    if person_w < 20 or person_h < 30:
        return {"too_small": True}

    attributes = {
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                 "width": person_w, "height": person_h},
        "visible_keypoints": list(keypoints.keys()),
        "keypoint_count": len(keypoints),
    }

    # Gate: is this person large enough for face-level analysis?
    face_analysis_ok = person_h >= MIN_PERSON_HEIGHT_FOR_FACE

    # ── Hair color (keypoint-guided head region) ──────────────────
    if face_analysis_ok:
        attributes["hair_color"] = _estimate_hair_color_kp(
            image_bgr, keypoints, (x1, y1, x2, y2))
    else:
        attributes["hair_color"] = {
            "color": "unknown", "confidence": 0.0,
            "reason": "person_too_distant",
        }

    # ── Tattoos (per-limb analysis) ───────────────────────────────
    # Tattoos occupy larger pixel areas so we keep a lower threshold
    attributes["tattoo_analysis"] = _detect_tattoos_by_region(
        image_bgr, keypoints)

    # ── Piercings (YuNet-only — no fallback) ──────────────────────
    if face_analysis_ok and yunet_faces:
        matched_face_kps = None
        for face in yunet_faces:
            merged = _match_face_to_person(face, person_box, keypoints)
            if merged is not None:
                matched_face_kps = _derive_face_points_yunet(merged)
                break

        if matched_face_kps is not None:
            attributes["piercing_analysis"] = _detect_piercings_by_zone(
                image_bgr, matched_face_kps)
        else:
            attributes["piercing_analysis"] = {
                "detected": False, "piercings": [], "zone_count": 0,
                "zones": {}, "note": "no_yunet_face_match",
            }
    else:
        reason = "person_too_distant" if not face_analysis_ok else "no_yunet_faces"
        attributes["piercing_analysis"] = {
            "detected": False, "piercings": [], "zone_count": 0,
            "zones": {}, "note": reason,
        }

    return attributes


def _estimate_hair_color_kp(image_bgr, keypoints, person_box):
    """
    Estimate hair color using keypoints to locate the hair region.

    Strategy: the hair region is above the eyes and ears. We take the
    bounding box top to the topmost eye/ear keypoint, and crop that
    strip. This avoids including forehead/face skin.

    Returns:
        dict with color category, confidence, and HSV stats
    """
    x1, y1, x2, y2 = person_box

    # Find the topmost face feature to define "above face" = hair
    face_features = ["left_eye", "right_eye", "left_ear", "right_ear", "nose"]
    face_ys = [keypoints[k][1] for k in face_features if k in keypoints]

    if not face_ys:
        # Fallback: top 15% of person box
        hair_bottom = y1 + int((y2 - y1) * 0.15)
        hair_top = y1
    else:
        # Hair is above the topmost face feature, with some margin
        topmost_y = min(face_ys)
        eye_nose_dist = 0
        if "nose" in keypoints and ("left_eye" in keypoints or "right_eye" in keypoints):
            eye_y = min(keypoints.get("left_eye", (0, 9999))[1],
                        keypoints.get("right_eye", (0, 9999))[1])
            eye_nose_dist = abs(keypoints["nose"][1] - eye_y)

        # Hair starts from above the topmost feature
        margin = max(eye_nose_dist * 0.3, 5)
        hair_bottom = int(topmost_y - margin)
        hair_top = y1

    # Also constrain horizontally to the ear-to-ear span if available
    face_xs = [keypoints[k][0] for k in ["left_ear", "right_ear",
                                           "left_eye", "right_eye"]
               if k in keypoints]
    if len(face_xs) >= 2:
        face_left = max(int(min(face_xs) - 15), x1)
        face_right = min(int(max(face_xs) + 15), x2)
    else:
        face_left, face_right = x1, x2

    hair_top = max(hair_top, 0)
    hair_bottom = max(hair_bottom, hair_top + 5)

    if hair_bottom - hair_top < 8 or face_right - face_left < 10:
        return {"color": "unknown", "confidence": 0.0, "reason": "insufficient_hair_region"}

    hair_crop = image_bgr[hair_top:hair_bottom, face_left:face_right]
    if hair_crop.size < 100:
        return {"color": "unknown", "confidence": 0.0, "reason": "crop_too_small"}

    return _classify_hair_color(hair_crop)


def _classify_hair_color(hair_bgr):
    """
    Classify hair color from a cropped hair region.

    Uses HSV analysis with skin-pixel exclusion and dominant-color
    clustering for more reliable classification.
    """
    hsv = cv2.cvtColor(hair_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Exclude likely skin pixels and very bright background
    skin_mask = (
        (h_ch >= 5) & (h_ch <= 25) &
        (s_ch >= 40) & (s_ch <= 170) &
        (v_ch >= 80) & (v_ch <= 230)
    )
    bg_mask = (v_ch > 240) | (v_ch < 15)
    exclude = skin_mask | bg_mask
    hair_mask = ~exclude

    if hair_mask.sum() < 50:
        return {"color": "unknown", "confidence": 0.0, "reason": "no_hair_pixels"}

    h_vals = h_ch[hair_mask].astype(float)
    s_vals = s_ch[hair_mask].astype(float)
    v_vals = v_ch[hair_mask].astype(float)

    med_h = float(np.median(h_vals))
    med_s = float(np.median(s_vals))
    med_v = float(np.median(v_vals))

    # Classification rules (refined)
    if med_v < 45:
        color = "black"
        confidence = 0.75 if med_v < 30 else 0.55
    elif med_s < 25 and med_v > 160:
        color = "gray" if med_v < 210 else "white"
        confidence = 0.60
    elif med_s > 130 and (med_h < 8 or med_h > 172):
        color = "red"
        confidence = 0.55
    elif med_s > 160 and 20 < med_h < 170:
        # Unusually saturated non-red — dyed colors
        if 90 < med_h < 140:
            color = "blue_dyed"
        elif 35 < med_h < 90:
            color = "green_dyed"
        elif 140 < med_h < 170:
            color = "purple_dyed"
        else:
            color = "dyed_other"
        confidence = 0.45
    elif med_v > 170 and med_s < 70:
        color = "blonde"
        confidence = 0.55
    elif 8 <= med_h <= 28 and 25 <= med_s <= 140:
        color = "brown" if med_v < 140 else "light_brown"
        confidence = 0.50
    else:
        color = "indeterminate"
        confidence = 0.25

    return {
        "color": color,
        "confidence": round(confidence, 3),
        "hsv_median": {"h": round(med_h, 1), "s": round(med_s, 1),
                        "v": round(med_v, 1)},
        "hair_pixels": int(hair_mask.sum()),
    }


def _detect_tattoos_by_region(image_bgr, keypoints):
    """
    Detect tattoo indicators on each visible body region.

    Uses keypoint pairs to extract oriented strips along each limb,
    then analyzes each for ink-like patterns on skin.

    Returns:
        dict with per-region results and summary
    """
    regions = {}
    detected_regions = []

    for region_name, (kp_a, kp_b) in BODY_REGIONS.items():
        if kp_a not in keypoints or kp_b not in keypoints:
            continue

        crop, mask = _get_limb_strip(image_bgr,
                                      keypoints[kp_a], keypoints[kp_b],
                                      width_factor=0.40)
        if crop is None:
            regions[region_name] = {"visible": False}
            continue

        result = _analyze_skin_for_tattoo(crop, mask)
        result["visible"] = True
        regions[region_name] = result

        if result["detected"]:
            detected_regions.append(region_name)

    # Also check torso if we have shoulder + hip keypoints
    torso_result = _check_torso_tattoo(image_bgr, keypoints)
    if torso_result is not None:
        regions["torso"] = torso_result
        if torso_result["detected"]:
            detected_regions.append("torso")

    return {
        "detected": len(detected_regions) > 0,
        "detected_regions": detected_regions,
        "region_count": len(regions),
        "regions": regions,
    }


def _check_torso_tattoo(image_bgr, keypoints):
    """Check the torso area (shoulders to hips) for tattoo indicators."""
    needed = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if not all(k in keypoints for k in needed):
        return None

    ls = keypoints["left_shoulder"]
    rs = keypoints["right_shoulder"]
    lh = keypoints["left_hip"]
    rh = keypoints["right_hip"]

    # Torso bounding box with small margin
    xs = [ls[0], rs[0], lh[0], rh[0]]
    ys = [ls[1], rs[1], lh[1], rh[1]]
    margin_x = (max(xs) - min(xs)) * 0.1
    margin_y = (max(ys) - min(ys)) * 0.05

    x1 = max(int(min(xs) - margin_x), 0)
    y1 = max(int(min(ys) - margin_y), 0)
    x2 = min(int(max(xs) + margin_x), image_bgr.shape[1])
    y2 = min(int(max(ys) + margin_y), image_bgr.shape[0])

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    crop = image_bgr[y1:y2, x1:x2]
    mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255
    result = _analyze_skin_for_tattoo(crop, mask)
    result["visible"] = True
    return result


def _analyze_skin_for_tattoo(crop_bgr, region_mask):
    """
    Analyze a body region crop for tattoo-like patterns on skin.

    Improved approach:
      1. Detect skin-tone pixels within the region mask
      2. Within skin areas, find high-contrast dark patterns (ink)
      3. Require contiguous blobs (not just scattered dark pixels)
      4. Check pattern complexity (tattoos have edges, not just shadows)

    Returns:
        dict with detected flag, confidence, and metrics
    """
    if crop_bgr.size < 500:
        return {"detected": False, "confidence": 0.0, "reason": "too_small"}

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # Broad skin-tone mask (accommodates diverse skin tones)
    skin_mask = (
        (hsv[:, :, 0] <= 28) &
        (hsv[:, :, 1] >= 20) & (hsv[:, :, 1] <= 200) &
        (hsv[:, :, 2] >= 50)
    ).astype(np.uint8) * 255

    # Intersect with region mask
    skin_mask = cv2.bitwise_and(skin_mask, region_mask)

    skin_pixels = cv2.countNonZero(skin_mask)
    if skin_pixels < 100:
        return {"detected": False, "confidence": 0.0,
                "reason": "insufficient_skin", "skin_pixels": skin_pixels}

    # Skin brightness stats
    skin_gray = gray[skin_mask > 0]
    skin_mean = float(skin_gray.mean())
    skin_std = float(skin_gray.std())

    # Tattoo ink: dark regions on skin that are significantly darker
    # than the surrounding skin (adaptive threshold)
    dark_threshold = max(skin_mean - 2.0 * skin_std, skin_mean * 0.55, 30)
    dark_on_skin = ((gray < dark_threshold) & (skin_mask > 0)).astype(np.uint8) * 255

    # Morphological cleanup — remove single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_on_skin = cv2.morphologyEx(dark_on_skin, cv2.MORPH_OPEN, kernel)

    # Find contiguous dark blobs
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_on_skin, connectivity=8)

    # Filter blobs: tattoos form coherent regions, not tiny specks
    min_blob_area = max(skin_pixels * 0.005, 20)  # at least 0.5% of skin
    max_blob_area = skin_pixels * 0.60  # not more than 60% (that's a shadow)

    tattoo_blobs = 0
    tattoo_pixel_count = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_blob_area < area < max_blob_area:
            # Check blob complexity: tattoos have edges/detail
            blob_mask = (labels == i).astype(np.uint8) * 255
            edges = cv2.Canny(
                cv2.bitwise_and(gray, gray, mask=blob_mask), 30, 100)
            edge_ratio = cv2.countNonZero(edges) / max(area, 1)
            # Tattoos have moderate edge density; uniform shadows don't
            if edge_ratio > 0.03:
                tattoo_blobs += 1
                tattoo_pixel_count += area

    dark_ratio = tattoo_pixel_count / max(skin_pixels, 1)

    # Detection threshold: meaningful ink coverage with coherent blobs
    detected = tattoo_blobs >= 1 and 0.01 < dark_ratio < 0.40
    if detected:
        if tattoo_blobs >= 3 and dark_ratio > 0.05:
            confidence = 0.70
        elif tattoo_blobs >= 2 or dark_ratio > 0.08:
            confidence = 0.55
        else:
            confidence = 0.35
    else:
        confidence = 0.0

    return {
        "detected": detected,
        "confidence": round(confidence, 3),
        "ink_coverage": round(dark_ratio, 4),
        "blob_count": tattoo_blobs,
        "skin_pixels": skin_pixels,
    }


def _detect_piercings_by_zone(image_bgr, face_kps):
    """
    Detect piercing indicators at specific face locations.

    Requires YuNet-derived face landmarks (precise 5-point facial
    geometry).  Uses inter-eye distance as the universal scale
    reference for zone radii.

    Args:
        image_bgr: Full image (BGR)
        face_kps: dict of landmark name → (x, y), including derived
                  points from _derive_face_points_yunet()

    Returns:
        dict with per-zone results and summary list
    """
    img_h, img_w = image_bgr.shape[:2]
    ied = face_kps.get("_inter_eye_dist", 30.0)

    # Minimum inter-eye distance for reliable analysis
    if ied < 15:
        return {
            "detected": False, "piercings": [], "zone_count": 0,
            "zones": {}, "note": "face_too_small",
        }

    zones = {}
    detected_piercings = []

    for zone_name, zone_def in PIERCING_ZONES.items():
        anchor_name = zone_def["anchor"]
        if anchor_name not in face_kps:
            continue

        center = face_kps[anchor_name]
        radius = int(ied * zone_def["radius_factor"])
        radius = max(radius, 6)  # minimum 6px

        # Extract circular zone
        cx, cy = int(center[0]), int(center[1])
        x1 = max(cx - radius, 0)
        y1 = max(cy - radius, 0)
        x2 = min(cx + radius, img_w)
        y2 = min(cy + radius, img_h)

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        zone_crop = image_bgr[y1:y2, x1:x2]
        mask = np.zeros(zone_crop.shape[:2], dtype=np.uint8)
        local_cx = cx - x1
        local_cy = cy - y1
        cv2.circle(mask, (local_cx, local_cy), radius, 255, -1)

        result = _check_zone_for_piercing(zone_crop, mask, zone_name, ied)
        zones[zone_name] = result

        if result["detected"]:
            detected_piercings.append(f"pierced_{zone_name}")

    return {
        "detected": len(detected_piercings) > 0,
        "piercings": detected_piercings,
        "zone_count": len(zones),
        "zones": zones,
        "inter_eye_dist": round(ied, 1),
    }


def _check_zone_for_piercing(zone_bgr, zone_mask, zone_name, inter_eye_dist):
    """
    Check a face zone for piercing indicators.

    Requires YuNet-derived landmarks for reliable zone placement.
    Strict detection to minimise false positives:
      - Size constraints relative to inter-eye distance
      - Bright metallic spots must have specular characteristics
      - Dark spots must have sharp boundaries (not gradual shadows)
      - Zone-specific thresholds (ears more permissive)

    Piercing indicators:
      1. Metallic highlights: small, very bright, low-saturation spots
      2. Dark studs/rings: small, very dark spots with sharp boundaries
      3. Colored gems: small spots with unusually high saturation

    Args:
        zone_bgr: cropped zone image
        zone_mask: circular mask within the crop
        zone_name: name of the zone (affects thresholds)
        inter_eye_dist: for size calibration
    """
    if zone_bgr.size < 100:
        return {"detected": False, "confidence": 0.0}

    gray = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2HSV)

    zone_pixels = cv2.countNonZero(zone_mask)
    if zone_pixels < 20:
        return {"detected": False, "confidence": 0.0}

    # Skin baseline in this zone
    skin_vals = gray[zone_mask > 0]
    skin_mean = float(skin_vals.mean())
    skin_std = float(skin_vals.std())

    # If the zone has very high variance, it's likely not a clean face area
    # (could be hair, background, clothing edge) — be skeptical
    high_variance = skin_std > 50

    # Piercing size constraints based on inter-eye distance
    # A typical piercing stud is ~2-4mm; at common photo distances
    # that's roughly 2-8% of inter-eye distance in pixels
    min_piercing_area = max(int(inter_eye_dist * 0.5), 3)
    max_piercing_area = max(int(inter_eye_dist * inter_eye_dist * 0.04), 50)

    candidates = 0
    evidence_types = []

    # Strategy 1: Specular metallic highlights
    # Metal piercings create small, bright, low-saturation spots
    bright_thresh = min(skin_mean + 3.0 * max(skin_std, 12), 245)
    bright_mask = (
        (gray > bright_thresh) &
        (hsv[:, :, 1] < 80) &  # low saturation (metallic, not skin flush)
        (zone_mask > 0)
    ).astype(np.uint8) * 255
    n_bright = _count_piercing_blobs_strict(
        bright_mask, min_piercing_area, max_piercing_area)
    if n_bright > 0:
        candidates += n_bright
        evidence_types.append("metallic_bright")

    # Strategy 2: Dark studs/rings with sharp edges
    # Must be distinctly darker than skin, not just shadows
    dark_thresh = max(skin_mean - 3.5 * max(skin_std, 12), 15)
    dark_mask = (
        (gray < dark_thresh) &
        (zone_mask > 0)
    ).astype(np.uint8) * 255
    # Extra check: dark blobs must have sharp boundaries (not gradual shadow)
    n_dark = _count_piercing_blobs_strict(
        dark_mask, min_piercing_area, max_piercing_area,
        require_sharp_edge=True, gray_img=gray)
    if n_dark > 0:
        candidates += n_dark
        evidence_types.append("dark_stud")

    # Strategy 3: Colored gems — high saturation, unusual hue for skin
    # Exclude skin tones (H 0-28) and common lip colors
    gem_mask = (
        (hsv[:, :, 1] > 160) &
        ((hsv[:, :, 0] > 28) | (hsv[:, :, 0] < 3)) &  # not skin hue
        (zone_mask > 0)
    ).astype(np.uint8) * 255
    n_gem = _count_piercing_blobs_strict(
        gem_mask, min_piercing_area, max_piercing_area)
    if n_gem > 0:
        candidates += n_gem
        evidence_types.append("colored_gem")

    # Detection logic: require multiple evidence signals for non-ear zones
    # Ears are more permissive (earrings are common and visually distinct)
    is_ear = "ear" in zone_name
    if is_ear:
        min_candidates = 1
    else:
        min_candidates = 2  # nose/lip/brow need stronger evidence

    # Penalize high-variance zones (likely not clean face skin)
    if high_variance and not is_ear:
        min_candidates += 1

    detected = candidates >= min_candidates

    if detected:
        n_types = len(evidence_types)
        if n_types >= 2 and candidates >= 2:
            confidence = 0.75
        elif is_ear and candidates >= 1:
            confidence = 0.60
        elif candidates >= 2:
            confidence = 0.50
        else:
            confidence = 0.35
    else:
        confidence = 0.0

    return {
        "detected": detected,
        "candidate_count": candidates,
        "evidence_types": evidence_types,
        "confidence": round(confidence, 3),
    }


def _count_piercing_blobs_strict(binary_mask, min_area, max_area,
                                  require_sharp_edge=False, gray_img=None):
    """Count small blob features consistent with piercings (strict version).

    v3: Stricter than v2 — explicit size bounds from inter-eye distance,
    optional sharp-edge requirement for dark blobs.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        # Piercings are roughly circular or elongated (hoops)
        if circularity < 0.25:
            continue

        # Optional: sharp edge check (for dark blobs — distinguish
        # studs from gradual shadows)
        if require_sharp_edge and gray_img is not None:
            blob_mask = np.zeros_like(binary_mask)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            # Dilate to get border pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            border = cv2.dilate(blob_mask, kernel) - blob_mask
            if cv2.countNonZero(border) > 5:
                inner = float(gray_img[blob_mask > 0].mean())
                outer = float(gray_img[border > 0].mean())
                # Sharp boundary: significant brightness jump
                if abs(outer - inner) < 25:
                    continue  # gradual transition = shadow, not stud

        count += 1
    return count


# ══════════════════════════════════════════════════════════════════
# Plugin class
# ══════════════════════════════════════════════════════════════════

class ImageContentAnalysis(BaseAnalyzerModule):
    """
    Image content analysis for forensic research.

    Phase 1c research plugin — object detection, person identification,
    photorealism classification.

    Only the photorealism classification reports to the auditor (LOW level).
    Everything else is research data.
    """

    order = 10  # First research plugin to run

    def check_deps(self):
        """Requires torch + torchvision + opencv + scipy."""
        if not HAS_TORCH:
            logger.warning("ImageContentAnalysis: torch/torchvision not available")
            return False
        if not HAS_SCIPY:
            logger.warning("ImageContentAnalysis: scipy not available")
            return False
        return True

    def run(self, task):
        """Run image content analysis pipeline."""
        # Check settings
        from django.conf import settings
        if not getattr(settings, 'ENABLE_RESEARCH_CONTENT_ANALYSIS', True):
            logger.info("Image content analysis disabled via settings")
            self.results["content_analysis"] = {
                "enabled": False,
                "reason": "Disabled in settings (ENABLE_RESEARCH_CONTENT_ANALYSIS=False)",
            }
            return self.results

        try:
            raw_bytes = task.get_file_data
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image_bgr is None:
                logger.warning("[Task %s]: Could not decode image for content analysis", task.id)
                self.results["content_analysis"]["enabled"] = True
                self.results["content_analysis"]["error"] = "Could not decode image"
                return self.results

            self.results["content_analysis"]["enabled"] = True
            img_h, img_w = image_bgr.shape[:2]
            self.results["content_analysis"]["image_size"] = {
                "width": img_w, "height": img_h,
            }

            # ── Step 1: Photorealism classification ───────────────
            logger.info("[Task %s]: Running photorealism classification…", task.id)
            photo_result = _classify_photorealism(image_bgr)
            self.results["content_analysis"]["photorealism"] = photo_result

            # ── Step 2: Object detection (FasterRCNN for all objects) ─
            logger.info("[Task %s]: Running object detection (FasterRCNN COCO)…", task.id)
            detections = self._run_object_detection(image_bgr)
            self.results["content_analysis"]["objects"] = detections

            # ── Step 3: Person keypoint detection ─────────────────
            logger.info("[Task %s]: Running person keypoint detection…", task.id)
            person_keypoints = self._run_keypoint_detection(image_bgr)

            # ── Step 3b: YuNet face detection ─────────────────────
            logger.info("[Task %s]: Running YuNet face detection…", task.id)
            yunet_faces = _detect_faces_yunet(image_bgr)
            self.results["content_analysis"]["yunet_faces"] = len(yunet_faces)

            # ── Step 4: Person analysis (keypoint-guided + YuNet) ─
            if person_keypoints:
                logger.info("[Task %s]: Analyzing %d detected person(s) with keypoints…",
                            task.id, len(person_keypoints))
                person_analyses = []
                for i, pkp in enumerate(person_keypoints):
                    attrs = _analyze_person_keypoints(
                        image_bgr, pkp["box"], pkp["keypoints"],
                        img_h, img_w, yunet_faces=yunet_faces)
                    attrs["detection_confidence"] = pkp["score"]
                    person_analyses.append(attrs)
                self.results["content_analysis"]["persons"] = person_analyses
            else:
                self.results["content_analysis"]["persons"] = []

            # ── Step 5: Debug annotations ─────────────────────────
            self._save_debug_annotations(
                image_bgr, person_keypoints, yunet_faces,
                self.results["content_analysis"]["persons"],
                task)

            # ── Step 6: Build audit findings ──────────────────────
            n_persons = len(person_keypoints)
            self.results["content_analysis"]["audit_findings"] = (
                self._create_audit_findings(photo_result,
                                             self.results["content_analysis"]["persons"])
            )

            logger.info(
                "[Task %s]: Content analysis complete — "
                "photorealistic=%s (%.2f), %d objects, %d persons, "
                "%d YuNet faces",
                task.id,
                photo_result["is_photorealistic"],
                photo_result["confidence"],
                detections.get("total_detections", 0),
                n_persons,
                len(yunet_faces),
            )

        except Exception as e:
            logger.exception("[Task %s]: Content analysis error: %s", task.id, e)
            self.results["content_analysis"]["enabled"] = True
            self.results["content_analysis"]["error"] = str(e)

        return self.results

    def _run_object_detection(self, image_bgr):
        """Run FasterRCNN on the image and return structured detections."""
        try:
            model, weights = _get_detector()
            categories = weights.meta["categories"]

            # Convert BGR→RGB, to tensor, normalize
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Downsample very large images for speed
            h, w = image_rgb.shape[:2]
            max_dim = 1600
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image_rgb = cv2.resize(image_rgb, None, fx=scale, fy=scale)
                rescale = 1.0 / scale
            else:
                rescale = 1.0

            tensor = TF.to_tensor(image_rgb).unsqueeze(0)

            with torch.no_grad():
                predictions = model(tensor)[0]

            labels = predictions["labels"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()
            boxes = predictions["boxes"].cpu().numpy()

            detections = []
            class_counts = {}

            for label_id, score, box in zip(labels, scores, boxes):
                label_name = categories[label_id]
                # Rescale boxes back to original image coords
                x1, y1, x2, y2 = box * rescale

                det = {
                    "label": label_name,
                    "confidence": round(float(score), 4),
                    "bbox": {
                        "x1": round(float(x1), 1),
                        "y1": round(float(y1), 1),
                        "x2": round(float(x2), 1),
                        "y2": round(float(y2), 1),
                    },
                }
                detections.append(det)
                class_counts[label_name] = class_counts.get(label_name, 0) + 1

            return {
                "total_detections": len(detections),
                "class_counts": class_counts,
                "detections": detections,
            }

        except Exception as e:
            logger.warning("Object detection failed: %s", e)
            return {
                "total_detections": 0,
                "class_counts": {},
                "detections": [],
                "error": str(e),
            }

    def _run_keypoint_detection(self, image_bgr):
        """Run KeypointRCNN to get person boxes + 17 COCO body keypoints.

        Returns:
            list of dicts, each with:
                score: detection confidence
                box: (x1, y1, x2, y2) in original image coords
                keypoints: dict of visible keypoint name → (x, y)
                keypoint_scores: dict of keypoint name → score
        """
        try:
            model, weights = _get_keypoint_model()
            kp_names = weights.meta["keypoint_names"]

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            h, w = image_rgb.shape[:2]
            max_dim = 1200
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image_rgb = cv2.resize(image_rgb, None, fx=scale, fy=scale)
                rescale = 1.0 / scale
            else:
                rescale = 1.0

            tensor = TF.to_tensor(image_rgb).unsqueeze(0)
            with torch.no_grad():
                preds = model(tensor)[0]

            persons = []
            for i in range(len(preds["boxes"])):
                score = float(preds["scores"][i].item())
                if score < 0.5:
                    continue

                box = preds["boxes"][i].cpu().numpy() * rescale
                kps = preds["keypoints"][i].cpu().numpy()
                kps_scores = preds["keypoints_scores"][i].cpu().numpy()

                visible_kps = _extract_keypoints(
                    kps, kps_scores, kp_names, rescale=rescale)

                persons.append({
                    "score": round(score, 4),
                    "box": tuple(box.tolist()),
                    "keypoints": visible_kps,
                    "keypoint_scores": {
                        name: round(float(kps_scores[j]), 2)
                        for j, name in enumerate(kp_names)
                    },
                })

            return persons

        except Exception as e:
            logger.warning("Keypoint detection failed: %s", e)
            return []

    def _save_debug_annotations(self, image_bgr, person_keypoints,
                                yunet_faces, person_analyses, task):
        """Generate and save annotated debug images for researcher review.

        Draws on a copy of the image:
          - Person bounding boxes (green)
          - Body keypoints (cyan dots)
          - Limb region strips (yellow outlines)
          - YuNet face boxes and landmarks (magenta)
          - Piercing zone circles (red=detected, gray=not detected)
          - Tattoo region highlights (orange where detected)
          - Hair region box (blue)

        Stored in MongoDB GridFS alongside the analysis results.
        The GridFS ID is recorded in the results dict for later retrieval.
        """
        from django.conf import settings
        if not getattr(settings, 'RESEARCH_SAVE_ANNOTATIONS', False):
            return

        try:
            from lib.db import save_file, is_mongo_available
            if not is_mongo_available():
                logger.warning("[Task %s]: Annotations enabled but MongoDB unavailable",
                               task.id)
                return
        except ImportError:
            logger.warning("[Task %s]: lib.db not importable, skipping annotations",
                           task.id)
            return

        try:
            annotated = image_bgr.copy()

            # Color palette (BGR)
            CLR_PERSON_BOX = (0, 255, 0)
            CLR_KEYPOINT = (255, 255, 0)
            CLR_LIMB_STRIP = (0, 255, 255)
            CLR_YUNET_BOX = (255, 0, 255)
            CLR_YUNET_LM = (255, 0, 255)
            CLR_PIERCE_DETECTED = (0, 0, 255)
            CLR_PIERCE_CLEAR = (160, 160, 160)
            CLR_TATTOO = (0, 140, 255)
            CLR_HAIR = (255, 100, 0)

            # ── Draw YuNet face boxes and landmarks ───────────────
            for face in yunet_faces:
                fx, fy, fw, fh = [int(v) for v in face["bbox"]]
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh),
                              CLR_YUNET_BOX, 2)
                cv2.putText(annotated, f"YuNet {face['score']:.2f}",
                            (fx, fy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, CLR_YUNET_BOX, 1)
                for lm_name, (lx, ly) in face["landmarks"].items():
                    cv2.circle(annotated, (int(lx), int(ly)), 3,
                               CLR_YUNET_LM, -1)

            # ── Draw per-person annotations ───────────────────────
            for p_idx, pkp in enumerate(person_keypoints):
                box = pkp["box"]
                bx1, by1, bx2, by2 = [int(v) for v in box]
                kps = pkp["keypoints"]
                pa = person_analyses[p_idx] if p_idx < len(person_analyses) else {}

                # Person bounding box
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2),
                              CLR_PERSON_BOX, 2)
                cv2.putText(annotated, f"P{p_idx+1} {pkp['score']:.2f}",
                            (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, CLR_PERSON_BOX, 1)

                # Body keypoints
                for kp_name, (kx, ky) in kps.items():
                    cv2.circle(annotated, (int(kx), int(ky)), 4,
                               CLR_KEYPOINT, -1)

                # Limb strips
                for region_name, (kp_a, kp_b) in BODY_REGIONS.items():
                    if kp_a in kps and kp_b in kps:
                        self._draw_limb_strip(annotated, kps[kp_a], kps[kp_b],
                                               CLR_LIMB_STRIP)

                # Tattoo highlights
                tat = pa.get("tattoo_analysis", {})
                for reg_name in tat.get("detected_regions", []):
                    if reg_name == "torso":
                        self._draw_torso_highlight(annotated, kps, CLR_TATTOO)
                    elif reg_name in BODY_REGIONS:
                        kp_a, kp_b = BODY_REGIONS[reg_name]
                        if kp_a in kps and kp_b in kps:
                            self._draw_limb_strip(annotated, kps[kp_a],
                                                   kps[kp_b], CLR_TATTOO,
                                                   thickness=3)

                # Piercing zones (only drawn when YuNet matched)
                pier = pa.get("piercing_analysis", {})
                if pier.get("note") not in ("no_yunet_face_match",
                                            "person_too_distant",
                                            "no_yunet_faces"):
                    detected_piercings = set(pier.get("piercings", []))
                    ied = pier.get("inter_eye_dist", 30.0)

                    # Re-derive face points for drawing
                    matched_lm = None
                    if yunet_faces:
                        for face in yunet_faces:
                            merged = _match_face_to_person(face, box, kps)
                            if merged is not None:
                                matched_lm = _derive_face_points_yunet(merged)
                                break

                    if matched_lm is not None:
                        for zone_name, zone_def in PIERCING_ZONES.items():
                            anchor = zone_def["anchor"]
                            if anchor not in matched_lm:
                                continue
                            cx = int(matched_lm[anchor][0])
                            cy = int(matched_lm[anchor][1])
                            radius = max(int(ied * zone_def["radius_factor"]), 6)
                            is_det = f"pierced_{zone_name}" in detected_piercings
                            color = CLR_PIERCE_DETECTED if is_det else CLR_PIERCE_CLEAR
                            thickness = 2 if is_det else 1
                            cv2.circle(annotated, (cx, cy), radius, color, thickness)
                            label = f"*{zone_name}*" if is_det else zone_name
                            cv2.putText(annotated, label,
                                        (cx + radius + 2, cy + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                # Hair region
                hair = pa.get("hair_color", {})
                if hair.get("color") and hair["color"] not in ("unknown",):
                    face_features = ["left_eye", "right_eye", "left_ear",
                                     "right_ear", "nose"]
                    face_ys = [kps[k][1] for k in face_features if k in kps]
                    face_xs = [kps[k][0] for k in ["left_ear", "right_ear",
                                                    "left_eye", "right_eye"]
                               if k in kps]
                    if face_ys:
                        hair_bottom = int(min(face_ys)) - 5
                        hair_top = by1
                        hx1 = int(min(face_xs) - 15) if len(face_xs) >= 2 else bx1
                        hx2 = int(max(face_xs) + 15) if len(face_xs) >= 2 else bx2
                        cv2.rectangle(annotated, (hx1, hair_top),
                                      (hx2, hair_bottom), CLR_HAIR, 1)
                        cv2.putText(annotated, f"hair:{hair['color']}",
                                    (hx1, hair_top - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    CLR_HAIR, 1)

            # ── Encode and save to GridFS ─────────────────────────
            success, png_buf = cv2.imencode(".png", annotated)
            if not success:
                logger.warning("Failed to encode annotation image")
                return

            file_id = save_file(data=png_buf.tobytes(),
                                content_type="image/png")
            self.results["content_analysis"]["annotation_gridfs_id"] = file_id
            logger.info("[Task %s]: Annotation saved to GridFS: %s",
                        task.id, file_id)

        except Exception as e:
            logger.warning("[Task %s]: Annotation generation failed: %s",
                           task.id, e)

    @staticmethod
    def _draw_limb_strip(image, pt_a, pt_b, color, thickness=1,
                          width_factor=0.40):
        """Draw an oriented rectangle strip between two keypoints."""
        a = np.array(pt_a, dtype=np.float32)
        b = np.array(pt_b, dtype=np.float32)
        length = np.linalg.norm(b - a)
        if length < 15:
            return

        direction = (b - a) / length
        perp = np.array([-direction[1], direction[0]])
        half_w = length * width_factor

        corners = np.array([
            a - perp * half_w,
            a + perp * half_w,
            b + perp * half_w,
            b - perp * half_w,
        ], dtype=np.int32)

        cv2.polylines(image, [corners], True, color, thickness)

    @staticmethod
    def _draw_torso_highlight(image, kps, color):
        """Draw torso region outline between shoulders and hips."""
        needed = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not all(k in kps for k in needed):
            return
        pts = np.array([
            [int(kps["left_shoulder"][0]), int(kps["left_shoulder"][1])],
            [int(kps["right_shoulder"][0]), int(kps["right_shoulder"][1])],
            [int(kps["right_hip"][0]), int(kps["right_hip"][1])],
            [int(kps["left_hip"][0]), int(kps["left_hip"][1])],
        ], dtype=np.int32)
        cv2.polylines(image, [pts], True, color, 3)

    def _create_audit_findings(self, photo_result, persons):
        """
        Create audit findings from content analysis.

        Only photorealism classification reports to the auditor — and
        at LOW level only. Everything else is research data.
        """
        findings = []

        is_photo = photo_result["is_photorealistic"]
        confidence = photo_result["confidence"]
        composite = photo_result["composite_score"]

        if is_photo:
            findings.append(create_finding(
                level="LOW",
                category="Image Content Type",
                description=(
                    f"Image appears photorealistic "
                    f"(composite score: {composite:.2f}, "
                    f"confidence: {confidence:.0%})"
                ),
                is_positive=True,
                confidence=confidence,
            ))
        else:
            findings.append(create_finding(
                level="LOW",
                category="Image Content Type",
                description=(
                    f"Image appears non-photorealistic "
                    f"(animated/drawn/illustrated) "
                    f"(composite score: {composite:.2f}, "
                    f"confidence: {confidence:.0%})"
                ),
                is_positive=False,
                confidence=confidence,
            ))

        # Validate before returning
        is_valid, errors = validate_audit_findings(findings, "content_analysis")
        if not is_valid:
            logger.warning("Content analysis produced invalid findings: %s", errors)
            return []

        return findings
