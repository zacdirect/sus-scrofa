# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Compliance Auditor — Engine-Level Post-Processing.

This is NOT a plugin.  It runs after all plugins complete and reads
from the accumulated results dict.  It never opens the image file
itself — every fact it needs was already extracted by plugins.

Architecture:
    _process_image() in processing.py:
        Phase 1: Run all plugins (order-independent for scoring)
        Phase 2: Engine post-processing:
            a) auditor.audit(results)   → results['audit']
            b) confidence scoring       → results['confidence']
            c) save to MongoDB

Scoring model:
    Detectors report findings on a 6-point scale:
        {LOW, MEDIUM, HIGH} × {positive=True, positive=False}

    Detectors do NOT assign point values.  They are domain experts
    that classify what they found; the auditor is the sole authority
    on how much each finding is worth.

    Point table (auditor-owned):
        LOW     →   5 pts
        MEDIUM  →  15 pts
        HIGH    →  50 pts

    Positive findings add, negative findings subtract, from a
    base of 50 (uncertain).  Clamped to 0–100.

    HIGH negative should be reserved for virtual certainty:
        ✓  AI generator keyword in filename/tags
        ✓  ML model says AI with >80% confidence
        ✗  Missing EXIF (too common in normal images)
        ✗  Square dimensions (profile photos do this)

    The auditor may also create "big picture" findings that combine
    signals from multiple detectors (convergent evidence).
"""

import logging
from typing import Dict, List, Optional

from lib.analyzer.plugin_contract import get_audit_findings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Finding — the atomic unit of audit evidence
# ─────────────────────────────────────────────────────────────────

class Finding:
    """A single audit finding on the 6-point scale.

    Detectors create findings; only the auditor assigns point values.

    Attributes:
        level:       LOW, MEDIUM, or HIGH
        category:    Short tag for grouping (e.g. "AI Indicator")
        description: Human-readable explanation
        is_positive: True = evidence of authenticity,
                     False = evidence of inauthenticity
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    def __init__(self, level: str, category: str, description: str,
                 is_positive: bool = False):
        self.level = level
        self.category = category
        self.description = description
        self.is_positive = is_positive


# ─────────────────────────────────────────────────────────────────
# Storage contract — keys each plugin MUST write
# ─────────────────────────────────────────────────────────────────
#
# The auditor reads these top-level keys from the results dict.
# Plugins that don't write their key are simply skipped — the
# auditor never fails because a plugin was unavailable.
#
#   file_name         (str)   — info plugin (order 10)
#   metadata          (dict)  — metadata plugin (order 10)
#     .dimensions     (list)  — [width, height]
#     .Exif           (dict)  — grouped EXIF tags
#     .gps            (dict)  — GPS position data (if present)
#   noise_analysis    (dict)  — noise plugin (order 25)
#     .inconsistency_score, .anomaly_count, .suspicious
#   frequency_analysis(dict)  — frequency plugin (order 26)
#     .checkerboard_score, .peak_ratio, .anomaly_score
#   opencv_manipulation(dict) — opencv plugin (order 40)
#     .enabled, .manipulation_detection, .noise_analysis, .jpeg_artifacts
#   ai_detection      (dict)  — ai_detection plugin (order 30)
#     .enabled, .detection_layers[]


# ─────────────────────────────────────────────────────────────────
# Auditor point table — the ONLY place points are assigned
# ─────────────────────────────────────────────────────────────────

POINTS = {
    Finding.LOW:    5,
    Finding.MEDIUM: 15,
    Finding.HIGH:   50,
}


# ─────────────────────────────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────────────────────────────

LEGITIMATE_CAMERAS = {
    'Google': ['Pixel'],
    'Apple': ['iPhone', 'iPad'],
    'Canon': True,
    'Nikon': True,
    'Sony': True,
    'Samsung': True,
    'Fujifilm': True,
    'Olympus': True,
    'Panasonic': True,
    'Leica': True,
    'Hasselblad': True,
    'Phase One': True,
    'DJI': ['Mavic', 'Phantom', 'Inspire'],
}

OBSOLETE_SOFTWARE = {
    'Picasa': {'discontinued': 2016},
    'Windows Photo Gallery': {'discontinued': 2012},
    'iPhoto': {'discontinued': 2015},
    'Adobe Photoshop Album': {'discontinued': 2006},
}

AI_INDICATORS = [
    'midjourney', 'dall-e', 'dalle', 'stable_diffusion', 'stable-diffusion', 'sd-',
    'gemini_generated', 'chatgpt', 'gpt-', 'ai-generated', 'ai_generated',
    'generated_image', 'synthetic', 'deepdream', 'artbreeder',
    'nightcafe', 'craiyon', 'lexica', 'playground-ai', 'firefly',
]

AI_TYPICAL_DIMENSIONS = [
    (512, 512), (768, 768), (1024, 1024),
    (1536, 1536), (2048, 2048),
]

CAMERA_RESOLUTIONS = {
    '4:3': [(640, 480), (800, 600), (1024, 768), (1600, 1200),
            (2048, 1536), (4000, 3000)],
    '3:2': [(1500, 1000), (3000, 2000), (4288, 2848), (6000, 4000)],
    '16:9': [(1280, 720), (1920, 1080), (2560, 1440),
             (3840, 2160), (4096, 2160)],
    '1:1': [(1080, 1080)],
}


# ─────────────────────────────────────────────────────────────────
# Public API — called by _process_image()
# ─────────────────────────────────────────────────────────────────

def audit(results: dict) -> dict:
    """Run the compliance audit on accumulated plugin results.

    Returns a dict stored at ``results['audit']``.
    """
    findings: List[Finding] = []

    # Extract commonly-used data
    file_name = results.get('file_name', '')
    metadata = results.get('metadata', {})
    exif_data = metadata.get('Exif', {})
    dimensions = metadata.get('dimensions', [0, 0])
    width = dimensions[0] if len(dimensions) >= 2 else 0
    height = dimensions[1] if len(dimensions) >= 2 else 0

    # === INDIVIDUAL DETECTOR CHECKS ===
    findings.extend(_check_ai_indicators(file_name, exif_data))
    findings.extend(_check_obsolete_software(exif_data))
    findings.extend(_check_ai_dimensions(width, height))
    findings.extend(_check_suspicious_exif(exif_data))
    findings.extend(_check_missing_metadata(exif_data))
    findings.extend(_check_minimal_exif(exif_data))
    findings.extend(_check_legitimate_camera(exif_data))
    findings.extend(_check_gps_data(metadata))
    findings.extend(_check_camera_settings(exif_data))
    findings.extend(_check_photo_resolution(width, height))

    ai_det = results.get('ai_detection', {})
    if ai_det.get('enabled'):
        findings.extend(_check_ml_model_results(
            ai_det.get('detection_layers', [])))

    findings.extend(_check_noise_consistency(
        results.get('noise_analysis', {})))
    findings.extend(_check_frequency_analysis(
        results.get('frequency_analysis', {})))
    findings.extend(_check_opencv_findings(
        results.get('opencv_manipulation', {})))

    # === PLUGIN CONTRACT (standardized audit_findings) ===
    # Check all plugins for audit_findings following the data contract
    findings.extend(_collect_plugin_findings(results))

    # === BIG PICTURE (cross-detector convergence) ===
    findings.extend(_check_convergent_evidence(findings, results))

    # === CALCULATE SCORES ===
    authenticity_score = _calculate_authenticity_score(findings)
    detected_types = _collect_detector_types(findings)
    evidence = _format_evidence(findings, authenticity_score)
    ai_probability = _calculate_ai_probability(findings)
    manipulation_probability = _calculate_manipulation_probability(findings)

    neg = [f for f in findings if not f.is_positive]
    pos = [f for f in findings if f.is_positive]

    return {
        'authenticity_score': authenticity_score,
        'ai_probability': ai_probability,
        'manipulation_probability': manipulation_probability,
        'findings_count': len(findings),
        'evidence': evidence,
        'detected_types': detected_types,
        'findings_summary': {
            'high_neg': sum(1 for f in neg if f.level == Finding.HIGH),
            'medium_neg': sum(1 for f in neg if f.level == Finding.MEDIUM),
            'low_neg': sum(1 for f in neg if f.level == Finding.LOW),
            'high_pos': sum(1 for f in pos if f.level == Finding.HIGH),
            'medium_pos': sum(1 for f in pos if f.level == Finding.MEDIUM),
            'low_pos': sum(1 for f in pos if f.level == Finding.LOW),
        },
    }


# ─────────────────────────────────────────────────────────────────
# Individual checks — each returns a list of Finding objects
#
# Detectors declare (level, is_positive) only.
# HIGH negative = virtual certainty of inauthenticity.
# ─────────────────────────────────────────────────────────────────

def _check_ai_indicators(file_name: str, exif_data: dict) -> List[Finding]:
    """Check 1: AI generator keywords in filename or EXIF software tag."""
    targets = [file_name.lower()]
    software = exif_data.get('Image', {}).get('Software', '')
    if software:
        targets.append(software.lower())

    for indicator in AI_INDICATORS:
        if any(indicator in t for t in targets):
            # Virtual certainty — this IS an AI keyword.
            return [Finding(
                Finding.HIGH, "AI Indicator",
                f"AI generator keyword detected: '{indicator}'",
            )]
    return []


def _check_obsolete_software(exif_data: dict) -> List[Finding]:
    """Check 2: Obsolete or suspicious editing software."""
    software = exif_data.get('Image', {}).get('Software', '')
    if not software:
        return []

    for obs_name, info in OBSOLETE_SOFTWARE.items():
        if obs_name.lower() in software.lower():
            return [Finding(
                Finding.MEDIUM, "Obsolete Software",
                f"Obsolete software: '{software}' "
                f"(discontinued {info.get('discontinued', '?')})",
            )]
    return []


def _check_ai_dimensions(width: int, height: int) -> List[Finding]:
    """Check 3: AI-typical power-of-2 square dimensions.

    LOW because square dimensions are common for profile photos,
    social media crops, and thumbnails.
    """
    if width == height and (width, height) in AI_TYPICAL_DIMENSIONS:
        return [Finding(
            Finding.LOW, "AI Dimensions",
            f"Square power-of-2 dimensions: {width}×{height}",
        )]
    return []


def _check_suspicious_exif(exif_data: dict) -> List[Finding]:
    """Check 4: Suspicious EXIF patterns (e.g. all-zero unique IDs)."""
    for group in exif_data.values():
        if isinstance(group, dict):
            unique_id = group.get('ImageUniqueID') or \
                        group.get('image_unique_id')
            if unique_id and str(unique_id).endswith('00000000000000'):
                return [Finding(
                    Finding.MEDIUM, "Suspicious EXIF",
                    f"Suspicious ImageUniqueId: {unique_id}",
                )]
    return []


def _check_missing_metadata(exif_data: dict) -> List[Finding]:
    """Check 5: Completely missing EXIF metadata.

    LOW because metadata stripping is routine for privacy, web
    optimization, and social-media upload pipelines.
    """
    if not exif_data:
        return [Finding(
            Finding.LOW, "Missing Metadata",
            "No EXIF metadata (commonly stripped for privacy/web use)",
        )]
    return []


def _check_minimal_exif(exif_data: dict) -> List[Finding]:
    """Check 6: Suspiciously few EXIF tags."""
    if not exif_data:
        return []

    tag_count = sum(
        len(group) for group in exif_data.values()
        if isinstance(group, dict)
    )
    if tag_count < 10:
        return [Finding(
            Finding.LOW, "Minimal EXIF",
            f"Only {tag_count} EXIF tags (real cameras typically write 50+)",
        )]
    return []


def _check_legitimate_camera(exif_data: dict) -> List[Finding]:
    """Check 7: Verified camera manufacturer + model."""
    image_info = exif_data.get('Image', {})
    make = image_info.get('Make', '')
    model = image_info.get('Model', '')
    if not make or not model:
        return []

    make_clean = make.strip()
    model_clean = model.strip()

    for manufacturer, models in LEGITIMATE_CAMERAS.items():
        if manufacturer.lower() in make_clean.lower():
            if models is True:
                return [Finding(
                    Finding.MEDIUM, "Legitimate Camera",
                    f"Verified camera: {make_clean} {model_clean}",
                    is_positive=True,
                )]
            for valid_model in models:
                if valid_model.lower() in model_clean.lower():
                    return [Finding(
                        Finding.MEDIUM, "Legitimate Camera",
                        f"Verified camera: {make_clean} {model_clean}",
                        is_positive=True,
                    )]
    return []


def _check_gps_data(metadata: dict) -> List[Finding]:
    """Check 8: GPS location data present."""
    if metadata.get('gps'):
        return [Finding(
            Finding.MEDIUM, "GPS Data",
            "GPS location data present (rare in AI-generated images)",
            is_positive=True,
        )]
    return []


def _check_camera_settings(exif_data: dict) -> List[Finding]:
    """Check 9: Realistic camera settings."""
    photo_info = exif_data.get('Photo', {})
    if not photo_info:
        return []

    count = sum(1 for k in ('ISOSpeedRatings', 'ExposureTime',
                             'FNumber', 'FocalLength')
                if k in photo_info)

    if count >= 3:
        return [Finding(
            Finding.MEDIUM, "Camera Settings",
            f"Realistic camera settings present ({count}/4 key params)",
            is_positive=True,
        )]
    return []


def _check_photo_resolution(width: int, height: int) -> List[Finding]:
    """Check 10: Standard camera resolution and natural aspect ratio."""
    findings = []

    for aspect_name, resolutions in CAMERA_RESOLUTIONS.items():
        if (width, height) in resolutions or (height, width) in resolutions:
            findings.append(Finding(
                Finding.LOW, "Standard Resolution",
                f"Standard camera resolution: {width}×{height} ({aspect_name})",
                is_positive=True,
            ))
            break

    if width > 0 and height > 0:
        ratio = width / height
        if (1.2 < ratio < 2.0 or 0.5 < ratio < 0.83) and \
                abs(ratio - 1.0) > 0.15:
            findings.append(Finding(
                Finding.LOW, "Natural Aspect Ratio",
                f"Aspect ratio {ratio:.2f}:1 typical for photographs",
                is_positive=True,
            ))

    return findings


def _check_ml_model_results(detection_layers: list) -> List[Finding]:
    """Check 11: ML model detector results.

    HIGH is appropriate here because a trained ML model with >80%
    confidence is as close to certainty as static analysis gets.
    """
    findings = []

    for layer in detection_layers:
        if layer.get('method') != 'ml_model':
            continue

        verdict = layer.get('verdict')
        score = layer.get('score', 0.0)
        confidence = layer.get('confidence', 'NONE')
        evidence = layer.get('evidence', 'ml_model')

        if verdict == 'AI' and score is not None and score > 0.8:
            findings.append(Finding(
                Finding.HIGH, "ML Model Detection",
                f"ML model: {score:.0%} AI probability ({evidence})",
            ))
        elif verdict == 'AI' and score is not None and score > 0.5:
            findings.append(Finding(
                Finding.MEDIUM, "ML Model Detection",
                f"ML model: {score:.0%} AI probability ({evidence})",
            ))
        elif verdict == 'Real' and score is not None and score < 0.3 and \
                confidence in ('HIGH', 'MEDIUM', 'CERTAIN'):
            findings.append(Finding(
                Finding.MEDIUM, "ML Model Assessment",
                f"ML model: {1 - score:.0%} authentic ({evidence})",
                is_positive=True,
            ))

    return findings


def _check_noise_consistency(noise_data: dict) -> List[Finding]:
    """Check 12: Noise analysis — synthetic vs natural sensor patterns.

    Very low inconsistency (< 2.0) is strongly synthetic → MEDIUM.
    Moderately low (2.0–3.5) is suspicious → LOW.
    HIGH is not used here because heavily-compressed JPEGs from
    real cameras can also show uniform noise.
    """
    findings = []
    if not noise_data:
        return findings

    inconsistency = noise_data.get('inconsistency_score')
    anomaly_count = noise_data.get('anomaly_count')

    if inconsistency is not None:
        if inconsistency < 2.0:
            findings.append(Finding(
                Finding.MEDIUM, "Synthetic Noise",
                f"Very uniform noise (inconsistency: {inconsistency:.2f})",
            ))
        elif inconsistency < 3.5:
            findings.append(Finding(
                Finding.LOW, "Suspicious Noise",
                f"Low noise inconsistency ({inconsistency:.2f})",
            ))
        elif inconsistency > 5.5:
            findings.append(Finding(
                Finding.MEDIUM, "Natural Noise",
                f"Natural noise variation ({inconsistency:.2f})",
                is_positive=True,
            ))
        elif inconsistency > 4.5:
            findings.append(Finding(
                Finding.LOW, "Moderate Noise",
                f"Moderate noise variation ({inconsistency:.2f})",
                is_positive=True,
            ))

    if anomaly_count is not None:
        if anomaly_count > 500:
            findings.append(Finding(
                Finding.LOW, "High Anomaly Count",
                f"Many noise anomalies ({anomaly_count}) — typical of real sensor",
                is_positive=True,
            ))
        elif anomaly_count < 100:
            findings.append(Finding(
                Finding.LOW, "Low Anomaly Count",
                f"Few noise anomalies ({anomaly_count}) — unusually uniform",
            ))

    return findings


def _check_frequency_analysis(freq_data: dict) -> List[Finding]:
    """Check 13: Frequency analysis — checkerboard and spectral patterns.

    High checkerboard score = energy at GAN-typical frequencies.
    MEDIUM because heavy JPEG compression can produce similar patterns.
    """
    findings = []
    if not freq_data:
        return findings

    checkerboard = freq_data.get('checkerboard_score')
    if checkerboard is not None:
        if checkerboard <= 90.0:
            findings.append(Finding(
                Finding.HIGH, "Frequency Pattern",
                f"High checkerboard energy: {checkerboard:.1f}%",
                is_positive=False,
            ))
        elif checkerboard < 95.0:
            findings.append(Finding(
                Finding.MEDIUM, "Frequency Pattern",
                f"Low checkerboard energy: {checkerboard:.1f}%",
                is_positive=False,
            ))

    return findings


def _check_opencv_findings(opencv_data: dict) -> List[Finding]:
    """Check 14: OpenCV computer-vision manipulation signals."""
    findings = []
    if not opencv_data or not opencv_data.get('enabled'):
        return findings

    manip = opencv_data.get('manipulation_detection', {})
    if manip:
        confidence = manip.get('confidence', 0.0)
        num_anomalies = manip.get('num_anomalies', 0)

        if confidence > 0.7 and num_anomalies > 1000:
            findings.append(Finding(
                Finding.MEDIUM, "Forensic Manipulation",
                f"Strong manipulation signals: {num_anomalies} anomalies, "
                f"{confidence * 100:.1f}% confidence",
            ))
        elif confidence > 0.5 and num_anomalies > 500:
            findings.append(Finding(
                Finding.LOW, "Forensic Manipulation",
                f"Moderate manipulation signals: {num_anomalies} anomalies, "
                f"{confidence * 100:.1f}% confidence",
            ))
        elif confidence < 0.3 and num_anomalies < 200:
            findings.append(Finding(
                Finding.LOW, "Forensic Clean",
                f"Clean manipulation check: {num_anomalies} anomalies, "
                f"{confidence * 100:.1f}% confidence",
                is_positive=True,
            ))

    noise = opencv_data.get('noise_analysis', {})
    if noise:
        is_inconsistent = noise.get('is_noise_inconsistent', False)
        consistency = noise.get('noise_consistency', 0.0)

        if is_inconsistent and consistency < 0.6:
            findings.append(Finding(
                Finding.MEDIUM, "Forensic Noise",
                f"Inconsistent noise: {consistency * 100:.1f}% consistency",
            ))
        elif consistency > 0.8:
            findings.append(Finding(
                Finding.LOW, "Forensic Noise",
                f"Consistent noise: {consistency * 100:.1f}% consistency",
                is_positive=True,
            ))

    jpeg = opencv_data.get('jpeg_artifacts', {})
    if jpeg:
        is_inconsistent = jpeg.get('has_inconsistent_artifacts', False)
        confidence = jpeg.get('confidence', 0.0)
        compression_var = jpeg.get('compression_variation', 0.0)

        if is_inconsistent and confidence > 0.7:
            findings.append(Finding(
                Finding.MEDIUM, "JPEG Artifacts",
                f"Inconsistent JPEG compression: {compression_var:.2f} "
                f"variation, {confidence * 100:.1f}% confidence",
            ))
        elif is_inconsistent and confidence > 0.5:
            findings.append(Finding(
                Finding.LOW, "JPEG Artifacts",
                f"Minor JPEG inconsistencies: {compression_var:.2f} "
                f"variation, {confidence * 100:.1f}% confidence",
            ))

    return findings


# ─────────────────────────────────────────────────────────────────
# Big picture — cross-detector convergent evidence
# ─────────────────────────────────────────────────────────────────

def _collect_plugin_findings(results: dict) -> List[Finding]:
    """
    Collect standardized audit_findings from all plugins.
    
    Plugins following the data contract should populate:
        results['plugin_name']['audit_findings'] = [...]
    
    This allows plugins to report directly to the auditor without
    needing plugin-specific parsing logic.
    """
    findings = []
    
    # Known plugin names that might have audit_findings
    plugin_names = [
        'photoholmes',
        'ela',
        'noise_analysis',
        'frequency_analysis',
        'opencv_manipulation',
        'opencv_analysis',
        'imghash',
        'signatures',
        'hash',
        'similar',
        # Phase 1c research plugins
        'content_analysis',
    ]
    
    for plugin_name in plugin_names:
        plugin_findings = get_audit_findings(results, plugin_name)
        
        for pf in plugin_findings:
            findings.append(Finding(
                level=pf['level'],
                category=pf['category'],
                description=pf['description'],
                is_positive=pf['is_positive']
            ))
    
    return findings


def _check_convergent_evidence(findings: List[Finding],
                               results: dict) -> List[Finding]:
    """Promote severity when multiple independent detectors agree.

    This is the "better together" idea: any single signal might be
    noise, but when several independent detectors all point the same
    direction, the combined evidence is stronger than the sum of its
    parts.
    """
    extra = []
    neg = [f for f in findings if not f.is_positive]

    # Count distinct negative categories
    neg_categories = set(f.category for f in neg)

    # If 3+ independent negative categories fire, that's convergent
    # evidence worth a MEDIUM negative finding on its own.
    if len(neg_categories) >= 3:
        extra.append(Finding(
            Finding.MEDIUM, "Convergent Evidence",
            f"Multiple independent signals agree: "
            f"{', '.join(sorted(neg_categories))}",
        ))

    # If 3+ independent positive categories fire, same idea.
    pos_categories = set(f.category for f in findings if f.is_positive)
    if len(pos_categories) >= 3:
        extra.append(Finding(
            Finding.MEDIUM, "Convergent Evidence",
            f"Multiple authenticity signals: "
            f"{', '.join(sorted(pos_categories))}",
            is_positive=True,
        ))

    return extra


# ─────────────────────────────────────────────────────────────────
# Score calculation — the ONLY place points are assigned
# ─────────────────────────────────────────────────────────────────

def _calculate_authenticity_score(findings: List[Finding]) -> int:
    """Calculate overall authenticity score (0–100).

    0 = definitely fake, 50 = uncertain, 100 = definitely real.

    Uses the fixed point table:
        LOW = 5,  MEDIUM = 15,  HIGH = 50
    Positive findings add, negative findings subtract.

    Clamping: Each HIGH finding tightens the bound on the opposite
    side by 5, to a maximum squeeze of 50.  This means an image
    with 10 HIGH negatives can never score above 50 no matter how
    many LOW/MEDIUM positives pile up.

        Each HIGH negative  →  ceiling -= 5  (min 50)
        Each HIGH positive  →  floor   += 5  (max 50)
    """
    score = 50  # uncertain baseline

    for f in findings:
        pts = POINTS[f.level]
        if f.is_positive:
            score += pts
        else:
            score -= pts

    # Clamp: each HIGH finding tightens the opposite bound by 5
    high_neg_count = sum(1 for f in findings
                         if f.level == Finding.HIGH and not f.is_positive)
    high_pos_count = sum(1 for f in findings
                         if f.level == Finding.HIGH and f.is_positive)

    ceiling = max(50, 100 - high_neg_count * 5)
    floor = min(50, high_pos_count * 5)

    return max(floor, min(ceiling, score))


def _calculate_ai_probability(findings: List[Finding]) -> float:
    """Estimate AI generation probability from AI-specific findings."""
    ai_neg_pts = 0
    ai_pos_pts = 0

    for f in findings:
        if f.category in ('AI Indicator', 'AI Dimensions',
                          'ML Model Detection', 'ML Model Assessment',
                          'Synthetic Noise', 'Suspicious Noise',
                          'Frequency Pattern'):
            pts = POINTS[f.level]
            if f.is_positive:
                ai_pos_pts += pts
            else:
                ai_neg_pts += pts

    total = ai_neg_pts + ai_pos_pts
    if total == 0:
        return 0.0
    return min(100.0, (ai_neg_pts / total) * 100)


def _calculate_manipulation_probability(findings: List[Finding]) -> float:
    """Estimate manipulation probability from forensic findings."""
    manip_neg_pts = 0
    manip_pos_pts = 0

    for f in findings:
        if f.category in ('Forensic Manipulation', 'Forensic Noise',
                          'Forensic Clean', 'JPEG Artifacts',
                          'Convergent Evidence'):
            pts = POINTS[f.level]
            if f.is_positive:
                manip_pos_pts += pts
            else:
                manip_neg_pts += pts

    total = manip_neg_pts + manip_pos_pts
    if total == 0:
        return 0.0
    return min(100.0, (manip_neg_pts / total) * 100)


def _collect_detector_types(findings: List[Finding]) -> List[str]:
    """Collect detector types that triggered negative findings."""
    category_map = {
        'AI Indicator': 'ai_generation',
        'AI Dimensions': 'ai_dimensions',
        'ML Model Detection': 'ai_generation',
        'Synthetic Noise': 'noise_analysis',
        'Suspicious Noise': 'noise_analysis',
        'Frequency Pattern': 'frequency_analysis',
        'Missing Metadata': 'metadata_anomaly',
        'Minimal EXIF': 'metadata_anomaly',
        'Suspicious EXIF': 'metadata_anomaly',
        'Obsolete Software': 'metadata_anomaly',
        'Forensic Manipulation': 'forensic_analysis',
        'Forensic Noise': 'forensic_analysis',
        'JPEG Artifacts': 'forensic_analysis',
        'Convergent Evidence': 'convergent_evidence',
    }

    detected = []
    for f in findings:
        if f.is_positive:
            continue
        dtype = category_map.get(f.category)
        if dtype and dtype not in detected:
            detected.append(dtype)
    return detected


def _format_evidence(findings: List[Finding],
                     authenticity_score: int) -> str:
    """Format findings into human-readable evidence string."""
    if not findings:
        return f"No significant findings. Authenticity: {authenticity_score}/100"

    neg_high = [f for f in findings if not f.is_positive and f.level == Finding.HIGH]
    neg_med = [f for f in findings if not f.is_positive and f.level == Finding.MEDIUM]
    neg_low = [f for f in findings if not f.is_positive and f.level == Finding.LOW]
    pos = [f for f in findings if f.is_positive]

    parts = [f"Compliance Audit — Authenticity: {authenticity_score}/100"]

    if neg_high:
        parts.append("\n🚨 HIGH:")
        for f in neg_high:
            parts.append(f"  • [{POINTS[f.level]:+d}] {f.description}")

    if neg_med:
        parts.append("\n⚠️  MEDIUM:")
        for f in neg_med:
            parts.append(f"  • [{POINTS[f.level]:+d}] {f.description}")

    if neg_low:
        parts.append("\n⚡ LOW:")
        for f in neg_low:
            parts.append(f"  • [{POINTS[f.level]:+d}] {f.description}")

    if pos:
        parts.append("\n✅ POSITIVE:")
        for f in pos:
            parts.append(f"  • [+{POINTS[f.level]}] {f.description}")

    return "\n".join(parts)
