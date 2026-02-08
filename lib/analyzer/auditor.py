# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
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

Philosophy: Zero Trust — start at 50 (uncertain), build or erode
confidence based on verifiable findings.  Each finding has a risk
level (POSITIVE / LOW / MEDIUM / HIGH) and a score_impact.

Score capping ensures 100 means "zero doubt" and 0 means "definitive
fake" — contradicting evidence always pulls the score away from the
extremes.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Finding — the atomic unit of audit evidence
# ─────────────────────────────────────────────────────────────────

class Finding:
    """Represents a single audit finding."""

    POSITIVE = "POSITIVE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    def __init__(self, risk_level: str, category: str, description: str, score_impact: int):
        self.risk_level = risk_level
        self.category = category
        self.description = description
        self.score_impact = score_impact


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
#       .Image.Make, .Image.Model, .Image.Software, .Image.DateTime
#       .Photo.ISOSpeedRatings, .Photo.ExposureTime, .Photo.FNumber,
#              .Photo.FocalLength
#     .gps            (dict)  — GPS position data (if present)
#   noise_analysis    (dict)  — noise plugin (order 25)
#     .inconsistency_score, .anomaly_count, .suspicious
#   frequency_analysis(dict)  — frequency plugin (order 26)
#   opencv_manipulation(dict) — opencv plugin (order 65)
#     .enabled, .manipulation_detection, .noise_analysis, .jpeg_artifacts
#   ai_detection      (dict)  — ai_detection plugin (order 30)
#     .enabled, .detection_layers[]
#       each layer: {method, evidence, verdict, confidence, score}
#
# The auditor writes:
#   audit             (dict)  — this module
#     .authenticity_score, .ai_probability, .manipulation_probability,
#     .findings_count, .evidence, .detected_types, .findings_summary
#


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
    'Picasa': {'discontinued': 2016, 'risk': 'HIGH'},
    'Windows Photo Gallery': {'discontinued': 2012, 'risk': 'HIGH'},
    'iPhoto': {'discontinued': 2015, 'risk': 'HIGH'},
    'Adobe Photoshop Album': {'discontinued': 2006, 'risk': 'HIGH'},
    'Photoshop Express': {'version_check': True, 'risk': 'MEDIUM'},
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
    '4:3': [(640, 480), (800, 600), (1024, 768), (1600, 1200), (2048, 1536), (4000, 3000)],
    '3:2': [(1500, 1000), (3000, 2000), (4288, 2848), (6000, 4000)],
    '16:9': [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (4096, 2160)],
    '1:1': [(1080, 1080)],
}


# ─────────────────────────────────────────────────────────────────
# Public API — called by _process_image()
# ─────────────────────────────────────────────────────────────────

def audit(results: dict) -> dict:
    """
    Run the compliance audit on accumulated plugin results.

    This is the single entry point called by the engine after all
    plugins have completed.  It reads everything from ``results``
    and returns a dict to be stored at ``results['audit']``.

    Args:
        results: The accumulated results dict from all plugins.

    Returns:
        dict with keys:
            authenticity_score      (int 0-100)
            ai_probability          (float 0-100)
            manipulation_probability(float 0-100)
            findings_count          (int)
            evidence                (str, human-readable)
            detected_types          (list of str)
            findings_summary        (dict, counts by risk level)
    """
    findings: List[Finding] = []

    # Extract commonly-used data from results
    file_name = results.get('file_name', '')
    metadata = results.get('metadata', {})
    exif_data = metadata.get('Exif', {})
    dimensions = metadata.get('dimensions', [0, 0])
    width = dimensions[0] if len(dimensions) >= 2 else 0
    height = dimensions[1] if len(dimensions) >= 2 else 0

    # === HIGH-RISK CHECKS ===
    findings.extend(_check_ai_indicators(file_name, exif_data))
    findings.extend(_check_obsolete_software(exif_data))

    # === MEDIUM-RISK CHECKS ===
    findings.extend(_check_ai_dimensions(width, height))
    findings.extend(_check_suspicious_exif(exif_data))

    # === LOW-RISK CHECKS ===
    findings.extend(_check_missing_metadata(exif_data))
    findings.extend(_check_minimal_exif(exif_data))

    # === POSITIVE CHECKS ===
    findings.extend(_check_legitimate_camera(exif_data))
    findings.extend(_check_gps_data(metadata))
    findings.extend(_check_camera_settings(exif_data))
    findings.extend(_check_photo_resolution(width, height))

    # === ML MODEL RESULTS (from ai_detection plugin) ===
    ai_det = results.get('ai_detection', {})
    if ai_det.get('enabled'):
        findings.extend(_check_ml_model_results(ai_det.get('detection_layers', [])))

    # === FORENSIC METHOD RESULTS ===
    findings.extend(_check_noise_consistency(results.get('noise_analysis', {})))
    findings.extend(_check_opencv_findings(results.get('opencv_manipulation', {})))

    # === CALCULATE SCORES ===
    authenticity_score = _calculate_authenticity_score(findings)
    detected_types = _collect_detector_types(findings)
    evidence = _format_evidence(findings, authenticity_score)
    ai_probability = _calculate_ai_probability(findings)
    manipulation_probability = _calculate_manipulation_probability(findings)

    return {
        'authenticity_score': authenticity_score,
        'ai_probability': ai_probability,
        'manipulation_probability': manipulation_probability,
        'findings_count': len(findings),
        'evidence': evidence,
        'detected_types': detected_types,
        'findings_summary': {
            'high': sum(1 for f in findings if f.risk_level == Finding.HIGH),
            'medium': sum(1 for f in findings if f.risk_level == Finding.MEDIUM),
            'low': sum(1 for f in findings if f.risk_level == Finding.LOW),
            'positive': sum(1 for f in findings if f.risk_level == Finding.POSITIVE),
        },
    }


# ─────────────────────────────────────────────────────────────────
# Individual checks — each returns a list of Finding objects
# ─────────────────────────────────────────────────────────────────

def _check_ai_indicators(file_name: str, exif_data: dict) -> List[Finding]:
    """Check 1: AI generator keywords in filename or software tag."""
    findings = []
    targets = [file_name.lower()]

    software = exif_data.get('Image', {}).get('Software', '')
    if software:
        targets.append(software.lower())

    for indicator in AI_INDICATORS:
        if any(indicator in t for t in targets):
            findings.append(Finding(
                Finding.HIGH, "AI Indicator",
                f"AI generator keyword detected: '{indicator}'",
                -100,
            ))
            break

    return findings


def _check_obsolete_software(exif_data: dict) -> List[Finding]:
    """Check 2: Obsolete or suspicious editing software."""
    findings = []
    software = exif_data.get('Image', {}).get('Software', '')
    if not software:
        return findings

    for obs_name, info in OBSOLETE_SOFTWARE.items():
        if obs_name.lower() in software.lower():
            risk = info['risk']
            impact = -100 if risk == 'HIGH' else -60
            findings.append(Finding(
                risk, "Obsolete Software",
                f"Obsolete/suspicious software: '{software}' "
                f"(discontinued {info.get('discontinued', '?')})",
                impact,
            ))
    return findings


def _check_ai_dimensions(width: int, height: int) -> List[Finding]:
    """Check 3: AI-typical power-of-2 square dimensions."""
    if width == height and (width, height) in AI_TYPICAL_DIMENSIONS:
        return [Finding(
            Finding.MEDIUM, "AI Dimensions",
            f"Perfect square power-of-2 dimensions: {width}x{height}",
            -60,
        )]
    return []


def _check_suspicious_exif(exif_data: dict) -> List[Finding]:
    """Check 4: Suspicious EXIF patterns (e.g. all-zero unique IDs)."""
    findings = []
    # Check across all EXIF groups for ImageUniqueID
    for group in exif_data.values():
        if isinstance(group, dict):
            unique_id = group.get('ImageUniqueID') or group.get('image_unique_id')
            if unique_id and str(unique_id).endswith('00000000000000'):
                findings.append(Finding(
                    Finding.MEDIUM, "Suspicious EXIF",
                    f"Suspicious ImageUniqueId: {unique_id}",
                    -30,
                ))
                break
    return findings


def _check_missing_metadata(exif_data: dict) -> List[Finding]:
    """Check 5: Completely missing EXIF metadata."""
    if not exif_data:
        return [Finding(
            Finding.LOW, "Missing Metadata",
            "No EXIF metadata found (common in AI-generated or heavily edited images)",
            -15,
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
            f"Suspiciously few EXIF tags: {tag_count} (real cameras have 50+)",
            -10,
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
                    Finding.POSITIVE, "Legitimate Camera",
                    f"Verified camera signature: {make_clean} {model_clean}",
                    +50,
                )]
            for valid_model in models:
                if valid_model.lower() in model_clean.lower():
                    return [Finding(
                        Finding.POSITIVE, "Legitimate Camera",
                        f"Verified camera signature: {make_clean} {model_clean}",
                        +50,
                    )]
    return []


def _check_gps_data(metadata: dict) -> List[Finding]:
    """Check 8: GPS location data present."""
    gps = metadata.get('gps')
    if gps:
        return [Finding(
            Finding.POSITIVE, "GPS Data",
            "GPS location data present (rare in AI-generated images)",
            +30,
        )]
    return []


def _check_camera_settings(exif_data: dict) -> List[Finding]:
    """Check 9: Realistic camera settings (ISO, exposure, aperture, focal length)."""
    photo_info = exif_data.get('Photo', {})
    if not photo_info:
        return []

    has_iso = 'ISOSpeedRatings' in photo_info
    has_exposure = 'ExposureTime' in photo_info
    has_aperture = 'FNumber' in photo_info
    has_focal = 'FocalLength' in photo_info
    count = sum([has_iso, has_exposure, has_aperture, has_focal])

    if count >= 3:
        return [Finding(
            Finding.POSITIVE, "Camera Settings",
            f"Realistic camera settings present ({count}/4 key settings)",
            +25,
        )]
    return []


def _check_photo_resolution(width: int, height: int) -> List[Finding]:
    """Check 10: Standard camera resolution and natural aspect ratio."""
    findings = []

    for aspect_name, resolutions in CAMERA_RESOLUTIONS.items():
        if (width, height) in resolutions or (height, width) in resolutions:
            findings.append(Finding(
                Finding.POSITIVE, "Photo Resolution",
                f"Standard camera resolution: {width}x{height} ({aspect_name} aspect ratio)",
                +20,
            ))
            break

    if width > 0 and height > 0:
        aspect_ratio = width / height
        if (1.2 < aspect_ratio < 2.0 or 0.5 < aspect_ratio < 0.83) and abs(aspect_ratio - 1.0) > 0.15:
            findings.append(Finding(
                Finding.POSITIVE, "Natural Aspect Ratio",
                f"Aspect ratio {aspect_ratio:.2f}:1 is typical for photos (not perfect square)",
                +10,
            ))

    return findings


def _check_ml_model_results(detection_layers: list) -> List[Finding]:
    """
    Check 11: ML model detector results from ai_detection plugin.

    Reads the detection_layers list that the ai_detection plugin wrote.
    Each layer has: method, evidence, verdict, confidence, score.
    """
    findings = []

    for layer in detection_layers:
        method = layer.get('method', '')
        if method != 'ml_model':
            continue

        verdict = layer.get('verdict')
        score = layer.get('score', 0.0)
        confidence = layer.get('confidence', 'NONE')
        evidence = layer.get('evidence', method)

        if verdict == 'AI' and score is not None and score > 0.5:
            findings.append(Finding(
                Finding.HIGH, "ML Model Detection",
                f"ML model detected AI generation: {score:.1%} probability ({evidence})",
                -100,
            ))
        elif verdict == 'Real' and score is not None and score < 0.3 and confidence in ('HIGH', 'MEDIUM', 'CERTAIN'):
            findings.append(Finding(
                Finding.POSITIVE, "ML Model Assessment",
                f"ML model suggests authentic: {(1 - score):.1%} confidence ({evidence})",
                +10,
            ))

    return findings


def _check_noise_consistency(noise_data: dict) -> List[Finding]:
    """Check 12: Noise analysis — synthetic vs natural sensor patterns."""
    findings = []
    if not noise_data:
        return findings

    inconsistency = noise_data.get('inconsistency_score')
    anomaly_count = noise_data.get('anomaly_count')

    if inconsistency is not None:
        if inconsistency < 3.0:
            findings.append(Finding(
                Finding.HIGH, "Synthetic Noise Pattern",
                f"Unnaturally uniform noise (inconsistency: {inconsistency:.2f}) "
                "- typical of AI generation",
                -70,
            ))
        elif inconsistency < 4.2:
            findings.append(Finding(
                Finding.MEDIUM, "Suspicious Noise Pattern",
                f"Low noise inconsistency ({inconsistency:.2f}) - may indicate synthetic origin",
                -40,
            ))
        elif inconsistency > 5.5:
            findings.append(Finding(
                Finding.POSITIVE, "Natural Noise Pattern",
                f"Natural noise variation ({inconsistency:.2f}) - consistent with real camera",
                +35,
            ))
        elif inconsistency > 4.8:
            findings.append(Finding(
                Finding.POSITIVE, "Moderate Noise Variation",
                f"Moderate noise variation ({inconsistency:.2f}) - likely authentic",
                +20,
            ))

    if anomaly_count is not None:
        if anomaly_count > 500:
            findings.append(Finding(
                Finding.POSITIVE, "High Noise Variation",
                f"Many noise anomalies ({anomaly_count}) - typical of real sensor",
                +15,
            ))
        elif anomaly_count < 100:
            findings.append(Finding(
                Finding.LOW, "Low Noise Variation",
                f"Few noise anomalies ({anomaly_count}) - unusually uniform",
                -10,
            ))

    return findings


def _check_opencv_findings(opencv_data: dict) -> List[Finding]:
    """Check 13: OpenCV computer-vision manipulation signals."""
    findings = []
    if not opencv_data or not opencv_data.get('enabled'):
        return findings

    # Gaussian Blur manipulation detection
    manip = opencv_data.get('manipulation_detection', {})
    if manip:
        confidence = manip.get('confidence', 0.0)
        num_anomalies = manip.get('num_anomalies', 0)

        if confidence > 0.7 and num_anomalies > 1000:
            findings.append(Finding(
                Finding.HIGH, "Forensic Manipulation",
                f"Strong manipulation signals: {num_anomalies} anomalies, "
                f"{confidence * 100:.1f}% confidence",
                -60,
            ))
        elif confidence > 0.5 and num_anomalies > 500:
            findings.append(Finding(
                Finding.MEDIUM, "Forensic Manipulation",
                f"Moderate manipulation signals: {num_anomalies} anomalies, "
                f"{confidence * 100:.1f}% confidence",
                -35,
            ))
        elif confidence > 0.3 or num_anomalies > 200:
            findings.append(Finding(
                Finding.LOW, "Forensic Manipulation",
                f"Minor manipulation signals: {num_anomalies} anomalies, "
                f"{confidence * 100:.1f}% confidence",
                -15,
            ))
        else:
            findings.append(Finding(
                Finding.POSITIVE, "Forensic Clean",
                f"No significant manipulation: {num_anomalies} anomalies, "
                f"{confidence * 100:.1f}% confidence",
                +10,
            ))

    # OpenCV noise consistency
    noise = opencv_data.get('noise_analysis', {})
    if noise:
        is_inconsistent = noise.get('is_noise_inconsistent', False)
        consistency_score = noise.get('noise_consistency', 0.0)

        if is_inconsistent and consistency_score < 0.6:
            findings.append(Finding(
                Finding.MEDIUM, "Forensic Manipulation",
                f"Inconsistent noise patterns: {consistency_score * 100:.1f}% consistency",
                -30,
            ))
        elif consistency_score > 0.8:
            findings.append(Finding(
                Finding.POSITIVE, "Forensic Clean",
                f"Consistent noise patterns: {consistency_score * 100:.1f}% consistency",
                +8,
            ))

    # JPEG artifact analysis
    jpeg = opencv_data.get('jpeg_artifacts', {})
    if jpeg:
        is_inconsistent = jpeg.get('has_inconsistent_artifacts', False)
        confidence = jpeg.get('confidence', 0.0)
        compression_var = jpeg.get('compression_variation', 0.0)

        if is_inconsistent and confidence > 0.7:
            findings.append(Finding(
                Finding.MEDIUM, "Forensic Manipulation",
                f"Inconsistent JPEG compression: {compression_var:.2f} variation, "
                f"{confidence * 100:.1f}% confidence",
                -25,
            ))
        elif is_inconsistent and confidence > 0.5:
            findings.append(Finding(
                Finding.LOW, "Forensic Manipulation",
                f"Minor JPEG inconsistencies: {compression_var:.2f} variation, "
                f"{confidence * 100:.1f}% confidence",
                -12,
            ))

    return findings


# ─────────────────────────────────────────────────────────────────
# Score calculation
# ─────────────────────────────────────────────────────────────────

def _calculate_authenticity_score(findings: List[Finding]) -> int:
    """
    Calculate overall authenticity score (0–100).

    0 = definitely fake, 50 = uncertain, 100 = definitely real.

    Score capping:
    - Each HIGH risk finding lowers the ceiling by 10 (min 55).
    - Each MEDIUM risk finding lowers the ceiling by 5 (min 55).
    - Each POSITIVE finding raises the floor by 5 (max 45).
    - 100 means "zero doubt", not "preponderance of evidence".
    """
    base_score = 50
    for finding in findings:
        base_score += finding.score_impact

    ceiling = 100
    floor = 0

    high_count = sum(1 for f in findings if f.risk_level == Finding.HIGH)
    medium_count = sum(1 for f in findings if f.risk_level == Finding.MEDIUM)
    positive_count = sum(1 for f in findings if f.risk_level == Finding.POSITIVE)

    if high_count > 0:
        ceiling -= high_count * 10
        ceiling = max(55, ceiling)

    if medium_count > 0:
        ceiling -= medium_count * 5
        ceiling = max(55, ceiling)

    if positive_count > 0:
        floor += positive_count * 5
        floor = min(45, floor)

    return max(floor, min(ceiling, base_score))


def _calculate_ai_probability(findings: List[Finding]) -> float:
    """
    Calculate AI generation probability (0–100) from AI-specific findings.
    """
    ai_score = 0.0
    max_possible = 0.0

    for finding in findings:
        if 'AI' in finding.category or 'Synthetic' in finding.category or 'ML Model' in finding.category:
            weight = {
                Finding.HIGH: 40.0, Finding.MEDIUM: 25.0,
                Finding.LOW: 10.0, Finding.POSITIVE: 0.0,
            }.get(finding.risk_level, 0)
            max_possible += weight
            if finding.risk_level != Finding.POSITIVE:
                ai_score += weight

        elif 'dimension' in finding.description.lower() and finding.risk_level in (Finding.HIGH, Finding.MEDIUM):
            weight = 30.0 if finding.risk_level == Finding.HIGH else 15.0
            max_possible += weight
            ai_score += weight

        elif 'noise' in finding.category.lower() and 'uniform' in finding.description.lower():
            if finding.risk_level == Finding.HIGH:
                max_possible += 35.0
                ai_score += 35.0

    if max_possible > 0:
        return min(100.0, (ai_score / max_possible) * 100)
    return 0.0


def _calculate_manipulation_probability(findings: List[Finding]) -> float:
    """
    Calculate manipulation probability (0–100) from forensic findings.
    """
    manip_score = 0.0
    max_possible = 0.0

    for finding in findings:
        if 'Forensic' in finding.category or 'Manipulation' in finding.category:
            if finding.risk_level == Finding.POSITIVE:
                continue
            weight = {
                Finding.HIGH: 40.0, Finding.MEDIUM: 25.0, Finding.LOW: 10.0,
            }.get(finding.risk_level, 0)
            max_possible += weight
            manip_score += weight

        elif 'noise' in finding.category.lower() and 'uniform' not in finding.description.lower():
            if finding.risk_level == Finding.POSITIVE:
                continue
            if finding.risk_level in (Finding.HIGH, Finding.MEDIUM):
                weight = 30.0 if finding.risk_level == Finding.HIGH else 20.0
                max_possible += weight
                manip_score += weight

        elif 'frequency' in finding.category.lower():
            if finding.risk_level in (Finding.HIGH, Finding.MEDIUM):
                weight = 25.0 if finding.risk_level == Finding.HIGH else 15.0
                max_possible += weight
                manip_score += weight

        elif 'metadata' in finding.category.lower():
            if finding.risk_level in (Finding.MEDIUM, Finding.LOW):
                weight = 15.0 if finding.risk_level == Finding.MEDIUM else 5.0
                max_possible += weight
                manip_score += weight

    if max_possible > 0:
        return min(100.0, (manip_score / max_possible) * 100)
    return 0.0


def _collect_detector_types(findings: List[Finding]) -> List[str]:
    """Collect detector types that triggered (informational)."""
    detected_types = []

    for finding in findings:
        if finding.risk_level not in (Finding.HIGH, Finding.MEDIUM):
            continue

        if 'AI' in finding.category or 'Synthetic' in finding.category:
            if 'ai_generation' not in detected_types:
                detected_types.append('ai_generation')

        if 'dimension' in finding.description.lower() and '1024' in finding.description:
            if 'ai_dimensions' not in detected_types:
                detected_types.append('ai_dimensions')

        if 'noise' in finding.category.lower():
            if 'noise_analysis' not in detected_types:
                detected_types.append('noise_analysis')

        if 'metadata' in finding.category.lower() or 'EXIF' in finding.description:
            if 'metadata_anomaly' not in detected_types:
                detected_types.append('metadata_anomaly')

        if 'frequency' in finding.category.lower():
            if 'frequency_analysis' not in detected_types:
                detected_types.append('frequency_analysis')

    return detected_types


def _format_evidence(findings: List[Finding], authenticity_score: int) -> str:
    """Format findings into human-readable evidence string."""
    if not findings:
        return f"No significant findings. Authenticity: {authenticity_score}/100"

    by_risk = {Finding.HIGH: [], Finding.MEDIUM: [], Finding.LOW: [], Finding.POSITIVE: []}
    for f in findings:
        by_risk[f.risk_level].append(f)

    parts = [f"Compliance Audit - Authenticity: {authenticity_score}/100"]

    if by_risk[Finding.HIGH]:
        parts.append("\n🚨 HIGH RISK:")
        for f in by_risk[Finding.HIGH]:
            parts.append(f"  • {f.description}")

    if by_risk[Finding.MEDIUM]:
        parts.append("\n⚠️  MEDIUM RISK:")
        for f in by_risk[Finding.MEDIUM]:
            parts.append(f"  • {f.description}")

    if by_risk[Finding.LOW]:
        parts.append("\n⚡ LOW RISK:")
        for f in by_risk[Finding.LOW]:
            parts.append(f"  • {f.description}")

    if by_risk[Finding.POSITIVE]:
        parts.append("\n✅ POSITIVE EVIDENCE:")
        for f in by_risk[Finding.POSITIVE]:
            parts.append(f"  • {f.description}")

    return "\n".join(parts)
