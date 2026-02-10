# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Confidence scoring — Zero Trust Auditor Model.

Architecture:
    The engine runs the compliance auditor (lib/analyzer/auditor.py) as a
    post-processing step after all plugins complete.  The auditor writes
    its verdict to results['audit'].  This module reads that verdict and
    formats it for storage and display at results['confidence'].

    Both the auditor and this module are engine internals, NOT plugins.
    They always run and cannot be removed or reordered.

    The auditor produces three output buckets:
        1. authenticity_score       (0-100): fake ← → real
        2. ai_probability           (0-100): AI generation likelihood
        3. manipulation_probability (0-100): traditional editing likelihood

    Supporting indicators from individual forensic methods (ELA, noise,
    frequency, photoholmes, hashes) are collected here for transparency
    but do NOT influence the score.
"""

import logging

logger = logging.getLogger(__name__)


def calculate_manipulation_confidence(results):
    """
    Aggregate all detection methods into overall confidence score.

    Returns dict with:
        authenticity_score: 0-100 (NEW: primary metric from compliance audit)
        detected_types: list of what was detected (NEW: from compliance audit)
        verdict: 'ai_generated' | 'manipulated' | 'authentic'
        verdict_label: Human-readable verdict string
        verdict_confidence: 0-100 certainty in the verdict
        verdict_certainty: 'high' | 'moderate' | 'low' | 'inconclusive'
        confidence_score: 0-100 manipulation suspicion score (legacy forensic methods)
        ai_generated_probability: 0-100 AI generation probability (legacy)
        manipulation_detected: bool (legacy compat)
        indicators: list of evidence items
    """
    confidence = {
        'manipulation_detected': False,
        'confidence_score': 0.0,
        'ai_generated_probability': 0.0,
        'authenticity_score': None,  # NEW: 0-100 fake-to-real from compliance audit
        'detected_types': [],  # NEW: what detectors triggered
        'verdict': 'authentic',
        'verdict_label': 'Image Appears Authentic',
        'verdict_confidence': 0.0,
        'verdict_certainty': 'inconclusive',
        'indicators': [],
        'deterministic_methods': {},
        'ai_ml_methods': {},
        'methods': {}  # backward compat
    }

    # ================================================================
    # PRIORITY 1: Check for Compliance Audit Result
    # ================================================================
    # The audit dict is written by the engine after all plugins complete.
    # It contains the authoritative three-bucket scores.

    audit_data = results.get('audit', {})

    if audit_data and 'authenticity_score' in audit_data and audit_data['authenticity_score'] is not None:
        auth_score = audit_data['authenticity_score']
        confidence['authenticity_score'] = auth_score
        confidence['detected_types'] = audit_data.get('detected_types', [])

        # Set AI and manipulation probabilities from audit calculations
        confidence['ai_generated_probability'] = round(audit_data.get('ai_probability', 0.0), 1)
        confidence['confidence_score'] = round(audit_data.get('manipulation_probability', 0.0), 1)

        # Convert authenticity score to verdict
        # 0-40: Fake, 41-59: Uncertain, 60-100: Real
        if auth_score <= 40:
            confidence['verdict'] = 'ai_generated' if 'ai_generation' in confidence['detected_types'] else 'manipulated'
            confidence['manipulation_detected'] = True
            confidence['verdict_confidence'] = round((40 - auth_score) / 40 * 100, 1)

            if auth_score <= 20:
                confidence['verdict_certainty'] = 'high'
                confidence['verdict_label'] = 'Definitely Fake (AI or Manipulated)'
            else:
                confidence['verdict_certainty'] = 'moderate'
                confidence['verdict_label'] = 'Likely Fake (AI or Manipulated)'

        elif auth_score >= 60:
            confidence['verdict'] = 'authentic'
            confidence['manipulation_detected'] = False
            confidence['verdict_confidence'] = round((auth_score - 60) / 40 * 100, 1)

            if auth_score >= 80:
                confidence['verdict_certainty'] = 'high'
                confidence['verdict_label'] = 'Highly Authentic'
            else:
                confidence['verdict_certainty'] = 'moderate'
                confidence['verdict_label'] = 'Likely Authentic'
        else:
            confidence['verdict'] = 'authentic'
            confidence['manipulation_detected'] = False
            confidence['verdict_confidence'] = 20.0  # Low certainty
            confidence['verdict_certainty'] = 'inconclusive'
            confidence['verdict_label'] = 'Uncertain - Borderline Case'

        # Add compliance audit as indicator
        types_str = ', '.join(confidence['detected_types']) if confidence['detected_types'] else 'comprehensive analysis'
        confidence['indicators'].append({
            'method': 'Compliance Audit (Aggregated)',
            'evidence': f"Authenticity score: {auth_score}/100 (detected: {types_str})",
            'type': 'audit'
        })

        # Extract individual auditor findings for dedicated display
        auditor_findings = []
        for finding in audit_data.get('findings', []):
            # Include MEDIUM and HIGH findings for visibility
            if finding['level'] in ('MEDIUM', 'HIGH'):
                auditor_findings.append({
                    'level': finding['level'],
                    'category': finding['category'],
                    'description': finding['description'],
                    'is_positive': finding['is_positive'],
                })
        confidence['auditor_findings'] = auditor_findings

        # Collect supporting indicators for display (not scoring)
        _collect_supporting_indicators(results, confidence)
        return confidence

    # No compliance audit result — report as not analyzed.
    # Individual forensic methods without the auditor cannot produce a
    # meaningful aggregated score.  Rather than inventing numbers from
    # ad-hoc weights, we report that the scoring pipeline didn't run.
    confidence['verdict'] = 'not_analyzed'
    confidence['verdict_label'] = 'Analysis Incomplete — Auditor Did Not Run'
    confidence['verdict_certainty'] = 'inconclusive'
    confidence['verdict_confidence'] = 0.0

    # Still collect whatever raw indicators exist for transparency
    _collect_supporting_indicators(results, confidence)
    return confidence


def _collect_supporting_indicators(results, confidence):
    """
    Collect indicators from individual methods for display only.

    These do NOT influence the score — the auditor already incorporated
    them via its findings system.  They are shown in the UI beneath the
    main verdict for transparency / explainability.
    """
    _add_forensic_indicators(results, confidence)
    _add_ai_detection_indicators(results, confidence)
    _add_photoholmes_indicators(results, confidence)
    _add_hash_evidence_indicators(results, confidence)


# ─────────────────────────────────────────────────────────────────
# Supporting indicator collectors (display only, no scoring)
# ─────────────────────────────────────────────────────────────────

def _add_forensic_indicators(results, confidence):
    """Collect forensic method indicators for display."""

    # ELA
    if 'ela' in results and 'max_difference' in results['ela']:
        ela_score = min(results['ela']['max_difference'] / 100.0, 1.0)
        if ela_score > 0.3:
            confidence['deterministic_methods']['ela'] = ela_score
            confidence['methods']['ela'] = ela_score
            confidence['indicators'].append({
                'method': 'ELA Analysis',
                'evidence': f"High error level detected (score: {ela_score:.2f})",
                'type': 'deterministic',
            })

    # Noise
    if 'noise_analysis' in results:
        raw_inconsistency = results['noise_analysis'].get('inconsistency_score', 0)
        is_suspicious = results['noise_analysis'].get('suspicious', False)

        if is_suspicious or raw_inconsistency > 2.0:
            label = "Suspicious" if is_suspicious else "Elevated"
            confidence['indicators'].append({
                'method': 'Noise Analysis',
                'evidence': f"{label} noise inconsistency ({raw_inconsistency:.1f}%)",
                'type': 'deterministic',
            })

    # Frequency
    if 'frequency_analysis' in results:
        freq = results['frequency_analysis']
        freq_signals = []

        if freq.get('suspicious', False):
            freq_signals.append(f"anomaly score: {freq.get('anomaly_score', 0):.1f}%")

        checkerboard = freq.get('checkerboard_score', 100.0)
        if checkerboard < 99.5:
            freq_signals.append(f"checkerboard: {checkerboard:.1f}/100")

        peak_ratio = freq.get('peak_ratio', 0)
        if peak_ratio > 0.4:
            freq_signals.append(f"peak ratio: {peak_ratio:.2f}")

        if freq_signals:
            confidence['indicators'].append({
                'method': 'Frequency Analysis',
                'evidence': "; ".join(freq_signals),
                'type': 'deterministic',
            })

    # Metadata
    if 'metadata' in results:
        meta = results['metadata']
        meta_signals = []

        has_exif = bool(meta.get('Exif'))
        has_iptc = bool(meta.get('Iptc'))
        has_xmp = bool(meta.get('Xmp'))

        if not has_exif and not has_iptc and not has_xmp:
            meta_signals.append("No EXIF, IPTC, or XMP metadata")
        else:
            if not meta.get('Exif', {}).get('Image', {}).get('Make'):
                meta_signals.append("Missing camera make")
            if not meta.get('Exif', {}).get('Image', {}).get('Model'):
                meta_signals.append("Missing camera model")

        if meta_signals:
            confidence['indicators'].append({
                'method': 'Metadata Analysis',
                'evidence': "; ".join(meta_signals),
                'type': 'deterministic',
            })

    # OpenCV
    if 'opencv_manipulation' in results and results['opencv_manipulation'].get('enabled', False):
        opencv_suspicious = results['opencv_manipulation'].get('is_suspicious', False)
        opencv_score = results['opencv_manipulation'].get('overall_confidence', 0)

        if opencv_suspicious:
            evidence_parts = []
            manip = results['opencv_manipulation'].get('manipulation_detection', {})
            if manip.get('is_manipulated'):
                evidence_parts.append(f"Gaussian blur: {manip['confidence']*100:.1f}%")
            noise = results['opencv_manipulation'].get('noise_analysis', {})
            if noise.get('is_noise_inconsistent'):
                evidence_parts.append(f"Noise: {noise.get('confidence', 0)*100:.1f}%")
            jpeg = results['opencv_manipulation'].get('jpeg_artifacts', {})
            if jpeg.get('has_inconsistent_artifacts'):
                evidence_parts.append(f"JPEG artifacts: {jpeg['confidence']*100:.1f}%")

            evidence = (
                "; ".join(evidence_parts)
                if evidence_parts
                else f"Overall suspicion: {opencv_score*100:.1f}%"
            )
            confidence['indicators'].append({
                'method': 'OpenCV Manipulation (Computer Vision)',
                'evidence': evidence,
                'type': 'ai_ml',
            })


def _add_ai_detection_indicators(results, confidence):
    """Collect AI detection indicators for display."""
    if 'ai_detection' not in results or not results['ai_detection'].get('enabled', False):
        return

    ai_det = results['ai_detection']
    layers = ai_det.get('detection_layers', [])

    for layer in layers:
        verdict = layer.get('verdict', 'Unknown')
        method = layer.get('method', 'Unknown')
        score = layer.get('score')
        layer_confidence = layer.get('confidence', 'unknown')

        if verdict == 'AI' or (score is not None and score > 0.2):
            evidence = f"Verdict: {verdict}"
            if score is not None:
                evidence += f" (score: {score:.1%})"
            evidence += f" — {layer_confidence} confidence"

            confidence['indicators'].append({
                'method': f'AI Detection ({method})',
                'evidence': evidence,
                'type': 'ai_ml',
            })


def _add_photoholmes_indicators(results, confidence):
    """Collect photoholmes forgery detection indicators for display."""
    if 'photoholmes' not in results or not results['photoholmes'].get('enabled', False):
        return

    summary = results['photoholmes'].get('summary', {})
    methods = results['photoholmes'].get('methods', {})
    methods_run = summary.get('methods_run', 0)

    if methods_run == 0:
        return

    avg_score = summary.get('avg_detection_score', 0.0)
    consensus = summary.get('consensus_forgery', False)

    if avg_score > 0.05 or consensus:
        evidence_parts = []
        for method_key, method_result in methods.items():
            if isinstance(method_result, dict) and 'detection_score' in method_result:
                name = method_result.get('method', method_key)
                score = method_result['detection_score']
                if score is not None:
                    label = "FORGED" if method_result.get('forgery_detected') else "clean"
                    evidence_parts.append(f"{name}: {score:.1%} ({label})")

        evidence = (
            "; ".join(evidence_parts)
            if evidence_parts
            else f"Avg score: {avg_score:.1%}"
        )

        confidence['indicators'].append({
            'method': f'Photoholmes Forgery Detection ({methods_run} methods)',
            'evidence': evidence,
            'type': 'deterministic',
        })


def _add_hash_evidence_indicators(results, confidence):
    """Collect perceptual hash evidence indicators for display."""
    similar = results.get('similar', {})

    # Perceptual hash list matches
    hash_list_matches = similar.get('hash_list_matches', [])
    if hash_list_matches:
        list_names = set(m.get('list_name', 'Unknown') for m in hash_list_matches)
        confidence['indicators'].append({
            'method': 'Perceptual Hash List Match',
            'evidence': (
                f"{len(hash_list_matches)} matches across lists: "
                f"{', '.join(list_names)}"
            ),
            'type': 'deterministic',
        })

    # Cross-case links (informational)
    cross_case = similar.get('cross_case', [])
    if cross_case:
        exact_cross = [m for m in cross_case if m.get('classification') == 'exact']
        near_cross = [m for m in cross_case if m.get('classification') == 'near_duplicate']

        if exact_cross or near_cross:
            confidence['indicators'].append({
                'method': 'Cross-Case Image Link',
                'evidence': (
                    f"{len(exact_cross)} exact + {len(near_cross)} near-duplicate "
                    f"matches across "
                    f"{len(set(m.get('case_id') for m in cross_case))} other cases"
                ),
                'type': 'informational',
            })
