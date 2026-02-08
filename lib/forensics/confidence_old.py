# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Confidence scoring system for aggregating detection results.

Architecture:
    Two independent verdicts are computed:

    1. AI GENERATION VERDICT
       - Deterministic proof (metadata/filename) is treated as an override —
         a filename like "gemini_generated_xxx.jpg" is near-certain evidence.
       - ML model probability is the primary probabilistic signal.
       - OpenCV corroboration can boost the probability when ML alone is low.
       - Absence of deterministic markers is NOT evidence against AI — removing
         metadata is trivial.

    2. MANIPULATION VERDICT
       - Weighted aggregate of forensic methods: ELA, noise, frequency,
         OpenCV manipulation detection.
       - Metadata absence contributes weakly (it's common in non-manipulated
         images too).

    The final verdict picks the stronger signal: if AI detection fires with
    high confidence, that overrides the manipulation score.
"""

import logging

logger = logging.getLogger(__name__)


def _collect_legacy_indicators(results, confidence, override_verdict=True):
    """
    Collect forensic indicators from legacy methods.
    
    Args:
        results: Detection results dict
        confidence: Confidence dict to populate
        override_verdict: If False, only collect indicators without changing verdict
    """
    # ================================================================
    # PHASE 1: Collect forensic evidence (manipulation suspicion score)
    # ================================================================
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
    # PRIORITY 1: Check for Compliance Audit Result (NEW)
    # ================================================================
    # The compliance audit aggregates ALL evidence (forensics + detectors)
    # into a single 0-100 authenticity score. When present, this is the
    # primary truth and should be displayed prominently.
    
    if 'ai_detection' in results and results['ai_detection'].get('enabled', False):
        ai_det = results['ai_detection']
        
        # Check if we have the new authenticity_score from compliance audit
        if 'authenticity_score' in ai_det and ai_det['authenticity_score'] is not None:
            auth_score = ai_det['authenticity_score']
            confidence['authenticity_score'] = auth_score
            confidence['detected_types'] = ai_det.get('detected_types', [])
            
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
            
            # Still collect legacy indicators for display, but audit is primary
            # Continue to phases 1-2 but don't override verdict
            _collect_legacy_indicators(results, confidence, override_verdict=False)
            return confidence

    # ================================================================
    # PHASE 1: Collect forensic evidence (manipulation suspicion score)
    # ================================================================
    # This is a weighted aggregate of image forensic methods.
    # It answers: "Does this image show signs of editing/tampering?"
    # AI detection is NOT included here — it's a separate question.

    forensic_score = 0.0
    forensic_weights = {
        'ela': 0.20,
        'noise': 0.20,
        'frequency': 0.15,
        'metadata': 0.05,
        'opencv': 0.40,
    }

    # --- ELA ---
    if 'ela' in results and 'max_difference' in results['ela']:
        ela_score = min(results['ela']['max_difference'] / 100.0, 1.0)
        if ela_score > 0.3:
            forensic_score += forensic_weights['ela'] * ela_score
            confidence['deterministic_methods']['ela'] = ela_score
            confidence['methods']['ela'] = ela_score
            confidence['indicators'].append({
                'method': 'ELA Analysis (Deterministic)',
                'evidence': f"High error level detected (score: {ela_score:.2f})",
                'type': 'deterministic'
            })

    # --- Noise ---
    if 'noise_analysis' in results:
        raw_inconsistency = results['noise_analysis'].get('inconsistency_score', 0)
        is_suspicious = results['noise_analysis'].get('suspicious', False)

        if is_suspicious:
            noise_score = min(raw_inconsistency / 100.0, 1.0)
        elif raw_inconsistency > 2.0:
            noise_score = min((raw_inconsistency - 2.0) / 18.0, 1.0) * 0.5
        else:
            noise_score = 0

        if noise_score > 0:
            forensic_score += forensic_weights['noise'] * noise_score
            confidence['deterministic_methods']['noise'] = noise_score
            confidence['methods']['noise'] = noise_score
            label = "Suspicious" if is_suspicious else "Elevated"
            confidence['indicators'].append({
                'method': 'Noise Analysis (Deterministic)',
                'evidence': f"{label} noise inconsistency ({raw_inconsistency:.1f}%)",
                'type': 'deterministic'
            })

    # --- Frequency ---
    if 'frequency_analysis' in results:
        freq = results['frequency_analysis']
        freq_signals = []
        freq_total = 0.0

        anomaly_score = freq.get('anomaly_score', 0)
        if freq.get('suspicious', False):
            anomaly_contrib = min(anomaly_score / 100.0, 1.0)
            freq_total += anomaly_contrib * 0.4
            freq_signals.append(f"anomaly score: {anomaly_score:.1f}%")

        checkerboard = freq.get('checkerboard_score', 100.0)
        if checkerboard < 99.5:
            cb_contrib = min((99.5 - checkerboard) / 14.5, 1.0)
            freq_total += cb_contrib * 0.35
            freq_signals.append(f"checkerboard: {checkerboard:.1f}/100")

        peak_ratio = freq.get('peak_ratio', 0)
        if peak_ratio > 0.4:
            pr_contrib = min((peak_ratio - 0.4) / 0.6, 1.0)
            freq_total += pr_contrib * 0.25
            freq_signals.append(f"peak ratio: {peak_ratio:.2f}")

        if freq_total > 0:
            forensic_score += forensic_weights['frequency'] * freq_total
            confidence['deterministic_methods']['frequency'] = freq_total
            confidence['methods']['frequency'] = freq_total
            confidence['indicators'].append({
                'method': 'Frequency Analysis (Deterministic)',
                'evidence': "; ".join(freq_signals),
                'type': 'deterministic'
            })

    # --- Metadata (weak signal for manipulation) ---
    if 'metadata' in results:
        meta = results['metadata']
        meta_signals = []
        meta_score = 0.0

        has_exif = bool(meta.get('Exif'))
        has_iptc = bool(meta.get('Iptc'))
        has_xmp = bool(meta.get('Xmp'))

        if not has_exif and not has_iptc and not has_xmp:
            meta_score = 1.0
            meta_signals.append("No EXIF, IPTC, or XMP metadata")
        else:
            if not meta.get('Exif', {}).get('Image', {}).get('Make'):
                meta_score += 0.3
                meta_signals.append("Missing camera make")
            if not meta.get('Exif', {}).get('Image', {}).get('Model'):
                meta_score += 0.3
                meta_signals.append("Missing camera model")

        if meta_score > 0:
            forensic_score += forensic_weights['metadata'] * meta_score
            confidence['deterministic_methods']['metadata'] = meta_score
            confidence['methods']['metadata'] = meta_score
            confidence['indicators'].append({
                'method': 'Metadata Analysis (Deterministic)',
                'evidence': "; ".join(meta_signals),
                'type': 'deterministic'
            })

    # --- OpenCV manipulation ---
    opencv_suspicious_count = 0
    if 'opencv_manipulation' in results and results['opencv_manipulation'].get('enabled', False):
        opencv_suspicious = results['opencv_manipulation'].get('is_suspicious', False)
        opencv_score = results['opencv_manipulation'].get('overall_confidence', 0)

        if opencv_suspicious:
            forensic_score += forensic_weights['opencv'] * opencv_score
            confidence['ai_ml_methods']['opencv'] = opencv_score
            confidence['methods']['opencv'] = opencv_score

            evidence_parts = []
            if results['opencv_manipulation'].get('manipulation_detection', {}).get('is_manipulated'):
                manip_conf = results['opencv_manipulation']['manipulation_detection']['confidence']
                evidence_parts.append(f"Gaussian blur analysis: {manip_conf*100:.1f}%")
                opencv_suspicious_count += 1
            if results['opencv_manipulation'].get('noise_analysis', {}).get('is_noise_inconsistent'):
                noise_conf = results['opencv_manipulation']['noise_analysis'].get('confidence', 0)
                evidence_parts.append(f"Noise consistency: {noise_conf*100:.1f}%")
                opencv_suspicious_count += 1
            if results['opencv_manipulation'].get('jpeg_artifacts', {}).get('has_inconsistent_artifacts'):
                jpeg_conf = results['opencv_manipulation']['jpeg_artifacts']['confidence']
                evidence_parts.append(f"JPEG artifacts: {jpeg_conf*100:.1f}%")
                opencv_suspicious_count += 1

            evidence = "; ".join(evidence_parts) if evidence_parts else f"Overall suspicion: {opencv_score*100:.1f}%"
            confidence['indicators'].append({
                'method': 'OpenCV Manipulation (Computer Vision)',
                'evidence': evidence,
                'type': 'ai_ml'
            })

    # Store forensic manipulation score (0-100)
    confidence['confidence_score'] = round(min(forensic_score * 100.0, 100.0), 1)

    # ================================================================
    # PHASE 2: AI Generation Detection (separate from manipulation)
    # ================================================================
    # This answers: "Was this image created by an AI generator?"
    # Deterministic proof (metadata/filename) is an OVERRIDE, not a weight.
    # ML model is the primary probabilistic signal.
    # OpenCV can corroborate but not override.

    ai_probability = 0.0
    ai_deterministic_proof = False
    ai_detection_confidence = 'unknown'

    if 'ai_detection' in results and results['ai_detection'].get('enabled', False):
        ai_det = results['ai_detection']
        ai_pct = ai_det.get('ai_probability', 0)  # 0-100
        ai_detection_confidence = ai_det.get('confidence', 'unknown')

        # Check if we have deterministic proof (metadata/filename match)
        has_deterministic_layer = False
        if ai_det.get('detection_layers'):
            for layer in ai_det['detection_layers']:
                method = layer.get('method', '').lower()
                if method in ('metadata', 'filename') and layer.get('verdict') == 'AI':
                    has_deterministic_layer = True
                    break

        if has_deterministic_layer and ai_det.get('likely_ai', False):
            # Deterministic proof of AI generation.
            # This is near-certain — filename/metadata containing generator
            # names is planted by the tool itself. Confidence is high.
            ai_deterministic_proof = True
            ai_probability = max(ai_pct, 95.0)  # Floor at 95%
        else:
            # Probabilistic only — use ML model output directly
            ai_probability = ai_pct

        # Record indicator
        if ai_probability > 20:
            detection_method = "Unknown"
            if ai_det.get('detection_layers'):
                for layer in ai_det['detection_layers']:
                    if layer.get('verdict') == 'AI':
                        detection_method = layer.get('method', 'Unknown')
                        break

            confidence['indicators'].append({
                'method': f'AI Detection (ML: {detection_method})',
                'evidence': f"AI generation probability: {ai_probability:.1f}% ({ai_detection_confidence} confidence)",
                'type': 'ai_ml'
            })

    # OpenCV corroboration boost (only when ML model alone is uncertain)
    if opencv_suspicious_count >= 2 and ai_probability < 50.0:
        flagged_confs = []
        ocv = results.get('opencv_manipulation', {})
        if ocv.get('manipulation_detection', {}).get('is_manipulated'):
            flagged_confs.append(ocv['manipulation_detection']['confidence'])
        if ocv.get('noise_analysis', {}).get('is_noise_inconsistent'):
            flagged_confs.append(ocv['noise_analysis'].get('confidence', 0))
        if ocv.get('jpeg_artifacts', {}).get('has_inconsistent_artifacts'):
            flagged_confs.append(ocv['jpeg_artifacts']['confidence'])

        if flagged_confs:
            avg_conf = sum(flagged_confs) / len(flagged_confs)
            method_ratio = opencv_suspicious_count / 3.0
            opencv_ai_boost = avg_conf * method_ratio * 40.0
            ai_probability = max(ai_probability, round(opencv_ai_boost, 1))

    confidence['ai_generated_probability'] = round(ai_probability, 1)

    # ================================================================
    # PHASE 3: Final verdict — pick the strongest signal
    # ================================================================
    # Three possible verdicts:
    #   'ai_generated'  — AI detector fired with confidence
    #   'manipulated'   — forensic score above threshold
    #   'authentic'     — nothing significant found
    #
    # AI detection overrides the forensic score when confident, because
    # "this is AI-generated" is a more specific and actionable finding
    # than "this shows signs of manipulation."

    manipulation_threshold = 55.0
    ai_threshold = 50.0

    if ai_probability >= ai_threshold:
        # --- AI Generated verdict ---
        confidence['verdict'] = 'ai_generated'
        confidence['manipulation_detected'] = True  # legacy compat

        if ai_deterministic_proof:
            # Deterministic proof — very high certainty
            confidence['verdict_confidence'] = min(ai_probability, 99.0)
            confidence['verdict_certainty'] = 'high'
            confidence['verdict_label'] = 'AI-Generated Image'
        elif ai_probability >= 80:
            confidence['verdict_confidence'] = round(ai_probability, 1)
            confidence['verdict_certainty'] = 'high'
            confidence['verdict_label'] = 'Likely AI-Generated'
        elif ai_probability >= 65:
            confidence['verdict_confidence'] = round(ai_probability, 1)
            confidence['verdict_certainty'] = 'moderate'
            confidence['verdict_label'] = 'Possibly AI-Generated'
        else:
            confidence['verdict_confidence'] = round(ai_probability, 1)
            confidence['verdict_certainty'] = 'low'
            confidence['verdict_label'] = 'Suspected AI-Generated'

    elif confidence['confidence_score'] > manipulation_threshold:
        # --- Manipulation verdict ---
        confidence['verdict'] = 'manipulated'
        confidence['manipulation_detected'] = True

        score = confidence['confidence_score']
        distance = score - manipulation_threshold
        max_distance = 100.0 - manipulation_threshold
        confidence['verdict_confidence'] = round(min(distance / max_distance * 100.0, 100.0), 1)

        if confidence['verdict_confidence'] >= 80:
            confidence['verdict_certainty'] = 'high'
            confidence['verdict_label'] = 'Manipulation Detected'
        elif confidence['verdict_confidence'] >= 45:
            confidence['verdict_certainty'] = 'moderate'
            confidence['verdict_label'] = 'Likely Manipulated'
        else:
            confidence['verdict_certainty'] = 'low'
            confidence['verdict_label'] = 'Possibly Manipulated'

    else:
        # --- Authentic verdict ---
        confidence['verdict'] = 'authentic'
        confidence['manipulation_detected'] = False

        # Certainty is based on how far we are from both thresholds.
        # Take the closer threat (manipulation or AI) and measure distance.
        manip_distance = manipulation_threshold - confidence['confidence_score']
        ai_distance = ai_threshold - ai_probability

        # Use the closer threat to determine certainty
        closest_distance = min(manip_distance, ai_distance)
        closest_threshold = manipulation_threshold if manip_distance <= ai_distance else ai_threshold

        verdict_conf = min(closest_distance / closest_threshold * 100.0, 100.0)
        confidence['verdict_confidence'] = round(verdict_conf, 1)

        if verdict_conf >= 80:
            confidence['verdict_certainty'] = 'high'
            confidence['verdict_label'] = 'Image Appears Authentic'
        elif verdict_conf >= 45:
            confidence['verdict_certainty'] = 'moderate'
            confidence['verdict_label'] = 'Probably Authentic'
        elif verdict_conf >= 15:
            confidence['verdict_certainty'] = 'low'
            confidence['verdict_label'] = 'Possibly Authentic'
        else:
            confidence['verdict_certainty'] = 'inconclusive'
            confidence['verdict_label'] = 'Inconclusive'

    return confidence
