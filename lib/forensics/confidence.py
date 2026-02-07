# Ghiro - Copyright (C) 2013-2026 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Confidence scoring system for aggregating detection results.
"""


def calculate_manipulation_confidence(results):
    """
    Aggregate all detection methods into overall confidence score.
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        Dictionary with confidence scores and indicators
    """
    confidence = {
        'manipulation_detected': False,
        'confidence_score': 0.0,
        'ai_generated_probability': 0.0,
        'indicators': [],
        'methods': {}
    }
    
    # Weight different detection methods
    weights = {
        'ela_anomalies': 0.20,
        'noise_inconsistency': 0.25,
        'frequency_anomalies': 0.15,
        'ai_artifacts': 0.35,
        'metadata_issues': 0.10,
    }
    
    # ELA analysis
    if 'ela' in results and 'max_difference' in results['ela']:
        ela_score = min(results['ela']['max_difference'] / 100.0, 1.0)
        if ela_score > 0.3:
            confidence['confidence_score'] += weights['ela_anomalies'] * ela_score
            confidence['methods']['ela'] = ela_score
            confidence['indicators'].append({
                'method': 'ELA Analysis',
                'evidence': f"High error level detected (score: {ela_score:.2f})"
            })
    
    # Noise analysis
    if 'noise_analysis' in results and results['noise_analysis'].get('suspicious', False):
        noise_score = results['noise_analysis'].get('inconsistency_score', 0) / 100.0
        confidence['confidence_score'] += weights['noise_inconsistency'] * noise_score
        confidence['methods']['noise'] = noise_score
        confidence['indicators'].append({
            'method': 'Noise Analysis',
            'evidence': f"Inconsistent noise patterns detected ({results['noise_analysis']['inconsistency_score']:.1f}%)"
        })
    
    # Frequency analysis
    if 'frequency_analysis' in results and results['frequency_analysis'].get('suspicious', False):
        freq_score = results['frequency_analysis'].get('anomaly_score', 0) / 100.0
        confidence['confidence_score'] += weights['frequency_anomalies'] * freq_score
        confidence['methods']['frequency'] = freq_score
        confidence['indicators'].append({
            'method': 'Frequency Analysis',
            'evidence': "Suspicious frequency patterns detected"
        })
    
    # AI artifact detection
    if 'ai_detection' in results:
        ai_score = results['ai_detection'].get('ai_probability', 0) / 100.0
        confidence['ai_generated_probability'] = ai_score
        if ai_score > 0.5:
            confidence['confidence_score'] += weights['ai_artifacts'] * ai_score
            confidence['methods']['ai_detection'] = ai_score
            confidence['indicators'].append({
                'method': 'AI Detection',
                'evidence': f"AI generation probability: {ai_score*100:.1f}%"
            })
    
    # Metadata issues
    if 'metadata' in results:
        metadata_issues = 0
        if not results['metadata'].get('Exif', {}).get('Image', {}).get('Make'):
            metadata_issues += 1
        if not results['metadata'].get('Exif', {}).get('Image', {}).get('Model'):
            metadata_issues += 1
        
        if metadata_issues > 0:
            meta_score = metadata_issues / 2.0  # Normalize
            confidence['confidence_score'] += weights['metadata_issues'] * meta_score
            confidence['methods']['metadata'] = meta_score
            confidence['indicators'].append({
                'method': 'Metadata Analysis',
                'evidence': "Missing camera/device information"
            })
    
    # Determine if manipulation detected
    confidence['manipulation_detected'] = confidence['confidence_score'] > 0.50
    
    # Clamp score to 0-1 range
    confidence['confidence_score'] = min(confidence['confidence_score'], 1.0)
    
    return confidence
