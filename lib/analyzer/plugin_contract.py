# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Plugin Data Contract — Standard Interface for Auditor Integration.

All forensic plugins SHOULD report their findings in a standardized format
so the ComplianceAuditor can properly score them. This eliminates the need
for the auditor to know plugin-specific result structures.

Contract Format:
    results['<plugin_name>']['audit_findings'] = [
        {
            'level': 'LOW' | 'MEDIUM' | 'HIGH',
            'category': str,  # e.g., "JPEG Artifacts", "Noise Pattern"
            'description': str,  # Human-readable evidence
            'is_positive': bool,  # True = authentic, False = suspicious
            'confidence': float,  # 0.0-1.0, optional
        },
        ...
    ]

Guidelines:
    - Use HIGH sparingly — reserve for virtual certainty
    - category should be consistent across plugin invocations
    - description should be self-contained (auditor may display standalone)
    - is_positive=True means evidence SUPPORTS authenticity
    - confidence is optional; auditor uses level primarily

Example (from a hypothetical JPEG analysis plugin):
    {
        'level': 'MEDIUM',
        'category': 'JPEG Compression',
        'description': 'Double quantization detected (Q=85→75)',
        'is_positive': False,
        'confidence': 0.82,
    }

Plugins WITHOUT audit_findings:
    The auditor will use plugin-specific parsers (_check_noise_consistency,
    _check_frequency_analysis, etc.) as a fallback. This is less ideal but
    maintains backward compatibility.

Validation:
    Use validate_audit_findings() to check compliance before returning
    from plugin.run().
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Valid finding levels
VALID_LEVELS = {'LOW', 'MEDIUM', 'HIGH'}


def validate_audit_findings(findings: List[Dict[str, Any]], 
                            plugin_name: str = 'unknown') -> Tuple[bool, List[str]]:
    """
    Validate that audit_findings conform to the plugin contract.
    
    Args:
        findings: List of finding dictionaries
        plugin_name: Name of the plugin (for error messages)
        
    Returns:
        (is_valid, errors) tuple
    """
    if not isinstance(findings, list):
        return False, [f"{plugin_name}: audit_findings must be a list"]
    
    errors = []
    
    for i, finding in enumerate(findings):
        if not isinstance(finding, dict):
            errors.append(f"{plugin_name}[{i}]: Finding must be a dict")
            continue
        
        # Required fields
        if 'level' not in finding:
            errors.append(f"{plugin_name}[{i}]: Missing required field 'level'")
        elif finding['level'] not in VALID_LEVELS:
            errors.append(
                f"{plugin_name}[{i}]: Invalid level '{finding['level']}', "
                f"must be one of {VALID_LEVELS}"
            )
        
        if 'category' not in finding:
            errors.append(f"{plugin_name}[{i}]: Missing required field 'category'")
        elif not isinstance(finding['category'], str) or not finding['category']:
            errors.append(f"{plugin_name}[{i}]: category must be non-empty string")
        
        if 'description' not in finding:
            errors.append(f"{plugin_name}[{i}]: Missing required field 'description'")
        elif not isinstance(finding['description'], str):
            errors.append(f"{plugin_name}[{i}]: description must be string")
        
        if 'is_positive' not in finding:
            errors.append(f"{plugin_name}[{i}]: Missing required field 'is_positive'")
        elif not isinstance(finding['is_positive'], bool):
            errors.append(f"{plugin_name}[{i}]: is_positive must be boolean")
        
        # Optional fields
        if 'confidence' in finding:
            conf = finding['confidence']
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                errors.append(
                    f"{plugin_name}[{i}]: confidence must be float in [0.0, 1.0]"
                )
    
    return len(errors) == 0, errors


def create_finding(level: str, category: str, description: str,
                  is_positive: bool = False, confidence: Optional[float] = None) -> Dict[str, Any]:
    """
    Helper to create a properly formatted audit finding.
    
    Args:
        level: LOW, MEDIUM, or HIGH
        category: Finding category (e.g., "Noise Analysis")
        description: Human-readable evidence
        is_positive: True if evidence supports authenticity
        confidence: Optional 0.0-1.0 confidence score
        
    Returns:
        Finding dictionary
        
    Raises:
        ValueError: If parameters are invalid
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"level must be one of {VALID_LEVELS}, got: {level}")
    
    if not isinstance(category, str) or not category:
        raise ValueError("category must be non-empty string")
    
    if not isinstance(description, str):
        raise ValueError("description must be string")
    
    if not isinstance(is_positive, bool):
        raise ValueError("is_positive must be boolean")
    
    finding = {
        'level': level,
        'category': category,
        'description': description,
        'is_positive': is_positive,
    }
    
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be float in [0.0, 1.0]")
        finding['confidence'] = float(confidence)
    
    return finding


def get_audit_findings(results: dict, plugin_name: str) -> List[Dict[str, Any]]:
    """
    Extract audit_findings from plugin results, with validation.
    
    Args:
        results: Plugin results dictionary
        plugin_name: Name of the plugin
        
    Returns:
        List of validated findings (may be empty)
    """
    plugin_data = results.get(plugin_name, {})
    
    if not isinstance(plugin_data, dict):
        return []
    
    findings = plugin_data.get('audit_findings', [])
    
    if not findings:
        return []
    
    is_valid, errors = validate_audit_findings(findings, plugin_name)
    
    if not is_valid:
        logger.warning(
            f"Plugin '{plugin_name}' produced invalid audit_findings:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )
        return []
    
    return findings
