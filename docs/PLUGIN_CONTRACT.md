# Plugin Data Contract

## Overview

All forensic plugins SHOULD report their findings using a standardized format that the ComplianceAuditor can directly score. This eliminates the need for plugin-specific parsing logic and ensures consistent scoring across all detectors.

## Contract Format

Plugins should populate:

```python
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
```

## Field Specifications

### Required Fields

- **level**: `'LOW'`, `'MEDIUM'`, or `'HIGH'`
  - Use `HIGH` sparingly — reserve for virtual certainty
  - `MEDIUM` for strong but not definitive signals
  - `LOW` for weak or circumstantial evidence

- **category**: Short descriptive tag (e.g., `"Photoholmes Forgery Detection"`)
  - Should be consistent across plugin invocations
  - Used for grouping related findings

- **description**: Human-readable explanation of what was found
  - Should be self-contained (auditor may display standalone)
  - Include relevant numeric values (e.g., "85% forgery probability")

- **is_positive**: Boolean flag
  - `True` = evidence SUPPORTS authenticity
  - `False` = evidence suggests inauthenticity

### Optional Fields

- **confidence**: Float in range [0.0, 1.0]
  - Plugin's confidence in this specific finding
  - Auditor primarily uses `level`, but may factor this in

## Scoring Impact

The ComplianceAuditor uses these point values:

- `LOW` = ±5 points
- `MEDIUM` = ±15 points  
- `HIGH` = ±50 points

Positive findings add to the authenticity score (max 100), negative findings subtract (min 0).

## Example Implementation

```python
from lib.analyzer.plugin_contract import create_finding, validate_audit_findings

class MyForensicPlugin(BaseAnalyzerModule):
    def run(self, task):
        # ... do analysis ...
        
        findings = []
        
        if forgery_detected:
            findings.append(create_finding(
                level='MEDIUM',
                category='My Detector',
                description=f'Forgery probability: {score*100:.1f}%',
                is_positive=False,
                confidence=score
            ))
        
        # Validate before storing
        is_valid, errors = validate_audit_findings(findings, 'my_plugin')
        if is_valid:
            self.results['my_plugin']['audit_findings'] = findings
        else:
            logger.error(f"Invalid findings: {errors}")
        
        return self.results
```

## Helper Functions

### `create_finding()`

Helper to create properly formatted findings with validation:

```python
from lib.analyzer.plugin_contract import create_finding

finding = create_finding(
    level='HIGH',
    category='AI Detection',
    description='Stable Diffusion signature found in EXIF',
    is_positive=False,
    confidence=0.95
)
```

### `validate_audit_findings()`

Validate a list of findings before storing:

```python
from lib.analyzer.plugin_contract import validate_audit_findings

is_valid, errors = validate_audit_findings(findings, 'plugin_name')
if not is_valid:
    for error in errors:
        logger.error(error)
```

### `get_audit_findings()`

Extract and validate findings from results (used by auditor):

```python
from lib.analyzer.plugin_contract import get_audit_findings

findings = get_audit_findings(results, 'photoholmes')
# Returns validated list, or empty list if invalid/missing
```

## Backward Compatibility

Plugins that don't provide `audit_findings` continue to work via legacy parsers:
- `_check_noise_consistency()`
- `_check_frequency_analysis()`
- `_check_opencv_findings()`
- etc.

However, new plugins SHOULD use the contract for cleaner integration.

## Testing

Unit tests enforce contract compliance:

```bash
python -m unittest tests.test_plugin_contract
```

Key tests:
- Validation of required fields
- Type checking
- Level value validation
- Confidence range validation
- Photoholmes contract compliance

## Migration Guide

To migrate existing plugins:

1. Import the contract helpers:
   ```python
   from lib.analyzer.plugin_contract import create_finding, validate_audit_findings
   ```

2. Create findings list in your `run()` method:
   ```python
   findings = []
   # ... populate findings ...
   ```

3. Validate and store:
   ```python
   is_valid, errors = validate_audit_findings(findings, 'my_plugin')
   if is_valid:
       self.results['my_plugin']['audit_findings'] = findings
   ```

4. Test:
   ```python
   from lib.analyzer.plugin_contract import get_audit_findings
   
   findings = get_audit_findings(mock_results, 'my_plugin')
   assert len(findings) > 0
   ```

## Current Implementations

### Photoholmes

Fully implements the contract:
- Per-method findings for significant detections
- Consensus findings when ≥3 methods run
- Validation before storage

See: `plugins/ai_ml/photoholmes_detection.py`

### Planned Migrations

Candidates for migration to the contract:
- ELA analysis
- Noise analysis (partially uses legacy parser)
- Frequency analysis (partially uses legacy parser)
- OpenCV manipulation (partially uses legacy parser)
- Signature detection

## Benefits

1. **Simplified Auditor**: No plugin-specific parsers needed
2. **Consistent Scoring**: All findings scored uniformly
3. **Validation**: Contract violations caught early
4. **Testability**: Easy to unit test plugin output
5. **Documentation**: Self-documenting finding structure
6. **Extensibility**: New plugins integrate automatically

## See Also

- `lib/analyzer/plugin_contract.py` - Contract implementation
- `lib/analyzer/auditor.py` - Auditor that consumes findings
- `tests/test_plugin_contract.py` - Contract tests
- `plugins/ai_ml/photoholmes_detection.py` - Reference implementation
