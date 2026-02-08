"""
Compliance Audit Detector - Trust-Building Approach

Philosophy: Start with ZERO TRUST and build confidence based on positive evidence.

This detector treats image authenticity like a security audit:
- Assume suspicious until proven otherwise
- Build trust incrementally through verifiable evidence
- Any HIGH-RISK finding = immediate failure
- Multiple MEDIUM-RISK findings = failure
- Need strong POSITIVE evidence to pass

This approach is resilient to:
- New AI generators (we don't need to know them)
- Evolving AI technology
- ML model drift/obsolescence
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import exif

from .base import BaseDetector, DetectionResult, DetectionMethod

logger = logging.getLogger(__name__)


# Risk levels and their thresholds
RISK_THRESHOLD_HIGH = 100  # Any HIGH risk finding = fail
RISK_THRESHOLD_MEDIUM = 60  # Multiple MEDIUM risks = fail
RISK_THRESHOLD_LOW = 30    # Many LOW risks = suspicious


class Finding:
    """Represents a single audit finding."""
    
    # Risk levels
    POSITIVE = "POSITIVE"  # Builds trust
    LOW = "LOW"           # Slightly suspicious
    MEDIUM = "MEDIUM"     # Moderately suspicious
    HIGH = "HIGH"         # Highly suspicious (likely AI/manipulated)
    
    def __init__(self, risk_level: str, category: str, description: str, score_impact: int):
        self.risk_level = risk_level
        self.category = category
        self.description = description
        self.score_impact = score_impact  # Negative = suspicious, Positive = trust


class ComplianceAuditor:
    """
    Compliance auditor - the gatekeeper for all detection decisions.
    
    This is NOT a detector. It's a separate component that:
    - Reviews detector results and decides when to stop
    - Performs final compliance audit of the image
    - Aggregates all findings into a single verdict
    
    Philosophy: Start with zero trust, build confidence through verified evidence.
    """
    
    name = "Compliance Audit"
    
    # Known legitimate camera manufacturers
    LEGITIMATE_CAMERAS = {
        'Google': ['Pixel'],
        'Apple': ['iPhone', 'iPad'],
        'Canon': True,  # All Canon cameras
        'Nikon': True,
        'Sony': True,
        'Samsung': True,
        'Fujifilm': True,
        'Olympus': True,
        'Panasonic': True,
        'Leica': True,
        'Hasselblad': True,
        'Phase One': True,
        'DJI': ['Mavic', 'Phantom', 'Inspire'],  # Drones
    }
    
    # Obsolete/suspicious software (discontinued or commonly faked)
    OBSOLETE_SOFTWARE = {
        'Picasa': {'discontinued': 2016, 'risk': 'HIGH'},
        'Windows Photo Gallery': {'discontinued': 2012, 'risk': 'HIGH'},
        'iPhoto': {'discontinued': 2015, 'risk': 'HIGH'},
        'Adobe Photoshop Album': {'discontinued': 2006, 'risk': 'HIGH'},
        'Photoshop Express': {'version_check': True, 'risk': 'MEDIUM'},
    }
    
    # AI generator indicators in filenames/metadata
    AI_INDICATORS = [
        'midjourney', 'dall-e', 'dalle', 'stable_diffusion', 'stable-diffusion', 'sd-',
        'gemini_generated', 'chatgpt', 'gpt-', 'ai-generated', 'ai_generated',
        'generated_image', 'synthetic', 'deepdream', 'artbreeder',
        'nightcafe', 'craiyon', 'lexica', 'playground-ai', 'firefly',
    ]
    
    # AI-typical dimensions (power-of-2 squares common in generators)
    AI_TYPICAL_DIMENSIONS = [
        (512, 512), (768, 768), (1024, 1024), 
        (1536, 1536), (2048, 2048),
    ]
    
    @classmethod
    def should_stop_early(cls, current_results: List[DetectionResult]) -> bool:
        """
        Centralized decision logic for early stopping.
        
        Determines if we have enough evidence to skip remaining detectors
        and save compute resources.
        
        Stop early if:
        1. HIGH confidence (>90) with clear AI indicators
        2. CERTAIN confidence (>95) with definitive evidence
        3. Clear authenticity evidence (confidence >95, no suspicious findings)
        
        Args:
            current_results: Results from detectors run so far
            
        Returns:
            True if analysis can stop early, False to continue
        """
        if not current_results:
            return False
        
        for result in current_results:
            # Check for definitive findings from individual detectors
            if result.confidence is None:
                continue
            
            confidence_value = result.confidence if isinstance(result.confidence, (int, float)) else 0
            
            # Definitive AI detection
            if confidence_value >= 90 and result.score and result.score >= 0.7:
                # High confidence AND high suspicion score
                detected = result.detected_types or []
                ai_indicators = ['ai_generation', 'ai_dimensions', 'ai_tool', 'metadata_stripped']
                if any(ind in detected for ind in ai_indicators):
                    logger.info(f"Early stop: HIGH confidence AI detection ({confidence_value}%)")
                    return True
            
            # Definitive authenticity
            if confidence_value >= 95 and result.score and result.score <= 0.3:
                # Very high confidence AND low suspicion
                logger.info(f"Early stop: CERTAIN authenticity ({confidence_value}%)")
                return True
        
        # Continue to next detector
        return False
    
    # Common real camera resolutions (aspect ratios)
    CAMERA_RESOLUTIONS = {
        '4:3': [(640, 480), (800, 600), (1024, 768), (1600, 1200), (2048, 1536), (4000, 3000)],
        '3:2': [(1500, 1000), (3000, 2000), (4288, 2848), (6000, 4000)],  # DSLR common
        '16:9': [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (4096, 2160)],  # Video/modern
        '1:1': [(1080, 1080),],  # Instagram, but also AI
    }
    
    name = "Compliance Audit"
    
    def __init__(self):
        pass
    
    def detect(self, image_path: str, original_filename: str = None, previous_results: List = None, forensics_data: dict = None) -> DetectionResult:
        """
        Run compliance audit on image.
        
        Args:
            image_path: Path to image file
            original_filename: Original filename for pattern matching
            previous_results: Results from previous detectors (metadata, SPAI, etc.)
            forensics_data: Results from forensics analyzers (noise, frequency, etc.)
            
        Returns:
            DetectionResult with trust score
        """
        findings: List[Finding] = []
        
        # Use original filename if provided, otherwise extract from path
        filename = original_filename or Path(image_path).name
        
        # Open image for analysis
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            return self._error_result(f"Could not open image: {e}")
        
        # === HIGH-RISK CHECKS (Any one = likely AI/manipulated) ===
        
        # Check 1: AI generator in filename/metadata
        findings.extend(self._check_ai_indicators(filename, image_path))
        
        # Check 2: Obsolete/suspicious software
        findings.extend(self._check_obsolete_software(image_path))
        
        # === MEDIUM-RISK CHECKS (Multiple = suspicious) ===
        
        # Check 3: AI-typical dimensions
        findings.extend(self._check_ai_dimensions(width, height))
        
        # Check 4: Suspicious EXIF patterns
        findings.extend(self._check_suspicious_exif(image_path))
        
        # === LOW-RISK CHECKS (Many = gradually suspicious) ===
        
        # Check 5: Missing metadata
        findings.extend(self._check_missing_metadata(image_path))
        
        # Check 6: Minimal EXIF
        findings.extend(self._check_minimal_exif(image_path))
        
        # === POSITIVE CHECKS (Build trust) ===
        
        # Check 7: Legitimate camera signature
        findings.extend(self._check_legitimate_camera(image_path))
        
        # Check 8: GPS data present
        findings.extend(self._check_gps_data(image_path))
        
        # Check 9: Realistic camera settings
        findings.extend(self._check_camera_settings(image_path))
        
        # Check 10: Common photo resolution
        findings.extend(self._check_photo_resolution(width, height))
        
        # === ML MODEL RESULTS (If available) ===
        
        # Check 11: SPAI model results
        if previous_results:
            findings.extend(self._check_ml_model_results(previous_results))
        
        # Check 12: Noise consistency analysis
        if forensics_data:
            findings.extend(self._check_noise_consistency(forensics_data))
        
        # === CALCULATE AUTHENTICITY SCORE ===
        
        authenticity_score = self._calculate_authenticity_score(findings)
        detected_types = self._collect_detector_types(findings)
        evidence = self._format_evidence(findings, authenticity_score)
        
        # Calculate component probabilities for display
        ai_probability = self._calculate_ai_probability(findings)
        manipulation_probability = self._calculate_manipulation_probability(findings)
        
        return DetectionResult(
            method=DetectionMethod.HEURISTIC,
            authenticity_score=authenticity_score,
            evidence=evidence,
            detected_types=detected_types,
            metadata={
                'ai_probability': ai_probability,  # 0-100: likelihood of AI generation
                'manipulation_probability': manipulation_probability,  # 0-100: likelihood of manipulation
                'findings_count': len(findings)
            }
        )
    
    def _check_ai_indicators(self, filename: str, image_path: str) -> List[Finding]:
        """Check for AI generator indicators in filename/metadata."""
        findings = []
        filename_lower = filename.lower()
        
        for indicator in self.AI_INDICATORS:
            if indicator in filename_lower:
                findings.append(Finding(
                    risk_level=Finding.HIGH,
                    category="AI Indicator",
                    description=f"Filename contains AI generator keyword: '{indicator}'",
                    score_impact=-100
                ))
                break  # One is enough
        
        return findings
    
    def _check_obsolete_software(self, image_path: str) -> List[Finding]:
        """Check for obsolete or suspicious software tags."""
        findings = []
        
        try:
            with open(image_path, 'rb') as f:
                img = exif.Image(f)
            
            if img.has_exif:
                software = getattr(img, 'software', None)
                if software:
                    for obs_software, info in self.OBSOLETE_SOFTWARE.items():
                        if obs_software.lower() in software.lower():
                            risk = info['risk']
                            impact = -100 if risk == 'HIGH' else -60
                            findings.append(Finding(
                                risk_level=risk,
                                category="Obsolete Software",
                                description=f"EXIF contains obsolete/suspicious software: '{software}' (discontinued {info.get('discontinued', 'date unknown')})",
                                score_impact=impact
                            ))
        except Exception:
            pass  # No EXIF or error reading
        
        return findings
    
    def _check_ai_dimensions(self, width: int, height: int) -> List[Finding]:
        """Check for AI-typical dimensions."""
        findings = []
        
        # Perfect square power-of-2 (VERY suspicious but easily defeated)
        if width == height and (width, height) in self.AI_TYPICAL_DIMENSIONS:
            findings.append(Finding(
                risk_level=Finding.MEDIUM,
                category="AI Dimensions",
                description=f"Perfect square power-of-2 dimensions: {width}x{height} (typical AI generation size)",
                score_impact=-60  # Reduced from -80, easy to defeat with crop
            ))
        
        return findings
    
    def _check_suspicious_exif(self, image_path: str) -> List[Finding]:
        """Check for suspicious EXIF patterns."""
        findings = []
        
        try:
            with open(image_path, 'rb') as f:
                img = exif.Image(f)
            
            if img.has_exif:
                # Check for suspicious unique IDs (all zeros pattern)
                unique_id = getattr(img, 'image_unique_id', None)
                if unique_id and unique_id.endswith('00000000000000'):
                    findings.append(Finding(
                        risk_level=Finding.MEDIUM,
                        category="Suspicious EXIF",
                        description=f"Suspicious ImageUniqueId pattern: {unique_id} (likely generated)",
                        score_impact=-30
                    ))
        except Exception:
            pass
        
        return findings
    
    def _check_missing_metadata(self, image_path: str) -> List[Finding]:
        """Check for completely missing metadata."""
        findings = []
        
        try:
            with open(image_path, 'rb') as f:
                img = exif.Image(f)
            
            if not img.has_exif:
                findings.append(Finding(
                    risk_level=Finding.LOW,
                    category="Missing Metadata",
                    description="No EXIF metadata found (common in AI-generated or heavily edited images)",
                    score_impact=-15  # Reduced from -25
                ))
        except Exception:
            findings.append(Finding(
                risk_level=Finding.LOW,
                category="Missing Metadata",
                description="No EXIF metadata found",
                score_impact=-15  # Reduced from -25
            ))
        
        return findings
    
    def _check_minimal_exif(self, image_path: str) -> List[Finding]:
        """Check for suspiciously minimal EXIF."""
        findings = []
        
        try:
            with open(image_path, 'rb') as f:
                img = exif.Image(f)
            
            if img.has_exif:
                # Count EXIF tags
                tag_count = len(img.list_all())
                
                # Real cameras have 50+ EXIF tags typically
                if tag_count < 10:
                    findings.append(Finding(
                        risk_level=Finding.LOW,
                        category="Minimal EXIF",
                        description=f"Suspiciously few EXIF tags: {tag_count} (real cameras have 50+)",
                        score_impact=-10  # Reduced from -20
                    ))
        except Exception:
            pass
        
        return findings
    
    def _check_legitimate_camera(self, image_path: str) -> List[Finding]:
        """Check for legitimate camera signature (POSITIVE finding)."""
        findings = []
        
        try:
            with open(image_path, 'rb') as f:
                img = exif.Image(f)
            
            if img.has_exif:
                make = getattr(img, 'make', None)
                model = getattr(img, 'model', None)
                
                if make and model:
                    make_clean = make.strip()
                    model_clean = model.strip()
                    
                    # Check against legitimate cameras
                    for manufacturer, models in self.LEGITIMATE_CAMERAS.items():
                        if manufacturer.lower() in make_clean.lower():
                            if models is True:
                                # All models from this manufacturer are legitimate
                                findings.append(Finding(
                                    risk_level=Finding.POSITIVE,
                                    category="Legitimate Camera",
                                    description=f"Verified camera signature: {make_clean} {model_clean}",
                                    score_impact=+50
                                ))
                                break
                            else:
                                # Check specific models
                                for valid_model in models:
                                    if valid_model.lower() in model_clean.lower():
                                        findings.append(Finding(
                                            risk_level=Finding.POSITIVE,
                                            category="Legitimate Camera",
                                            description=f"Verified camera signature: {make_clean} {model_clean}",
                                            score_impact=+50
                                        ))
                                        break
        except Exception:
            pass
        
        return findings
    
    def _check_gps_data(self, image_path: str) -> List[Finding]:
        """Check for GPS data (POSITIVE finding)."""
        findings = []
        
        try:
            with open(image_path, 'rb') as f:
                img = exif.Image(f)
            
            if img.has_exif:
                # Check for GPS tags
                has_gps = (hasattr(img, 'gps_latitude') or 
                          hasattr(img, 'gps_longitude') or
                          'gps_latitude' in img.list_all())
                
                if has_gps:
                    findings.append(Finding(
                        risk_level=Finding.POSITIVE,
                        category="GPS Data",
                        description="GPS location data present (rare in AI-generated images)",
                        score_impact=+30
                    ))
        except Exception:
            pass
        
        return findings
    
    def _check_camera_settings(self, image_path: str) -> List[Finding]:
        """Check for realistic camera settings (POSITIVE finding)."""
        findings = []
        
        try:
            with open(image_path, 'rb') as f:
                img = exif.Image(f)
            
            if img.has_exif:
                # Check for camera settings that AI generators typically don't fake
                has_iso = hasattr(img, 'photographic_sensitivity')
                has_exposure = hasattr(img, 'exposure_time')
                has_aperture = hasattr(img, 'f_number')
                has_focal_length = hasattr(img, 'focal_length')
                
                settings_count = sum([has_iso, has_exposure, has_aperture, has_focal_length])
                
                if settings_count >= 3:
                    findings.append(Finding(
                        risk_level=Finding.POSITIVE,
                        category="Camera Settings",
                        description=f"Realistic camera settings present ({settings_count}/4 key settings)",
                        score_impact=+25
                    ))
        except Exception:
            pass
        
        return findings
    
    def _check_photo_resolution(self, width: int, height: int) -> List[Finding]:
        """Check if resolution matches common photo sizes (POSITIVE finding)."""
        findings = []
        
        # Check all common aspect ratios
        for aspect_name, resolutions in self.CAMERA_RESOLUTIONS.items():
            if (width, height) in resolutions or (height, width) in resolutions:
                findings.append(Finding(
                    risk_level=Finding.POSITIVE,
                    category="Photo Resolution",
                    description=f"Standard camera resolution: {width}x{height} ({aspect_name} aspect ratio)",
                    score_impact=+20  # Increased from +15
                ))
                break
        
        # Check for natural aspect ratios (not perfect squares)
        if width > 0 and height > 0:
            aspect_ratio = width / height
            # Common photo ratios: 4:3 (1.33), 3:2 (1.5), 16:9 (1.78)
            if 1.2 < aspect_ratio < 2.0 or 0.5 < aspect_ratio < 0.83:
                # Not a perfect square, good sign
                if abs(aspect_ratio - 1.0) > 0.15:  # Not close to 1:1
                    findings.append(Finding(
                        risk_level=Finding.POSITIVE,
                        category="Natural Aspect Ratio",
                        description=f"Aspect ratio {aspect_ratio:.2f}:1 is typical for photos (not perfect square)",
                        score_impact=+10
                    ))
        
        return findings
    
    def _calculate_authenticity_score(self, findings: List[Finding]) -> int:
        """
        Calculate overall authenticity score (0-100).
        
        Pure fake-to-real scale:
        - 0: Definitely fake (high confidence, severe findings)
        - 50: Uncertain/unknown
        - 100: Definitely real (high confidence, strong evidence)
        
        Score does NOT indicate type of manipulation, only fake vs real.
        
        Note: score_impact values are designed as risk impacts:
        - Negative impact = suspicious = lowers authenticity
        - Positive impact = evidence of authenticity = raises authenticity
        """
        # Start at neutral (50 = uncertain)
        base_score = 50
        
        # Apply all finding impacts
        # Negative impacts (suspicious) LOWER the score
        # Positive impacts (authentic evidence) RAISE the score
        for finding in findings:
            base_score += finding.score_impact
        
        # Clamp to 0-100 and return as authenticity
        # After applying impacts: low score = suspicious, high score = authentic
        return max(0, min(100, base_score))
    
    def _collect_detector_types(self, findings: List[Finding]) -> List[str]:
        """Collect detector types that triggered for informational purposes."""
        detected_types = []
        
        # Check what types of detection triggered
        for finding in findings:
            if finding.risk_level in [Finding.HIGH, Finding.MEDIUM]:
                # Map findings to detector types
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
    
    def _calculate_ai_probability(self, findings: List[Finding]) -> float:
        """
        Calculate AI generation probability (0-100) based on AI-specific findings.
        
        Focuses on evidence that specifically indicates AI generation:
        - AI indicators in filename/metadata
        - Perfect square dimensions (common in AI)
        - Synthetic noise patterns
        - ML model detections
        """
        ai_score = 0.0
        max_possible = 0.0
        
        for finding in findings:
            # Weight findings by risk level and AI relevance
            if 'AI' in finding.category or 'Synthetic' in finding.category or 'ML Model' in finding.category:
                weight = {
                    Finding.HIGH: 40.0,
                    Finding.MEDIUM: 25.0,
                    Finding.LOW: 10.0,
                    Finding.POSITIVE: 0.0  # Positive findings reduce AI probability
                }.get(finding.risk_level, 0)
                
                max_possible += weight
                if finding.risk_level != Finding.POSITIVE:
                    ai_score += weight
            
            # Dimension findings contribute to AI probability
            elif 'dimension' in finding.description.lower() and finding.risk_level in [Finding.HIGH, Finding.MEDIUM]:
                weight = 30.0 if finding.risk_level == Finding.HIGH else 15.0
                max_possible += weight
                ai_score += weight
            
            # Very low noise is AI-like
            elif 'noise' in finding.category.lower() and 'uniform' in finding.description.lower():
                if finding.risk_level == Finding.HIGH:
                    weight = 35.0
                    max_possible += weight
                    ai_score += weight
        
        # If we found AI-relevant evidence, calculate probability
        if max_possible > 0:
            probability = (ai_score / max_possible) * 100
            return min(100.0, probability)
        
        # No AI-specific evidence found
        return 0.0
    
    def _calculate_manipulation_probability(self, findings: List[Finding]) -> float:
        """
        Calculate manipulation probability (0-100) based on forensic findings.
        
        Focuses on evidence of editing/tampering:
        - Noise inconsistencies
        - Frequency anomalies
        - Metadata stripping
        - Forensic method detections
        
        Note: POSITIVE findings (evidence of authenticity) REDUCE probability.
        """
        manip_score = 0.0
        max_possible = 0.0
        
        for finding in findings:
            # Weight findings by risk level and manipulation relevance
            if 'Forensic' in finding.category or 'Manipulation' in finding.category:
                if finding.risk_level == Finding.POSITIVE:
                    # Positive evidence reduces manipulation probability
                    continue
                    
                weight = {
                    Finding.HIGH: 40.0,
                    Finding.MEDIUM: 25.0,
                    Finding.LOW: 10.0,
                }.get(finding.risk_level, 0)
                
                max_possible += weight
                manip_score += weight
            
            # Noise analysis (excluding very uniform which is more AI-like)
            elif 'noise' in finding.category.lower() and 'uniform' not in finding.description.lower():
                if finding.risk_level == Finding.POSITIVE:
                    # High noise variance is GOOD (authentic) - skip
                    continue
                elif finding.risk_level in [Finding.HIGH, Finding.MEDIUM]:
                    weight = 30.0 if finding.risk_level == Finding.HIGH else 20.0
                    max_possible += weight
                    manip_score += weight
            
            # Frequency anomalies
            elif 'frequency' in finding.category.lower():
                if finding.risk_level in [Finding.HIGH, Finding.MEDIUM]:
                    weight = 25.0 if finding.risk_level == Finding.HIGH else 15.0
                    max_possible += weight
                    manip_score += weight
            
            # Metadata anomalies
            elif 'metadata' in finding.category.lower():
                if finding.risk_level in [Finding.MEDIUM, Finding.LOW]:
                    weight = 15.0 if finding.risk_level == Finding.MEDIUM else 5.0
                    max_possible += weight
                    manip_score += weight
        
        # If we found manipulation-relevant evidence, calculate probability
        if max_possible > 0:
            probability = (manip_score / max_possible) * 100
            return min(100.0, probability)
        
        # No manipulation-specific evidence found
        return 0.0
    
    def _format_evidence(self, findings: List[Finding], authenticity_score: int) -> str:
        """Format findings into human-readable evidence string."""
        if not findings:
            return f"No significant findings. Authenticity: {authenticity_score}/100"
        
        # Group by risk level
        by_risk = {
            Finding.HIGH: [],
            Finding.MEDIUM: [],
            Finding.LOW: [],
            Finding.POSITIVE: []
        }
        
        for finding in findings:
            by_risk[finding.risk_level].append(finding)
        
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
    
    def _check_ml_model_results(self, previous_results: List) -> List[Finding]:
        """
        Check ML model results (e.g., SPAI).
        
        Strategy:
        - If ML says AI → HIGH RISK (trust the negative)
        - If ML says Real → LOW POSITIVE (weak trust boost)
        
        Rationale: ML models can become outdated and miss new AI types,
        but when they DO detect AI, it's worth trusting.
        """
        findings = []
        
        for result in previous_results:
            method_name = result.get('method', '')
            
            # Look for ML model results (SPAI, etc.)
            if 'ml' in method_name.lower() or 'spai' in method_name.lower():
                is_ai = result.get('is_ai_generated')
                score = result.get('score', 0.0)
                confidence = result.get('confidence', 'NONE')
                
                # ML model says it's AI-generated
                if is_ai is True and score > 0.5:
                    findings.append(Finding(
                        risk_level=Finding.HIGH,
                        category="ML Model Detection",
                        description=f"ML model ({method_name}) detected AI generation: {score:.1%} probability",
                        score_impact=-100
                    ))
                
                # ML model says it's real with high confidence
                elif is_ai is False and score < 0.3 and confidence in ['HIGH', 'MEDIUM']:
                    findings.append(Finding(
                        risk_level=Finding.POSITIVE,
                        category="ML Model Assessment",
                        description=f"ML model ({method_name}) suggests authentic: {(1-score):.1%} confidence",
                        score_impact=+10  # Low trust boost - ML can be outdated
                    ))
                
                # ML model is uncertain or gave weak signal - ignore it
                # (Don't penalize for ML uncertainty - that's expected with new AI types)
        
        return findings
    
    def _check_noise_consistency(self, forensics_data: dict) -> List[Finding]:
        """
        Check noise analysis results from forensics plugin.
        
        AI images have unnaturally LOW inconsistency (uniform synthetic noise).
        Real photos have HIGHER inconsistency (natural sensor noise variation).
        
        This is the OPPOSITE of manipulation detection, where high inconsistency
        indicates spliced/pasted regions.
        """
        findings = []
        
        noise_data = forensics_data.get('noise_analysis', {})
        if not noise_data:
            return findings
        
        inconsistency = noise_data.get('inconsistency_score')
        mean_variance = noise_data.get('mean_variance')
        anomaly_count = noise_data.get('anomaly_count')
        
        if inconsistency is not None:
            # AI images: inconsistency < 4.0 (uniform noise)
            if inconsistency < 3.0:
                findings.append(Finding(
                    risk_level=Finding.HIGH,
                    category="Synthetic Noise Pattern",
                    description=f"Unnaturally uniform noise (inconsistency: {inconsistency:.2f}) - typical of AI generation",
                    score_impact=-70
                ))
            elif inconsistency < 4.2:
                findings.append(Finding(
                    risk_level=Finding.MEDIUM,
                    category="Suspicious Noise Pattern",
                    description=f"Low noise inconsistency ({inconsistency:.2f}) - may indicate synthetic origin",
                    score_impact=-40
                ))
            # Real photos: inconsistency > 5.0 (natural sensor noise)
            elif inconsistency > 5.5:
                findings.append(Finding(
                    risk_level=Finding.POSITIVE,
                    category="Natural Noise Pattern",
                    description=f"Natural noise variation ({inconsistency:.2f}) - consistent with real camera",
                    score_impact=+35
                ))
            elif inconsistency > 4.8:
                findings.append(Finding(
                    risk_level=Finding.POSITIVE,
                    category="Moderate Noise Variation",
                    description=f"Moderate noise variation ({inconsistency:.2f}) - likely authentic",
                    score_impact=+20
                ))
        
        # Check anomaly count (real photos have more noise outliers)
        if anomaly_count is not None:
            if anomaly_count > 500:
                findings.append(Finding(
                    risk_level=Finding.POSITIVE,
                    category="High Noise Variation",
                    description=f"Many noise anomalies ({anomaly_count}) - typical of real sensor",
                    score_impact=+15
                ))
            elif anomaly_count < 100:
                findings.append(Finding(
                    risk_level=Finding.LOW,
                    category="Low Noise Variation",
                    description=f"Few noise anomalies ({anomaly_count}) - unusually uniform",
                    score_impact=-10
                ))
        
        return findings
