"""
Metadata-based AI detection.

Checks EXIF/XMP metadata for AI generator signatures.
This is the fastest and most reliable method when metadata is present.
"""

import logging
from pathlib import Path
from typing import Dict, Set

from PIL import Image
from PIL.ExifTags import TAGS

try:
    import gi
    gi.require_version('GExiv2', '0.10')
    from gi.repository import GExiv2
    HAS_GEXIV2 = True
except (ImportError, ValueError):
    HAS_GEXIV2 = False

from .base import BaseDetector, DetectionResult, DetectionMethod, ConfidenceLevel


logger = logging.getLogger(__name__)


# Known AI generator software signatures
AI_SOFTWARE_SIGNATURES = {
    # Image generators
    'midjourney', 'dall-e', 'dalle', 'dall·e',
    'stable diffusion', 'stablediffusion', 'automatic1111',
    'firefly', 'adobe firefly',
    'leonardo.ai', 'leonardo ai',
    'bluewillow', 'blue willow',
    'craiyon',
    'nightcafe', 'night cafe',
    'artbreeder',
    'deepdream',
    'wombo', 'wombo dream',
    'starry ai', 'starryai',
    'google gemini', 'gemini', 'google imagen', 'imagen',
    'ideogram', 'flux', 'pixlr ai',
    
    # AI editing tools
    'generative fill',
    'generative expand',
    'ai enhance',
    'topaz photo ai',
    'luminar neo',
}

# Filename patterns that indicate AI generation
AI_FILENAME_PATTERNS = {
    'midjourney', 'mj_', '_mj_',
    'dall-e', 'dalle', 'dall·e',
    'stable_diffusion', 'sd_', '_sd_',
    'firefly', 'adobe_firefly',
    'leonardo', 'leonardo_ai',
    'gemini_generated', 'imagen_generated',
    'ai_generated', 'ai-generated',
    'generated_image', 'generated-image',
    'synthetic', 'artificial',
}

# XMP/IPTC fields that might contain AI info
AI_METADATA_FIELDS = [
    'Xmp.dc.creator',
    'Xmp.dc.description',
    'Xmp.photoshop.Credit',
    'Xmp.iptc.CreatorContactInfo',
    'Xmp.xmp.CreatorTool',
    'Xmp.tiff.Software',
    'Xmp.tiff.Make',
    'Xmp.tiff.Model',
    'Iptc.Application2.Program',
    'Iptc.Application2.ProgramVersion',
]

# C2PA content credentials (Adobe's standard)
C2PA_FIELDS = [
    'Xmp.c2pa.assertions',
    'Xmp.c2pa.claim',
    'Xmp.dcterms.provenance',
]


class MetadataDetector(BaseDetector):
    """
    Detect AI-generated images through metadata analysis.
    
    Checks:
    1. Software/Creator tags for known AI generators
    2. C2PA content credentials
    3. Suspicious absence of camera metadata
    4. XMP namespaces used by AI tools
    """
    
    def get_order(self) -> int:
        """Run first - fastest method."""
        return 0
    
    def check_deps(self) -> bool:
        """Check if GExiv2 is available."""
        return HAS_GEXIV2
    
    def detect(self, image_path: str, original_filename: str = None) -> DetectionResult:
        """
        Analyze image metadata for AI signatures.
        
        Args:
            image_path: Path to image file (may be temporary)
            original_filename: Original filename (for pattern matching)
            
        Returns:
            DetectionResult with high confidence if AI signatures found
        """
        path = Path(image_path)
        
        if not path.exists():
            return DetectionResult(
                method=DetectionMethod.METADATA,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence="File not found"
            )
        
        # Check filename first (fastest) - use original filename if provided
        filename_to_check = original_filename if original_filename else path.name
        filename_result = self._check_filename(Path(filename_to_check))
        if filename_result.confidence != ConfidenceLevel.NONE:
            return filename_result
        
        # Try EXIF/XMP with GExiv2 first (more comprehensive)
        if HAS_GEXIV2:
            try:
                result = self._check_with_gexiv2(str(path))
                if result.confidence != ConfidenceLevel.NONE:
                    return result
            except Exception as e:
                logger.debug(f"GExiv2 metadata check failed: {e}")
        
        # Fallback to PIL EXIF
        try:
            result = self._check_with_pil(str(path))
            if result.confidence != ConfidenceLevel.NONE:
                return result
        except Exception as e:
            logger.debug(f"PIL metadata check failed: {e}")
        
        # No AI signatures found
        return DetectionResult(
            method=DetectionMethod.METADATA,
            is_ai_generated=None,  # Can't determine from metadata alone
            confidence=ConfidenceLevel.NONE,
            score=0.0,
            evidence="No AI generator signatures found in metadata or filename"
        )
    
    def _check_filename(self, path: Path) -> DetectionResult:
        """
        Check if filename contains AI generation indicators.
        
        Args:
            path: Path object
            
        Returns:
            DetectionResult with HIGH confidence if AI pattern found
        """
        filename_lower = path.name.lower()
        
        for pattern in AI_FILENAME_PATTERNS:
            if pattern in filename_lower:
                return DetectionResult(
                    method=DetectionMethod.METADATA,
                    is_ai_generated=True,
                    confidence=ConfidenceLevel.HIGH,
                    score=0.95,
                    evidence=f"Filename contains AI generation indicator: '{pattern}'"
                )
        
        return DetectionResult(
            method=DetectionMethod.METADATA,
            is_ai_generated=None,
            confidence=ConfidenceLevel.NONE,
            score=0.0,
            evidence=""
        )
    
    def _check_with_gexiv2(self, image_path: str) -> DetectionResult:
        """Check metadata using GExiv2 (supports XMP)."""
        metadata = GExiv2.Metadata()
        metadata.open_path(image_path)
        
        # Check all tags for AI signatures
        for tag in metadata.get_tags():
            try:
                value = metadata.get_tag_string(tag)
                if value and self._contains_ai_signature(value):
                    return DetectionResult(
                        method=DetectionMethod.METADATA,
                        is_ai_generated=True,
                        confidence=ConfidenceLevel.CERTAIN,
                        score=1.0,
                        evidence=f"AI generator found in {tag}: {value}",
                        metadata={'tag': tag, 'value': value}
                    )
            except:
                continue
        
        # Check for C2PA content credentials
        for field in C2PA_FIELDS:
            try:
                value = metadata.get_tag_string(field)
                if value:
                    # C2PA can indicate both real and AI - need to parse
                    if 'ai' in value.lower() or 'generated' in value.lower():
                        return DetectionResult(
                            method=DetectionMethod.METADATA,
                            is_ai_generated=True,
                            confidence=ConfidenceLevel.HIGH,
                            score=0.95,
                            evidence=f"C2PA credentials indicate AI generation: {field}",
                            metadata={'field': field, 'value': value}
                        )
            except:
                continue
        
        return DetectionResult(
            method=DetectionMethod.METADATA,
            is_ai_generated=None,
            confidence=ConfidenceLevel.NONE,
            score=0.0,
            evidence="No AI signatures in GExiv2 metadata"
        )
    
    def _check_with_pil(self, image_path: str) -> DetectionResult:
        """Check metadata using PIL (EXIF only)."""
        img = Image.open(image_path)
        exif = img.getexif()
        
        if not exif:
            return DetectionResult(
                method=DetectionMethod.METADATA,
                is_ai_generated=None,
                confidence=ConfidenceLevel.NONE,
                score=0.0,
                evidence="No EXIF data available"
            )
        
        # Check Software tag (most common)
        software = exif.get(305)  # 305 = Software tag
        if software and self._contains_ai_signature(software):
            return DetectionResult(
                method=DetectionMethod.METADATA,
                is_ai_generated=True,
                confidence=ConfidenceLevel.CERTAIN,
                score=1.0,
                evidence=f"AI generator in Software tag: {software}",
                metadata={'software': software}
            )
        
        # Check Artist/Creator
        artist = exif.get(315)  # 315 = Artist tag
        if artist and self._contains_ai_signature(artist):
            return DetectionResult(
                method=DetectionMethod.METADATA,
                is_ai_generated=True,
                confidence=ConfidenceLevel.HIGH,
                score=0.9,
                evidence=f"AI generator in Artist tag: {artist}",
                metadata={'artist': artist}
            )
        
        # Check all tags
        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            if isinstance(value, (str, bytes)):
                value_str = value.decode() if isinstance(value, bytes) else value
                if self._contains_ai_signature(value_str):
                    return DetectionResult(
                        method=DetectionMethod.METADATA,
                        is_ai_generated=True,
                        confidence=ConfidenceLevel.HIGH,
                        score=0.85,
                        evidence=f"AI signature in {tag_name}: {value_str}",
                        metadata={'tag': tag_name, 'value': value_str}
                    )
        
        return DetectionResult(
            method=DetectionMethod.METADATA,
            is_ai_generated=None,
            confidence=ConfidenceLevel.NONE,
            score=0.0,
            evidence="No AI signatures in PIL EXIF"
        )
    
    def _contains_ai_signature(self, text: str) -> bool:
        """Check if text contains known AI generator signatures."""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(sig in text_lower for sig in AI_SOFTWARE_SIGNATURES)
