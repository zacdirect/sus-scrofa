# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Modern metadata extraction plugin.

Uses pure Python libraries for wider format support:
- exif: Pure Python EXIF extraction (JPEG, TIFF, WebP, HEIF)
- pillow-heif: HEIF/HEIC support with metadata
- Pillow: Fallback for basic formats
"""

import logging
from io import BytesIO
from pathlib import Path

from PIL import Image

from lib.db import save_file
from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import to_unicode, add_metadata_description, AutoVivification, str2image, image2str

# Try importing modern metadata libraries
try:
    import exif
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False

try:
    from libxmp import XMPFiles, consts
    HAS_XMP = True
except ImportError:
    HAS_XMP = False

logger = logging.getLogger(__name__)


class MetadataModernAnalyzer(BaseAnalyzerModule):
    """Extracts image metadata using modern pure-Python libraries."""

    order = 10  # Same as old gexiv plugin

    def check_deps(self):
        """Check if required libraries are available."""
        if not HAS_EXIF:
            logger.warning("'exif' library not installed. Install with: pip install exif")
            return False
        if not HAS_XMP:
            logger.warning("'python-xmp-toolkit' library not installed. XMP metadata will be unavailable. Install with: pip install python-xmp-toolkit")
        return True

    def _extract_exif_tag(self, img, tag_name, default=None):
        """Safely extract an EXIF tag."""
        try:
            return getattr(img, tag_name, default)
        except (AttributeError, KeyError):
            return default

    def _get_dimensions(self, pil_image):
        """Extract image dimensions."""
        try:
            width, height = pil_image.size
            self.results["metadata"]["dimensions"] = [width, height]
        except Exception as e:
            logger.debug(f"Could not extract dimensions: {e}")

    def _get_basic_info(self, exif_img):
        """Extract basic EXIF info."""
        try:
            # Camera make and model
            make = self._extract_exif_tag(exif_img, 'make')
            model = self._extract_exif_tag(exif_img, 'model')
            
            if make or model:
                if 'Exif' not in self.results["metadata"]:
                    self.results["metadata"]["Exif"] = {}
                if 'Image' not in self.results["metadata"]["Exif"]:
                    self.results["metadata"]["Exif"]["Image"] = {}
                
                if make:
                    self.results["metadata"]["Exif"]["Image"]["Make"] = to_unicode(make)
                    add_metadata_description("Exif.Image.Make", 
                        "The manufacturer of the recording equipment")
                if model:
                    self.results["metadata"]["Exif"]["Image"]["Model"] = to_unicode(model)
                    add_metadata_description("Exif.Image.Model", 
                        "The model name or model number of the equipment")
        except Exception as e:
            logger.debug(f"Error extracting basic info: {e}")

    def _get_datetime_info(self, exif_img):
        """Extract date/time information."""
        try:
            datetime_original = self._extract_exif_tag(exif_img, 'datetime_original')
            datetime = self._extract_exif_tag(exif_img, 'datetime')
            
            if datetime_original or datetime:
                if 'Exif' not in self.results["metadata"]:
                    self.results["metadata"]["Exif"] = {}
                
                if datetime_original:
                    if 'Photo' not in self.results["metadata"]["Exif"]:
                        self.results["metadata"]["Exif"]["Photo"] = {}
                    self.results["metadata"]["Exif"]["Photo"]["DateTimeOriginal"] = to_unicode(datetime_original)
                    add_metadata_description("Exif.Photo.DateTimeOriginal", 
                        "The date and time when the original image data was generated")
                
                if datetime:
                    if 'Image' not in self.results["metadata"]["Exif"]:
                        self.results["metadata"]["Exif"]["Image"] = {}
                    self.results["metadata"]["Exif"]["Image"]["DateTime"] = to_unicode(datetime)
                    add_metadata_description("Exif.Image.DateTime", 
                        "The date and time of image creation")
        except Exception as e:
            logger.debug(f"Error extracting datetime: {e}")

    def _get_gps_data(self, exif_img):
        """Extract GPS coordinates."""
        try:
            has_gps = (hasattr(exif_img, 'gps_latitude') and 
                      hasattr(exif_img, 'gps_longitude') and
                      exif_img.gps_latitude and 
                      exif_img.gps_longitude)
            
            if has_gps:
                lat = exif_img.gps_latitude
                lon = exif_img.gps_longitude
                
                # Convert to decimal degrees
                lat_decimal = self._dms_to_decimal(lat, exif_img.gps_latitude_ref)
                lon_decimal = self._dms_to_decimal(lon, exif_img.gps_longitude_ref)
                
                altitude = 0.0
                if hasattr(exif_img, 'gps_altitude') and exif_img.gps_altitude:
                    altitude = float(exif_img.gps_altitude)
                
                self.results["metadata"]["gps"] = {
                    "pos": {
                        "Longitude": lon_decimal,
                        "Latitude": lat_decimal
                    },
                    "Altitude": altitude
                }
                
                # Also add to Exif structure for compatibility
                if 'Exif' not in self.results["metadata"]:
                    self.results["metadata"]["Exif"] = {}
                if 'GPSInfo' not in self.results["metadata"]["Exif"]:
                    self.results["metadata"]["Exif"]["GPSInfo"] = {}
                
                self.results["metadata"]["Exif"]["GPSInfo"]["GPSLatitude"] = str(lat)
                self.results["metadata"]["Exif"]["GPSInfo"]["GPSLongitude"] = str(lon)
                add_metadata_description("Exif.GPSInfo.GPSLatitude", "GPS latitude")
                add_metadata_description("Exif.GPSInfo.GPSLongitude", "GPS longitude")
                
        except Exception as e:
            logger.debug(f"Error extracting GPS data: {e}")

    def _dms_to_decimal(self, dms, ref):
        """Convert degrees, minutes, seconds to decimal degrees."""
        try:
            if isinstance(dms, (tuple, list)) and len(dms) >= 3:
                degrees = float(dms[0])
                minutes = float(dms[1])
                seconds = float(dms[2])
                decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                if ref in ['S', 'W']:
                    decimal = -decimal
                return decimal
            return float(dms)
        except Exception:
            return 0.0

    def _get_camera_settings(self, exif_img):
        """Extract camera settings (ISO, aperture, shutter speed, etc.)."""
        try:
            if 'Exif' not in self.results["metadata"]:
                self.results["metadata"]["Exif"] = {}
            if 'Photo' not in self.results["metadata"]["Exif"]:
                self.results["metadata"]["Exif"]["Photo"] = {}
            
            photo = self.results["metadata"]["Exif"]["Photo"]
            
            # ISO
            iso = self._extract_exif_tag(exif_img, 'photographic_sensitivity')
            if iso:
                photo["ISOSpeedRatings"] = str(iso)
                add_metadata_description("Exif.Photo.ISOSpeedRatings", 
                    "ISO speed rating")
            
            # Aperture (F-number)
            fnumber = self._extract_exif_tag(exif_img, 'f_number')
            if fnumber:
                photo["FNumber"] = str(fnumber)
                add_metadata_description("Exif.Photo.FNumber", 
                    "The F number (aperture)")
            
            # Exposure time
            exposure = self._extract_exif_tag(exif_img, 'exposure_time')
            if exposure:
                photo["ExposureTime"] = str(exposure)
                add_metadata_description("Exif.Photo.ExposureTime", 
                    "Exposure time in seconds")
            
            # Focal length
            focal = self._extract_exif_tag(exif_img, 'focal_length')
            if focal:
                photo["FocalLength"] = str(focal)
                add_metadata_description("Exif.Photo.FocalLength", 
                    "Focal length of the lens in mm")
            
            # Flash
            flash = self._extract_exif_tag(exif_img, 'flash')
            if flash is not None:
                photo["Flash"] = str(flash)
                add_metadata_description("Exif.Photo.Flash", 
                    "Flash status")
                
        except Exception as e:
            logger.debug(f"Error extracting camera settings: {e}")

    def _get_software_info(self, exif_img):
        """Extract software/editing information - crucial for AI detection."""
        try:
            software = self._extract_exif_tag(exif_img, 'software')
            user_comment = self._extract_exif_tag(exif_img, 'user_comment')
            image_description = self._extract_exif_tag(exif_img, 'image_description')
            
            if software:
                if 'Exif' not in self.results["metadata"]:
                    self.results["metadata"]["Exif"] = {}
                if 'Image' not in self.results["metadata"]["Exif"]:
                    self.results["metadata"]["Exif"]["Image"] = {}
                
                self.results["metadata"]["Exif"]["Image"]["Software"] = to_unicode(software)
                add_metadata_description("Exif.Image.Software", 
                    "The name and version of the software used to create the image")
            
            if user_comment:
                if 'Exif' not in self.results["metadata"]:
                    self.results["metadata"]["Exif"] = {}
                if 'Photo' not in self.results["metadata"]["Exif"]:
                    self.results["metadata"]["Exif"]["Photo"] = {}
                    
                self.results["metadata"]["Exif"]["Photo"]["UserComment"] = to_unicode(user_comment)
                add_metadata_description("Exif.Photo.UserComment", 
                    "User comment")
            
            if image_description:
                if 'Exif' not in self.results["metadata"]:
                    self.results["metadata"]["Exif"] = {}
                if 'Image' not in self.results["metadata"]["Exif"]:
                    self.results["metadata"]["Exif"]["Image"] = {}
                    
                self.results["metadata"]["Exif"]["Image"]["ImageDescription"] = to_unicode(image_description)
                add_metadata_description("Exif.Image.ImageDescription", 
                    "A character string giving the title of the image")
                
        except Exception as e:
            logger.debug(f"Error extracting software info: {e}")

    def _extract_all_exif_tags(self, exif_img):
        """Extract all available EXIF tags."""
        try:
            # Get list of all available tags
            for tag in dir(exif_img):
                if tag.startswith('_') or tag in ['get', 'has_exif', 'delete', 'delete_all']:
                    continue
                
                try:
                    value = getattr(exif_img, tag)
                    if value is not None and value != '':
                        # Map tag to EXIF structure
                        self._add_tag_to_structure(tag, value)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Error extracting all EXIF tags: {e}")

    def _add_tag_to_structure(self, tag, value):
        """Add a tag to the appropriate metadata structure."""
        try:
            # Map common tags to Exif.Image or Exif.Photo
            if 'Exif' not in self.results["metadata"]:
                self.results["metadata"]["Exif"] = {}
            
            # Determine group based on tag name
            if tag.startswith('gps_'):
                group = 'GPSInfo'
            elif tag in ['make', 'model', 'orientation', 'software', 
                        'datetime', 'artist', 'copyright', 'image_description']:
                group = 'Image'
            else:
                group = 'Photo'
            
            if group not in self.results["metadata"]["Exif"]:
                self.results["metadata"]["Exif"][group] = {}
            
            # Convert tag name to CamelCase
            tag_name = ''.join(word.capitalize() for word in tag.split('_'))
            self.results["metadata"]["Exif"][group][tag_name] = to_unicode(str(value))
            
        except Exception as e:
            logger.debug(f"Error adding tag {tag}: {e}")

    def _extract_pillow_info(self, pil_image):
        """Extract metadata using Pillow's info dict (fallback/supplement)."""
        try:
            info = pil_image.info
            
            # Check for common keys
            if 'comment' in info and not self.results["metadata"].get("comment"):
                self.results["metadata"]["comment"] = to_unicode(info['comment'])
            
            # Extract any text chunks (PNG)
            for key, value in info.items():
                if isinstance(value, (str, bytes)):
                    try:
                        text_value = value.decode('utf-8') if isinstance(value, bytes) else value
                        if key not in ['exif', 'icc_profile']:  # Skip binary data
                            if 'pillow_info' not in self.results["metadata"]:
                                self.results["metadata"]["pillow_info"] = {}
                            self.results["metadata"]["pillow_info"][key] = to_unicode(text_value)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Error extracting Pillow info: {e}")

    def _extract_heif_metadata(self, file_data):
        """Extract metadata from HEIF/HEIC files."""
        if not HAS_HEIF:
            return
        
        try:
            heif_file = pillow_heif.open_heif(BytesIO(file_data))
            
            # Extract EXIF from HEIF
            if hasattr(heif_file, 'info') and 'exif' in heif_file.info:
                exif_data = heif_file.info['exif']
                if exif_data:
                    # Parse EXIF from HEIF
                    exif_img = exif.Image(exif_data)
                    if exif_img.has_exif:
                        self._get_basic_info(exif_img)
                        self._get_datetime_info(exif_img)
                        self._get_gps_data(exif_img)
                        self._get_camera_settings(exif_img)
                        self._get_software_info(exif_img)
            
            # HEIF specific metadata
            if hasattr(heif_file, 'info'):
                if 'metadata' in heif_file.info:
                    self.results["metadata"]["heif_info"] = {
                        "has_metadata": True
                    }
                
        except Exception as e:
            logger.debug(f"Error extracting HEIF metadata: {e}")

    def _extract_xmp_metadata(self, file_path):
        """Extract XMP metadata from image file."""
        if not HAS_XMP:
            return
        
        try:
            xmpfile = XMPFiles(file_path=file_path, open_forupdate=False)
            xmp = xmpfile.get_xmp()
            
            if xmp:
                # Initialize XMP section
                if 'XMP' not in self.results["metadata"]:
                    self.results["metadata"]["XMP"] = {}
                
                # Extract common XMP namespaces
                # Using only the namespaces that exist in python-xmp-toolkit
                namespaces = [
                    (consts.XMP_NS_DC, "DublinCore"),      # Dublin Core
                    (consts.XMP_NS_EXIF, "EXIF"),          # EXIF in XMP
                    (consts.XMP_NS_TIFF, "TIFF"),          # TIFF in XMP
                    (consts.XMP_NS_Photoshop, "Photoshop"), # Photoshop
                    (consts.XMP_NS_XMP_Rights, "Rights"),   # Rights management
                ]
                
                for ns_uri, ns_name in namespaces:
                    try:
                        # Iterate through all properties in this namespace
                        iterator = xmp.iterator(ns_uri)
                        # Check if iterator is not None and is iterable
                        if iterator is not None:
                            try:
                                for prop in iterator:
                                    if prop and len(prop) >= 2:
                                        prop_path = prop[0]
                                        prop_value = prop[1]
                                        
                                        if prop_value:
                                            if ns_name not in self.results["metadata"]["XMP"]:
                                                self.results["metadata"]["XMP"][ns_name] = {}
                                            
                                            # Extract property name from path
                                            prop_name = prop_path.split('/')[-1] if '/' in prop_path else prop_path
                                            self.results["metadata"]["XMP"][ns_name][prop_name] = to_unicode(str(prop_value))
                            except TypeError:
                                # Iterator returned None or is not iterable
                                pass
                                
                    except Exception as e:
                        logger.debug(f"Error extracting XMP namespace {ns_name}: {e}")
                
                # Only log if we actually found XMP data
                xmp_ns_count = len(self.results['metadata']['XMP'])
                if xmp_ns_count > 0:
                    logger.debug(f"Extracted XMP metadata with {xmp_ns_count} namespaces")
                
            xmpfile.close_file()
            
        except Exception as e:
            logger.debug(f"Could not extract XMP metadata: {e}")

    def run(self, task):
        """Main extraction routine."""
        file_data = task.get_file_data
        
        # Try to detect format
        try:
            pil_image = Image.open(BytesIO(file_data))
            image_format = pil_image.format
            
            # Get dimensions
            self._get_dimensions(pil_image)
            
            # Extract Pillow info
            self._extract_pillow_info(pil_image)
            
        except Exception as e:
            logger.warning(f"Could not open image with Pillow: {e}")
            image_format = None
            pil_image = None
        
        # Try EXIF extraction with exif library
        if HAS_EXIF:
            try:
                exif_img = exif.Image(file_data)
                
                if exif_img.has_exif:
                    logger.debug("EXIF data found, extracting...")
                    self._get_basic_info(exif_img)
                    self._get_datetime_info(exif_img)
                    self._get_gps_data(exif_img)
                    self._get_camera_settings(exif_img)
                    self._get_software_info(exif_img)
                    self._extract_all_exif_tags(exif_img)
                else:
                    logger.debug("No EXIF data found in image")
                    
            except Exception as e:
                # Some images have corrupted or non-standard EXIF data
                if 'TiffByteOrder' in str(e) or 'unpack' in str(e):
                    logger.debug(f"Image has corrupted or non-standard EXIF data")
                else:
                    logger.debug(f"Could not extract EXIF: {e}")
        
        # Try HEIF-specific extraction
        if image_format in ['HEIF', 'HEIC'] or (image_format is None and HAS_HEIF):
            try:
                self._extract_heif_metadata(file_data)
            except Exception as e:
                logger.debug(f"HEIF extraction failed: {e}")
        
        # Try XMP extraction (requires file path)
        if HAS_XMP:
            temp_file_path = None
            try:
                # Save to temporary file for XMP extraction
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
                    temp_file.write(file_data)
                    temp_file_path = temp_file.name
                
                self._extract_xmp_metadata(temp_file_path)
                
            except Exception as e:
                logger.debug(f"XMP extraction failed: {e}")
            finally:
                # Clean up temp file
                if temp_file_path:
                    try:
                        import os
                        os.unlink(temp_file_path)
                    except Exception:
                        pass
        
        # Log what we found
        has_data = False
        if self.results["metadata"]:
            if 'Exif' in self.results["metadata"] and self.results["metadata"]["Exif"]:
                exif_keys = sum(len(group) for group in self.results["metadata"]["Exif"].values() 
                              if isinstance(group, dict))
                if exif_keys > 0:
                    logger.info(f"Found {exif_keys} EXIF tags")
                    has_data = True
            if 'XMP' in self.results["metadata"] and self.results["metadata"]["XMP"]:
                xmp_namespaces = len(self.results["metadata"]["XMP"])
                if xmp_namespaces > 0:
                    logger.info(f"Found XMP metadata with {xmp_namespaces} namespaces")
                    has_data = True
        
        if has_data:
            logger.info(f"Extracted metadata from {image_format or 'unknown'} format")
        else:
            logger.info(f"No metadata found in {image_format or 'unknown'} image (may be AI-generated or edited)")
        
        return self.results
