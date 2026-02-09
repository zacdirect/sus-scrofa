# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from io import BytesIO, StringIO
import magic
import tempfile
import logging
import logging.handlers

from PIL import Image

from lib.db import save_file, get_file


try:
    import chardet
    IS_CHARDET = True
except ImportError:
    IS_CHARDET = False


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def _convert_to_dict(self, d):
        if isinstance(d, AutoVivification):
            return dict((k, self._convert_to_dict(v)) for k, v in d.items())
        return d

    def to_dict(self):
        return self._convert_to_dict(self)

def to_unicode(data):
    """Attempt to fix non utf-8 string into utf-8. It tries to guess input encoding,
    if fail retry with a replace strategy (so undetectable chars will be escaped).
    @see: fuller list of encodings at http://docs.python.org/library/codecs.html#standard-encodings
    """

    def brute_enc(data):
        """Trying to decode via simple brute forcing."""
        result = None
        encodings = ("ascii", "utf8", "latin1")
        for enc in encodings:
            if result:
                break
            try:
                result = data.decode(enc)
            except (UnicodeDecodeError, AttributeError):
                pass
        return result

    def chardet_enc(data):
        """Guess encoding via chardet."""
        result = None
        enc = chardet.detect(data)["encoding"]

        try:
            result = data.decode(enc)
        except (UnicodeDecodeError, AttributeError):
            pass

        return result

    # If already a string (unicode in Python 3), return it
    if isinstance(data, str):
        return data
    
    # If numeric types, convert to string
    if isinstance(data, (int, float)):
        return str(data)
    
    # If bytes, try to decode
    if isinstance(data, bytes):
        # First try to decode against a little set of common encodings.
        result = brute_enc(data)

        # Try via chardet.
        if not result and IS_CHARDET:
            result = chardet_enc(data)

        # If not possible to convert the input string, try again with
        # a replace strategy.
        if not result:
            result = data.decode("utf-8", errors="replace")

        return result
    
    # For other types, convert to string
    return str(data)

def str2file(text_data):
    strIO = BytesIO()
    strIO.write(text_data)
    strIO.seek(0)
    return strIO

def str2temp_file(text_data, suffix=None):
    tmp = tempfile.NamedTemporaryFile(prefix="sus_scrofa-", suffix=suffix, delete=False)
    tmp.write(text_data)
    return tmp

def add_metadata_description(key, description):
    """Adds key metadata description to lookup table.
    @param key: fully qualified metadata key
    @param description: key description
    """
    # Lazy import to avoid circular dependency (analyses.models -> lib.utils -> analyses.models)
    from analyses.models import AnalysisMetadataDescription

    # Skip if no description is provided.
    if description:
        try:
            AnalysisMetadataDescription.objects.get(key=key.lower())
        except AnalysisMetadataDescription.DoesNotExist:
            obj = AnalysisMetadataDescription(key=key.lower(), description=description)
            obj.save()

def str2image(data):
    """Converts binary data to PIL Image object.
    @param data: binarydata
    @return: PIL Image object
    """
    output = BytesIO()
    output.write(data)
    output.seek(0)
    return Image.open(output)

def image2str(img):
    """Converts PIL Image object to binary data.
    @param img: PIL Image object
    @return:  binary data
    """
    f = BytesIO()
    img.save(f, "JPEG")
    return f.getvalue()

def create_thumb(file_path):
    """Create thumbnail
    @param file_path: file path
    @return: GridFS ID
    """
    try:
        thumb = Image.open(file_path)
        thumb.thumbnail([200, 150], Image.Resampling.LANCZOS)
        return save_file(data=image2str(thumb), content_type="image/jpeg")
    except:
        return None

def hexdump(image_id, length=8):
    """Hexdump representation.
    @param image_id: gridfs image id
    @return: hexdump
    @see: code inspired to http://code.activestate.com/recipes/142812/
    """

    # Get image from gridfs.
    try:
       file = get_file(image_id)
    except (InvalidId, TypeError):
        return  None

    # Read data.
    src = file.read()

    hex_dump = []

    # Deal with unicode (Python 3: str is unicode).
    if isinstance(src, str):
        digits = 4
        # Convert string to bytes for consistent handling
        src = src.encode('latin-1', errors='replace')
    else:
        digits = 2

    # Create hex view.
    for i in range(0, len(src), length):
        line = {}
        s = src[i:i+length]
        # In Python 3, iterating bytes yields integers, not characters
        hexa = b" ".join([("%0*X" % (digits, x)).encode('ascii') for x in s])
        text = b"".join([bytes([x]) if 0x20 <= x < 0x7F else b"." for x in s])
        line["address"] = ("%04X" % i).encode('ascii')
        line["hex"] = ("%-*s" % (length*(digits + 1), hexa.decode('ascii'))).encode('ascii')
        line["text"] = text
        hex_dump.append(line)
    return hex_dump

def import_is_available(module_name):
    """Checks if a module is available.
    @param module_name: module name
    @return: import status
    """
    try:
        __import__(module_name, globals(), locals(), ["dummy"], 0)
        return True
    except ImportError:
        return False

def deps_check():
    """Check for all dependencies."""
    deps = [{"name": "Django", "module": "django"},
            {"name": "exif (metadata)", "module": "exif"},
            {"name": "pillow-heif (HEIF support)", "module": "pillow_heif"},
            {"name": "python-xmp-toolkit (XMP metadata)", "module": "libxmp"},
            {"name": "Pillow", "module": "PIL"},
            {"name": "Pdfkit", "module": "pdfkit"},
            {"name": "Pymongo", "module": "pymongo"},
            {"name": "Chardet", "module": "chardet"}
            ]

    for dep in deps:
        dep["available"] = import_is_available(dep["module"])

    return deps

def get_content_type_from_file(file_path):
    """Returns content type of a file.
    @param file_path: file path
    @return: content type
    """
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path)