LOCAL_SETTINGS = True
from .settings import *

DATABASES = {
    'default': {
        # Engine type. Ends with 'postgresql_psycopg2', 'mysql', 'sqlite3' or 'oracle'.
        'ENGINE': 'django.db.backends.sqlite3',
        # Database name or path to database file if using sqlite3.
        'NAME': 'db.sqlite',
        # Credntials. The following settings are not used with sqlite3.
        'USER': '',
        'PASSWORD': '',
        # Empty for localhost through domain sockets or '127.0.0.1' for localhost through TCP.
        'HOST': '',
        # Set to empty string for default port.
        'PORT': '',
        # Set timeout (avoids SQLite "database is locked" errors).
        'timeout': 300,
    }
}

# MySQL tuning.
#DATABASE_OPTIONS = {
# "init_command": "SET storage_engine=INNODB",
#}

# Mongo database settings
MONGO_URI = "mongodb://localhost/"
MONGO_DB = "sus_scrofa_db"

# Max uploaded image size (in bytes).
# Default is 150MB.
MAX_FILE_UPLOAD = 157286400

# Allowed file types.
ALLOWED_EXT = ['image/bmp', 'image/x-canon-cr2', 'image/jpeg', 'image/png',
               'image/x-canon-crw', 'image/x-eps', 'image/x-nikon-nef',
               'application/postscript', 'image/gif', 'image/x-minolta-mrw',
               'image/x-olympus-orf', 'image/x-photoshop', 'image/x-fuji-raf',
               'image/x-panasonic-raw2', 'image/x-tga', 'image/tiff', 'image/pjpeg',
               'image/x-x3f', 'image/x-portable-pixmap',
               # Modern image formats
               'image/webp', 'image/heif', 'image/heic', 'image/avif',
               'image/heif-sequence', 'image/heic-sequence']

# Auto upload from directory.
# Set a directory path to enable auto upload from file system.
# Set to None to disable.
AUTO_UPLOAD_DIR = None
# Delete a file after upload and submission.
AUTO_UPLOAD_DEL_ORIGINAL = True
# Clean up AUTO_UPLOAD_DIR on startup.
AUTO_UPLOAD_STARTUP_CLEANUP = True

# ── Plugin Configuration ──────────────────────────────────────────

# Photoholmes forgery detection (AI/ML tier plugin)
# Set to False to disable photoholmes even if the library is installed.
# Photoholmes can take 15-20 minutes per image on CPU-only systems.
# Default: True (enabled if photoholmes library is available)
ENABLE_PHOTOHOLMES = False

# Research-tier image content analysis (Phase 1c plugin)
# Object detection, person attributes, photorealism classification.
# Uses torchvision FasterRCNN — moderate CPU cost (~30-60s per image).
# Default: True (enabled if torch/torchvision are available)
ENABLE_RESEARCH_CONTENT_ANALYSIS = True

# Minimum authenticity score (0-100) after Phase 1a+1b to run research.
# Below this threshold, the image is likely fake and research is skipped.
# Default: 40 (skip research only for images that are clearly failing)
RESEARCH_CONFIDENCE_THRESHOLD = 40

# Save annotated debug images for researcher review.
# When enabled, generates images with bounding boxes, keypoints, zone circles,
# and detection highlights overlaid for visual verification.
# Stored in MongoDB GridFS alongside the analysis results.
RESEARCH_SAVE_ANNOTATIONS = True

# Override default secret key stored in secret_key.py
# Make this unique, and don't share it with anybody.
# SECRET_KEY = "YOUR_RANDOM_KEY"

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = "en-us"

ADMINS = (
    # ("Your Name", "your_email@example.com"),
)

MANAGERS = ADMINS

# Allow verbose debug error message in case of application fault.
# It's strongly suggested to set it to False if you are serving the
# web application from a web server front-end (i.e. Apache).
DEBUG = True

# A list of strings representing the host/domain names that this Django site
# can serve.
# Values in this list can be fully qualified names (e.g. 'www.example.com').
# When DEBUG is True or when running tests, host validation is disabled; any
# host will be accepted. Thus it's usually only necessary to set it in production.
ALLOWED_HOSTS = ["*"]