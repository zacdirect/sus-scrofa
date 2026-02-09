# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import os
from django.conf import global_settings

DEBUG = True
TEMPLATE_DEBUG = DEBUG

ALLOWED_HOSTS = ['*']

SITE_ID = 1

# Actual SusScrofa release.
SUS_SCROFA_VERSION = "0.2.1"

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale.
USE_L10N = True

# Disabled time zone, using local time instead.
USE_TZ = False
TIME_ZONE = 'UTC'

# Default primary key field type (Django 4.2+)
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/var/www/example.com/media/"
MEDIA_ROOT = ''

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash.
# Examples: "http://example.com/media/", "http://media.example.com/"
MEDIA_URL = ''

# Absolute path to the directory static files should be collected to.
# Don't put anything in this directory yourself; store your static files
# in apps' "static/" subdirectories and in STATICFILES_DIRS.
# Example: "/var/www/example.com/static/"
STATIC_ROOT = ''

# URL prefix for static files.
# Example: "http://example.com/static/", "http://static.example.com/"
STATIC_URL = '/static/'

# Additional locations of static files
STATICFILES_DIRS = (
    os.path.join(os.getcwd(), "static"),
)

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
#    'django.contrib.staticfiles.finders.DefaultStorageFinder',
)

# Upload handler (store only on disk).
FILE_UPLOAD_HANDLERS = ("django.core.files.uploadhandler.TemporaryFileUploadHandler",)

# Unique secret key generator.
# Secret key will be placed in secret_key.py file.
try:
    from sus_scrofa.secret_key import SECRET_KEY
except ImportError:
    SETTINGS_DIR=os.path.abspath(os.path.dirname(__file__))
    # Using the same generation schema of Django startproject.
    from django.utils.crypto import get_random_string
    SECRET_KEY = get_random_string(50, "abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)")
    # Write secret_key.py
    with open(os.path.join(SETTINGS_DIR, "secret_key.py"), "w") as file:
        file.write("SECRET_KEY = \"{0}\"\n".format(SECRET_KEY))

# Templates configuration for Django 1.11+
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.static',
            ],
        },
    },
]

INTERNAL_IPS = ('127.0.0.1',)
MIDDLEWARE = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    #'debug_toolbar.middleware.DebugToolbarMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.http.ConditionalGetMiddleware',
    # Uncomment the next line for simple clickjacking protection:
    # 'django.middleware.clickjacking.XFrameOptionsMiddleware',
)

ROOT_URLCONF = 'sus_scrofa.urls'

# Python dotted path to the WSGI application used by Django's runserver.
WSGI_APPLICATION = 'sus_scrofa.wsgi.application'

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    #'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    #'debug_toolbar',
    "users",
    "analyses",
    "hashes",
    "manage",
)

# A sample logging configuration. The only tangible logging
# performed by this configuration is to send an email to
# the site admins on every HTTP 500 error when DEBUG=False.
# See http://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s %(levelname)s %(module)s %(message)s'
        },
        'simple': {
            'format': '%(asctime)s %(levelname)s %(message)s'
        },
        'management_command': {
            'format': "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        },
    },
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse'
        }
    },
    'handlers': {
        'mail_admins': {
            'level': 'ERROR',
            'filters': ['require_debug_false'],
            'class': 'django.utils.log.AdminEmailHandler'
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            },
        'processing': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'management_command',
            },
    },
    'loggers': {
        'django.request': {
            'handlers': ['mail_admins'],
            'level': 'ERROR',
            'propagate': True,
        },
        'lib': {
            'handlers': ['processing'],
            'level': 'DEBUG',
            'propagate': False,
            },
        'plugins': {
            'handlers': ['processing'],
            'level': 'DEBUG',
            'propagate': False,
            },
    }
}

# Custom user model.
AUTH_USER_MODEL = "users.Profile"
# Redirect logged users.
LOGIN_URL = "/users/login/"
LOGOUT_URL = "/users/logout/"
LOGIN_REDIRECT_URL = "/"

# Hack to import local settings.
try:
    LOCAL_SETTINGS
except NameError:
    try:
        from .local_settings import *
    except ImportError:
        pass