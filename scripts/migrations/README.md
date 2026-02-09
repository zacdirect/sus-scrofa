# Django Migration Scripts

This directory contains one-time migration scripts used during the project's evolution to fix compatibility issues with newer Django versions.

## Scripts

### fix_requestcontext.py
**Purpose:** Remove deprecated `context_instance=RequestContext(request)` from render calls  
**Django Version:** 1.8+ compatibility  
**Status:** Historical - applied during Django 1.8 upgrade

Removes the `context_instance` parameter from `render()` calls, which became unnecessary in Django 1.8+ as the request context processor is enabled by default.

### fix_reverse_calls.py
**Purpose:** Update `reverse()` calls to use URL names instead of view function paths  
**Django Version:** 2.0+ compatibility  
**Status:** Historical - applied during Django 2.0 upgrade

Converts old-style `reverse('app.views.function')` calls to new-style `reverse('url_name')` format required by Django 2.0+.

### fix_template_urls.py
**Purpose:** Update template `{% url %}` tags to use URL names  
**Django Version:** 2.0+ compatibility  
**Status:** Historical - applied during Django 2.0 upgrade

Updates Django templates to use named URL patterns instead of view function paths in `{% url %}` template tags.

## Usage

These scripts are kept for historical reference and should not need to be run again unless reverting to old code or applying similar migrations to other codebases.

If you need to run them:

```bash
cd /home/zac/repos/sus-scrofa/scripts/migrations
python3 fix_requestcontext.py
python3 fix_reverse_calls.py
python3 fix_template_urls.py
```

## Note

These scripts directly modify source files. Always ensure you have backups or are working in a clean git repository before running migration scripts.
