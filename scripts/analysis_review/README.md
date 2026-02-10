# Analysis Data Extraction Tool

Extract comprehensive analysis data from completed image analyses for review, debugging, and testing.

## Database Architecture

Sus Scrofa uses a **hybrid database model**:

- **Django SQLite** (`db.sqlite3`): Analysis records with metadata (ID, filename, state, timestamps)
- **MongoDB**: Full report data (EXIF, forensics results, AI detection, audit findings)

The `extract_data.py` script queries **both**:
1. Gets Analysis record from Django ORM (`Analysis.objects.get(id=...)`)
2. Reads `analysis.report` property which fetches MongoDB document
3. Combines into JSON with structured hierarchy

## Prerequisites

1. **MongoDB running**: `podman start sus-scrofa-mongodb` or `make start`
2. **Django environment**: Script auto-configures via `DJANGO_SETTINGS_MODULE`
3. **Python virtual environment** (optional but recommended): Activated `.venv`

## Finding Available Analysis IDs

### Quick Command
```bash
python -c "
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'sus_scrofa.settings'
import django
django.setup()
from analyses.models import Analysis
for a in Analysis.objects.filter(state='C').order_by('-id')[:20]:
    print(f'{a.id:3d}: {a.file_name}')
"
```

### Or Use Django Shell
```bash
python manage.py shell
>>> from analyses.models import Analysis
>>> Analysis.objects.filter(state='C').values('id', 'file_name', 'completed_at')
```

**States:**
- `C` = Completed (has full report data)
- `P` = Processing
- `Q` = Queued
- `E` = Error

## Usage

### Extract Specific Images
```bash
# From repository root
cd /home/zac/repos/sus-scrofa

# Single image
python scripts/analysis_review/extract_data.py 27

# Multiple images
python scripts/analysis_review/extract_data.py 27 24 17

# Hardcoded test set (IDs 3-10, update in script as needed)
python scripts/analysis_review/extract_data.py
```

## Common Issues

### "no such table: analyses_analysis"
The script is not finding Django's SQLite database.

**Fix**: Run from repository root, not from `scripts/analysis_review/`:
```bash
cd /home/user/repos/sus-scrofa  # ← Must be here
python scripts/analysis_review/extract_data.py 27
```

### "Analysis X not found"
That analysis ID doesn't exist in Django database.

**Fix**: List available IDs (see "Finding Available Analysis IDs" above)

### "No report found for analysis X"
Django record exists but MongoDB document is missing (analysis failed or was manually deleted).

**Fix**: Check MongoDB directly:
```python
from lib.db import get_db
db = get_db()
doc = db.analyses.find_one({'_id': ObjectId('...')})
```

Or re-run the analysis to regenerate the report.

### MongoDB Connection Error
MongoDB container not running.

**Fix**: 
```bash
make start  # Starts MongoDB + dev server
# or
podman start sus-scrofa-mongodb
```

## Use Cases

### 1. Debugging Low Scores
```bash
# Image scored 5/100 - why?
python scripts/analysis_review/extract_data.py 26
grep -A5 '"auditor"' scripts/analysis_review/analysis_26.json
```

### 2. Reviewing AI Detection
```bash
# Compare AI-generated vs real photos
python scripts/analysis_review/extract_data.py 24 17  # 24=Gemini AI, 17=Pixel 8
diff <(jq '.ai_detection.detection_layers' analysis_24.json) \
     <(jq '.ai_detection.detection_layers' analysis_17.json)
```

### 3. Testing Detector Changes
```bash
# Before code change
python scripts/analysis_review/extract_data.py 27
cp analysis_27.json analysis_27_before.json

# Make code changes, reprocess image, extract again
python scripts/analysis_review/extract_data.py 27

# Compare
diff analysis_27_before.json analysis_27.json
```

### 4. Batch Analysis Review
```bash
# Extract all completed analyses
python -c "
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'sus_scrofa.settings'
import django
django.setup()
from analyses.models import Analysis
ids = [a.id for a in Analysis.objects.filter(state='C')]
print(' '.join(map(str, ids)))
" | xargs python scripts/analysis_review/extract_data.py
```

## Data Privacy Note

Extracted JSON files may contain:
- Full EXIF metadata (GPS coordinates, camera serial numbers)
- Software signatures
- Perceptual hashes
- File content indicators

**Do not commit** `analysis_*.json` files to version control. They're in `.gitignore` by default.

## Troubleshooting Checklist

- [ ] Running from `/home/user/repos/sus-scrofa` (repo root)?
- [ ] MongoDB container running? (`podman ps | grep mongodb`)
- [ ] Using valid completed analysis ID? (state='C')
- [ ] Virtual environment activated? (`.venv/bin/activate`)
- [ ] Django migrations up to date? (`python manage.py migrate`)
