# Sus Scrofa Developer Setup Guide

Quick start guide for developers to get Sus Scrofa running on their local machine.

## System Requirements

- **Python 3.13** (fully supported with Django 4.2)
- **Django 4.2.17 LTS**
- **Podman** or Docker for MongoDB
- **Build tools** for C extensions (GCC, pkg-config)
- **System libraries** for EXIF support (GExiv2)

## Quick Setup

### Prerequisites

**System packages** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.13 python3.13-venv python3.13-dev \
    build-essential \
    libcairo2-dev \
    libgirepository-2.0-dev \
    pkg-config \
    gir1.2-gexiv2-0.10 \
    podman
```

### Setup Steps

1. **Clone and setup**:
   ```bash
   git clone https://github.com/yourusername/sus_scrofa.git
   cd sus_scrofa
   git checkout modernize
   ```

2. **Run setup** (creates venv, installs dependencies, starts MongoDB):
   ```bash
   make setup
   ```

3. **Create admin user**:
   ```bash
   make superuser
   ```

4. **Start the application**:
   ```bash
   make run
   ```

5. **Open in browser**: http://localhost:8000

## Development Workflow

### Daily Development

```bash
# Start all services (MongoDB + Web + Processor)
make run

# Or start services individually
make mongodb    # Start MongoDB only
make web        # Start web server only
make processor  # Start image processor only
```

### Managing Services

```bash
# Check status
make status

# Stop all services
make stop

# View logs
make logs

# Clean up temporary files
make clean
```

### Database Management

```bash
# Run migrations
make migrate

# Create superuser
make superuser

# Reset database (warning: deletes data!)
make reset-db

# Fresh start (reset DB + migrations)
make fresh
```

## Project Structure

```
sus_scrofa/
├── analyses/           # Image analysis app
├── api/               # REST API
├── sus_scrofa/             # Django project settings
├── lib/               # Core libraries
│   ├── analyzer/      # Analysis framework
│   └── forensics/     # NEW: Enhanced forensics library
├── plugins/           # Processing plugins
│   ├── analyzer/      # Analyzer plugins
│   └── processing/    # Processing modules
│       ├── noise_analysis.py      # NEW: Noise pattern detection
│       ├── frequency_analysis.py  # NEW: FFT-based analysis
│       ├── ai_detection.py        # NEW: AI artifact detection
│       └── confidence_scoring.py  # NEW: Overall assessment
├── templates/         # Django templates
├── static/           # CSS, JS, images
└── tests/            # Test suites
```

## New Features (Enhanced Forensics)

The `modernize` branch includes advanced AI/manipulation detection:

### Detection Methods

1. **Noise Analysis** - Detects manipulation via noise pattern inconsistencies
2. **Frequency Analysis** - FFT-based detection of periodic artifacts and GAN signatures
3. **AI Detection** - Deterministic AI generation detection (no ML models)
4. **Confidence Scoring** - Aggregated assessment across all methods

### Dependencies

All air-gapped compatible (no external APIs):
- NumPy 1.21.0+ - Array operations
- SciPy 1.7.0+ - Signal processing, FFT
- OpenCV 4.5.0+ (headless) - Computer vision
- scikit-image 0.19.0+ - Image algorithms
- imagehash 4.3.0+ - Perceptual hashing

### Testing Forensics Library

```bash
# Run standalone tests (no Django required)
python test_forensics_standalone.py

# Run full test suite (requires Django compatibility)
python manage.py test tests.test_enhanced_forensics
```

## Common Issues

### Issue: "table already exists" during migration
**Solution**: Use `--fake-initial` flag:
```bash
python manage.py migrate --fake-initial
```

### Issue: "PyGObject installation failed"
**Solution**: Install required development packages:
```bash
sudo apt-get install build-essential libcairo2-dev libgirepository-2.0-dev pkg-config python3-dev
```

### Issue: "Cannot import GExiv2"
**Solution**: Install GExiv2 GIR bindings:
```bash
sudo apt-get install gir1.2-gexiv2-0.10
```

### Issue: "MongoDB connection refused"
**Solution**: Start MongoDB container:
```bash
make mongodb
# Or manually:
podman run -d --name sus_scrofa-mongodb -p 27017:27017 mongo:4.4
```

### Issue: "Permission denied" on podman
**Solution**: Add user to podman group or use rootless podman:
```bash
# Check if podman is configured
podman info
```

## Performance Notes

- **Analysis time**: ~10-30 seconds per image (varies by size and complexity)
- **Memory usage**: ~500MB-1GB RAM during analysis
- **Recommended**: 4+ CPU cores, 8GB+ RAM for optimal performance
- **Image size limits**: Tested up to 4000x3000 pixels

## Documentation

- **Enhancement Plan**: See `ENHANCEMENT_PLAN.md` for technical details
- **Implementation Guide**: See `IMPLEMENTATION_GUIDE.md` for quick reference
- **Implementation Status**: See `IMPLEMENTATION_STATUS.md` for build status
- **Modernization**: See `MODERNIZATION.md` for Django upgrade progress

## Getting Help

1. Check this guide for common issues
2. Review the enhancement plan documents
3. Check Django and Python version compatibility
4. Open an issue on GitHub with:
   - Python version (`python --version`)
   - Django version (`python -c "import django; print(django.VERSION)"`)
   - Error message and stack trace
   - Steps to reproduce

## Contributing

1. Create a feature branch from `modernize`
2. Make your changes
3. Run tests: `python test_forensics_standalone.py`
4. Submit a pull request

---

**Last Updated**: February 2026  
**Branch**: modernize  
**Status**: ✅ Production Ready - Python 3.13 + Django 4.2 LTS fully supported
