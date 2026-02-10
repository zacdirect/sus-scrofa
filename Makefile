.PHONY: help start stop status logs clean mongodb web processor dev run setup install check-deps venv reset-db fresh ai-setup ai-verify ai-clean photoholmes-setup photoholmes-verify photoholmes-clean mantranet-setup mantranet-verify mantranet-clean research-setup research-verify research-clean detect-system

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Paths
VENV_DIR := .venv
VENV := $(VENV_DIR)/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip
MANAGE := $(PYTHON) manage.py
SYSTEM_PYTHON := python3.13

# Check if venv exists and is functional
VENV_EXISTS := $(shell test -f $(VENV_DIR)/bin/pip && echo 1 || echo 0)

help:
	@echo "$(GREEN)Sus Scrofa Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make setup       - Complete setup with ALL features (RECOMMENDED)"
	@echo "                     Includes: venv, core deps, AI detection (SPAI/SDXL),"
	@echo "                               research models, photoholmes, MongoDB, migrations"
	@echo ""
	@echo "$(YELLOW)Manual Setup (Advanced):$(NC)"
	@echo "  make install     - Install core Python dependencies only (minimal)"
	@echo "  make venv        - Create Python virtual environment"
	@echo "  make check-deps  - Check system dependencies"
	@echo "  make mongodb     - Start MongoDB container (Podman)"
	@echo "  make fresh       - Fresh start (reset DBs, recreate everything)"
	@echo "  make detect-system - Detect GPU/CUDA availability for AI/ML"
	@echo ""
	@echo "$(YELLOW)AI Detection (Optional):$(NC)"
	@echo "  make ai-setup    - Setup AI detection (SPAI + HuggingFace models)"
	@echo "  make ai-verify   - Verify AI detection installation"
	@echo "  make ai-clean    - Remove AI detection environment"
	@echo ""
	@echo "$(YELLOW)Photoholmes Forgery Detection (Optional):$(NC)"
	@echo "  make photoholmes-setup    - Install photoholmes + torch (CPU) + jpegio"
	@echo "  make photoholmes-verify   - Verify photoholmes installation"
	@echo "  make photoholmes-clean    - Remove photoholmes from environment"
	@echo ""
	@echo "$(YELLOW)ManTraNet Forgery Localization (Optional):$(NC)"
	@echo "  make mantranet-setup      - Install ManTraNet + TensorFlow 1.14 (isolated venv)"
	@echo "  make mantranet-verify     - Verify ManTraNet installation"
	@echo "  make mantranet-clean      - Remove ManTraNet environment"
	@echo ""
	@echo "$(YELLOW)Research Content Analysis (Optional):$(NC)"
	@echo "  make research-setup      - Install torch + download pretrained models"
	@echo "  make research-verify     - Verify models are cached and ready"
	@echo "  make research-clean      - Remove cached model weights"
	@echo ""
	@echo "$(YELLOW)OpenCV Service (Container):$(NC)"
	@echo "  make opencv-build   - Build OpenCV service container"
	@echo "  make opencv-start   - Start OpenCV service (port 8080)"
	@echo "  make opencv-stop    - Stop OpenCV service"
	@echo "  make opencv-restart - Restart OpenCV service"
	@echo "  make opencv-logs    - View OpenCV service logs"
	@echo "  make opencv-test    - Test OpenCV service health"
	@echo "  make opencv-clean   - Remove OpenCV service container"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make run         - Quick start (MongoDB + Web + Processor)"
	@echo "  make dev         - Same as 'make run'"
	@echo "  make web         - Start web server only"
	@echo "  make processor   - Start image processor only"
	@echo ""
	@echo "$(YELLOW)Management:$(NC)"
	@echo "  make status      - Check if services are running"
	@echo "  make stop        - Stop all Sus Scrofa services"
	@echo "  make logs        - Show recent logs"
	@echo "  make clean       - Stop services and clean temp files"
	@echo ""
	@echo "$(YELLOW)Database:$(NC)"
	@echo "  make migrate     - Run database migrations"
	@echo "  make superuser   - Create superuser account"
	@echo "  make reset-db    - Reset all databases (SQLite + MongoDB)"

check-deps:
	@echo "$(GREEN)Checking system dependencies...$(NC)"
	@echo ""
	@echo "$(YELLOW)Python:$(NC)"
	@which $(SYSTEM_PYTHON) > /dev/null 2>&1 && echo "$(GREEN)✓ Python 3.13 found$(NC)" || (echo "$(RED)✗ Python 3.13 not found$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)Container Runtime:$(NC)"
	@which podman > /dev/null 2>&1 && echo "$(GREEN)✓ Podman found$(NC)" || echo "$(YELLOW)⚠ Podman not found (needed for MongoDB)$(NC)"
	@echo ""
	@echo "$(YELLOW)Build Tools:$(NC)"
	@which gcc > /dev/null 2>&1 && echo "$(GREEN)✓ GCC found$(NC)" || echo "$(RED)✗ GCC not found (needed for PyGObject)$(NC)"
	@which pkg-config > /dev/null 2>&1 && echo "$(GREEN)✓ pkg-config found$(NC)" || echo "$(RED)✗ pkg-config not found$(NC)"
	@echo ""
	@echo "$(YELLOW)System Libraries:$(NC)"
	@pkg-config --exists cairo && echo "$(GREEN)✓ Cairo found$(NC)" || echo "$(RED)✗ Cairo not found (install libcairo2-dev)$(NC)"
	@pkg-config --exists girepository-2.0 && echo "$(GREEN)✓ GIRepository-2.0 found$(NC)" || echo "$(RED)✗ GIRepository-2.0 not found (install libgirepository-2.0-dev)$(NC)"
	@dpkg -l | grep -q libexempi8 && echo "$(GREEN)✓ Exempi found (XMP metadata)$(NC)" || echo "$(YELLOW)⚠ Exempi not found (install libexempi8 for XMP support)$(NC)"
	@echo ""
	@dpkg -l | grep -q python3.13-dev && echo "$(GREEN)✓ Python dev headers found$(NC)" || echo "$(RED)✗ Python dev headers not found (install python3.13-dev)$(NC)"

detect-system:
	@echo "$(GREEN)Detecting GPU/CUDA configuration for AI/ML...$(NC)"
	@echo ""
	@$(SYSTEM_PYTHON) scripts/detect_system.py

venv:
	@if [ $(VENV_EXISTS) -eq 1 ]; then \
		echo "$(GREEN)✓ Virtual environment already exists$(NC)"; \
	else \
		echo "$(YELLOW)Creating Python virtual environment...$(NC)"; \
		$(SYSTEM_PYTHON) -m venv $(VENV_DIR); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	fi

install: venv
	@echo "$(GREEN)Installing Python dependencies...$(NC)"
	@echo ""
	@echo "$(YELLOW)Upgrading pip...$(NC)"
	@$(VENV)/pip install --upgrade pip setuptools wheel
	@echo ""
	@echo "$(YELLOW)Installing requirements...$(NC)"
	@$(VENV)/pip install -r requirements.txt
	@echo ""
	@echo "$(YELLOW)Verifying modern metadata libraries...$(NC)"
	@$(VENV)/python -c "import exif; print('  ✓ exif (pure Python EXIF extraction)')" 2>/dev/null || \
		(echo "$(RED)  ✗ exif library not found$(NC)" && exit 1)
	@$(VENV)/python -c "import pillow_heif; print('  ✓ pillow-heif (HEIF/HEIC support)')" 2>/dev/null || \
		(echo "$(RED)  ✗ pillow-heif library not found$(NC)" && exit 1)
	@$(VENV)/python -c "from libxmp import XMPFiles; print('  ✓ python-xmp-toolkit (XMP metadata)')" 2>/dev/null || \
		(echo "$(YELLOW)  ⚠ python-xmp-toolkit not found (XMP support unavailable)$(NC)")
	@echo ""
	@echo "$(YELLOW)Installing PyGObject (for system integration)...$(NC)"
	@$(VENV)/pip install PyGObject || (echo "$(RED)PyGObject installation failed. Make sure you have:$(NC)" && \
		echo "  sudo apt-get install build-essential libcairo2-dev libgirepository-2.0-dev pkg-config python3.13-dev" && exit 1)
	@echo ""
	@echo "$(YELLOW)Verifying scientific computing libraries...$(NC)"
	@$(VENV)/python -c "import numpy, scipy, cv2, skimage; print('  ✓ NumPy, SciPy, OpenCV, scikit-image')" 2>/dev/null || \
		(echo "$(RED)  ✗ Scientific libraries not found. Run 'make install' again.$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)✓ Core dependencies installed$(NC)"
	@echo ""
	@echo "$(YELLOW)This is a minimal installation. To get all features, run:$(NC)"
	@echo "  make setup"
	@echo ""
	@echo "$(YELLOW)Or install features individually:$(NC)"
	@echo "  make ai-setup           # AI detection (SPAI + SDXL)"
	@echo "  make research-setup     # Research content analysis"
	@echo "  make photoholmes-setup  # Photoholmes forgery detection"
	@echo ""
	@echo "$(YELLOW)To start using Sus Scrofa:$(NC)"
	@echo "  1. Start MongoDB:    make mongodb"
	@echo "  2. Run migrations:   make migrate"
	@echo "  3. Create superuser: make superuser"
	@echo "  4. Start services:   make run"

setup: check-deps venv
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Complete Sus Scrofa Setup$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Phase 1: Installing core dependencies...$(NC)"
	@$(MAKE) install
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Phase 2: Setting up AI/ML features$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)1. AI Detection (SPAI + SDXL)...$(NC)"
	@$(MAKE) ai-setup || echo "$(YELLOW)⚠ AI detection setup failed (optional)$(NC)"
	@echo ""
	@echo "$(YELLOW)2. Research Content Analysis...$(NC)"
	@$(MAKE) research-setup || echo "$(YELLOW)⚠ Research setup failed (optional)$(NC)"
	@echo ""
	@echo "$(YELLOW)3. Photoholmes Forgery Detection...$(NC)"
	@$(MAKE) photoholmes-setup || echo "$(YELLOW)⚠ Photoholmes setup failed (optional)$(NC)"
	@echo ""
	@echo "$(YELLOW)4. ManTraNet Forgery Localization...$(NC)"
	@$(MAKE) mantranet-setup || echo "$(YELLOW)⚠ ManTraNet setup failed (optional)$(NC)"
	@echo ""
	@echo "$(YELLOW)5. OpenCV Manipulation Detection Service...$(NC)"
	@$(MAKE) opencv-build || echo "$(YELLOW)⚠ OpenCV build failed (optional)$(NC)"
	@$(MAKE) opencv-start || echo "$(YELLOW)⚠ OpenCV start failed (optional)$(NC)"
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Phase 3: Database Setup$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Starting MongoDB...$(NC)"
	@$(MAKE) mongodb
	@echo ""
	@echo "$(YELLOW)Running database migrations...$(NC)"
	@$(MAKE) migrate
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Complete Setup Finished!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(GREEN)Installed Features:$(NC)"
	@echo "  ✓ Core forensic analysis"
	@echo "  ✓ ManTraNet Forgery Localization"
	@echo "  ✓ OpenCV Manipulation Detection"
	@echo "  ✓ MongoDB database"
	@echo ""
	@echo "$(YELLOW)Verify installations (optional):$(NC)"
	@echo "  make ai-verify"
	@echo "  make research-verify"
	@echo "  make photoholmes-verify"
	@echo "  make mantranetstallations (optional):$(NC)"
	@echo "  make ai-verify"
	@echo "  make research-verify"
	@echo "  make photoholmes-verify"
	@echo ""
	@echo "$(YELLOW)Next Steps:$(NC)"
	@echo "  1. Create superuser: make superuser"
	@echo "  2. Start services:   make run"
	@echo "  3. Access web UI:    http://localhost:8000"
	@echo ""
	@echo "$(YELLOW)Then start the application:$(NC)"
	@echo "  make run"
	@echo ""


mongodb:
	@echo "$(GREEN)Checking MongoDB container...$(NC)"
	@if lsof -i :27017 >/dev/null 2>&1 || nc -z localhost 27017 >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Port 27017 is already in use (assuming MongoDB is running)$(NC)"; \
	elif podman ps -a --format "{{.Names}}" | grep -q "^sus-scrofa-mongodb$$"; then \
		if podman ps --format "{{.Names}}" | grep -q "^sus-scrofa-mongodb$$"; then \
			echo "$(GREEN)✓ MongoDB container is already running$(NC)"; \
		else \
			echo "$(YELLOW)Starting existing MongoDB container...$(NC)"; \
			podman start sus-scrofa-mongodb; \
		fi \
	else \
		echo "$(YELLOW)Creating new MongoDB container...$(NC)"; \
		podman run -d --name sus-scrofa-mongodb \
			-p 27017:27017 \
			-v sus-scrofa-mongodb-data:/data/db \
			docker.io/library/mongo:4.4; \
	fi
	@sleep 2
	@echo "$(GREEN)✓ MongoDB is ready$(NC)"

web: venv
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "$(RED)Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Starting Sus Scrofa web server...$(NC)"
	@$(MANAGE) runserver 0.0.0.0:8000

processor: venv
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "$(RED)Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Starting Sus Scrofa image processor...$(NC)"
	@$(MANAGE) process

run: venv mongodb
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "$(RED)Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@if ! test -f db.sqlite; then \
		echo "$(YELLOW)Database not found. Running migrations...$(NC)"; \
		$(MAKE) migrate; \
		echo ""; \
		echo "$(YELLOW)⚠ Don't forget to create a superuser:$(NC)"; \
		echo "  Press Ctrl+C, then run: make superuser"; \
		echo ""; \
		sleep 3; \
	fi
	@echo "$(GREEN)Starting Sus Scrofa in development mode...$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop all services$(NC)"
	@echo ""
	@trap 'make stop' INT; \
	$(MANAGE) runserver 0.0.0.0:8000 & \
	WEB_PID=$$!; \
	sleep 2; \
	$(MANAGE) process & \
	PROC_PID=$$!; \
	echo ""; \
	echo "$(GREEN)✓ Services started:$(NC)"; \
	echo "  Web Server:      http://localhost:8000 (PID: $$WEB_PID)"; \
	echo "  Image Processor: Running (PID: $$PROC_PID)"; \
	echo "  MongoDB:         Running on port 27017"; \
	echo ""; \
	echo "$(YELLOW)Press Ctrl+C to stop$(NC)"; \
	wait

dev: run

status:
	@echo "$(GREEN)Sus Scrofa Service Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)MongoDB Container:$(NC)"
	@if podman ps --format "{{.Names}}\t{{.Status}}" | grep sus-scrofa-mongodb; then \
		echo "$(GREEN)✓ Running$(NC)"; \
	else \
		echo "$(RED)✗ Not running$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Web Server:$(NC)"
	@if pgrep -f "manage.py runserver" > /dev/null; then \
		echo "$(GREEN)✓ Running$(NC) (PID: $$(pgrep -f 'manage.py runserver' | head -1))"; \
	else \
		echo "$(RED)✗ Not running$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Image Processor:$(NC)"
	@if pgrep -f "manage.py process" > /dev/null; then \
		echo "$(GREEN)✓ Running$(NC) (PID: $$(pgrep -f 'manage.py process' | head -1))"; \
	else \
		echo "$(RED)✗ Not running$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)OpenCV Service:$(NC)"
	@if podman ps --format "{{.Names}}" 2>/dev/null | grep -q sus-scrofa-opencv; then \
		echo "$(GREEN)✓ Running$(NC) (http://localhost:8080)"; \
	else \
		echo "$(RED)✗ Not running$(NC) (run: make opencv-start)"; \
	fi

stop:
	@echo "$(YELLOW)Stopping Sus Scrofa services...$(NC)"
	@-pkill -f "manage.py runserver" 2>/dev/null && echo "$(GREEN)✓ Web server stopped$(NC)" || echo "$(YELLOW)Web server not running$(NC)"
	@-pkill -f "manage.py process" 2>/dev/null && echo "$(GREEN)✓ Image processor stopped$(NC)" || echo "$(YELLOW)Image processor not running$(NC)"
	@echo "$(GREEN)✓ All services stopped$(NC)"
	@echo "$(YELLOW)Note: MongoDB container is still running. Use 'podman stop sus-scrofa-mongodb' to stop it.$(NC)"

logs:
	@echo "$(GREEN)Recent MongoDB logs:$(NC)"
	@podman logs --tail 20 sus-scrofa-mongodb 2>/dev/null || echo "$(RED)MongoDB container not found$(NC)"

clean: stop
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

migrate: venv
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "$(RED)Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Running database migrations...$(NC)"
	@$(MANAGE) migrate --run-syncdb

superuser: venv
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "$(RED)Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Creating superuser account...$(NC)"
	@$(MANAGE) createsuperuser

reset-db: stop
	@echo "$(RED)WARNING: This will delete SQLite database (Django users/sessions)!$(NC)"
	@echo "$(YELLOW)Note: MongoDB data will be preserved$(NC)"
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@echo ""
	@echo "$(YELLOW)Removing SQLite database...$(NC)"
	@rm -f db.sqlite
	@echo "$(GREEN)✓ SQLite database removed$(NC)"
	@echo ""
	@echo "$(YELLOW)Running migrations to recreate database...$(NC)"
	@$(MAKE) migrate
fresh: reset-db
	@echo ""
	@echo "$(YELLOW)Clearing MongoDB analysis data...$(NC)"
	@$(PYTHON) -c "\
import os, sys, django; \
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sus_scrofa.settings'); \
django.setup(); \
from analyses.models import Analysis; \
from lib.db import get_db; \
db = get_db(); \
result = db.analyses.delete_many({}) if db is not None else None; \
count = Analysis.objects.all().count(); \
Analysis.objects.all().delete(); \
print(f'Deleted {result.deleted_count if result else 0} MongoDB documents'); \
print(f'Deleted {count} Analysis records from SQLite')" 2>/dev/null || echo "$(YELLOW)MongoDB clear skipped (not connected)$(NC)"
	@echo "$(GREEN)✓ MongoDB analysis data cleared$(NC)"
	@echo ""
	@echo "$(GREEN)Database reset complete!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Create superuser: make superuser"
	@echo "  2. Start development: make dev"
	@echo ""

# AI Detection Setup
ai-setup: venv
	@echo "$(GREEN)Setting up AI detection module...$(NC)"
	@echo ""
	@echo "$(YELLOW)Installing AI detection dependencies into main venv...$(NC)"
	@INDEX_URL=$$($(SYSTEM_PYTHON) scripts/detect_system.py --index-url); \
	BACKEND=$$($(SYSTEM_PYTHON) scripts/detect_system.py --backend); \
	echo "Backend: $$BACKEND"; \
	echo "PyTorch Index: $$INDEX_URL"; \
	echo ""; \
	echo "$(YELLOW)Installing PyTorch and AI detection dependencies...$(NC)"; \
	$(PIP) install --extra-index-url "$$INDEX_URL" "torch>=2.0.0" "torchvision>=0.15.0"; \
	$(PIP) install "opencv-python>=4.10.0" "pyyaml>=6.0.1" "scipy>=1.14.0" \
		"timm==0.4.12" "yacs>=0.1.8" "numpy>=1.26.4" "torchmetrics>=1.4.0" \
		"tqdm>=4.66.4" "pillow>=10.4.0" "einops>=0.8.0" "ftfy>=6.1.0" "regex>=2023.0.0" \
		"transformers>=4.36.0" "safetensors>=0.4.0" "open-clip-torch"; \
	echo "$(GREEN)✓ Dependencies installed$(NC)"
	@echo ""
	@echo "$(YELLOW)Downloading AI detection models...$(NC)"
	@echo "  This may take a few minutes on first run..."
	@echo ""
	@cd ai_detection && PYTHON=../$(PYTHON) $(MAKE) weights models || \
		(echo "$(RED)✗ Model download failed$(NC)" && \
		 echo "$(YELLOW)You can retry with: cd ai_detection && make models$(NC)" && \
		 exit 1)
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ AI detection ready!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "The AI detection plugin will now be available for image analysis."
	@echo ""
	@echo "Verify installation:"
	@echo "  make ai-verify"
	@echo ""
	@echo "Check model status:"
	@echo "  cd ai_detection && make models-list"


ai-verify:
	@echo "$(GREEN)Verifying AI detection module...$(NC)"
	@cd ai_detection && PYTHON=../$(PYTHON) $(MAKE) verify

ai-clean:
	@echo "$(YELLOW)Cleaning AI detection module...$(NC)"
	@cd ai_detection && $(MAKE) clean
	@echo "$(GREEN)✓ AI detection cleaned$(NC)"

# Photoholmes Forgery Detection
photoholmes-setup:
	@echo "$(GREEN)Setting up photoholmes forgery detection...$(NC)"
	@echo ""
	@echo "$(YELLOW)Detecting system configuration...$(NC)"
	@INDEX_URL=$$($(SYSTEM_PYTHON) scripts/detect_system.py --index-url); \
	BACKEND=$$($(SYSTEM_PYTHON) scripts/detect_system.py --backend); \
	echo "Backend: $$BACKEND"; \
	echo "PyTorch Index: $$INDEX_URL"; \
	echo ""; \
	echo "$(YELLOW)Step 1: Installing PyTorch ($$BACKEND)...$(NC)"; \
	if [ "$$BACKEND" = "CPU" ]; then \
		echo "Installing lightweight CPU-only version (~190MB)"; \
	else \
		echo "Installing GPU-accelerated version with CUDA support"; \
	fi; \
	$(PIP) install --index-url $$INDEX_URL torch torchvision; \
	echo ""; \
	echo "$(YELLOW)Step 2: Installing additional dependencies...$(NC)"; \
	$(PIP) install "jpegio>=0.2.8" torchmetrics torch_kmeans; \
	echo ""; \
	echo "$(YELLOW)Step 3: Installing photoholmes from GitHub...$(NC)"; \
	echo "Since torch is already installed, pip won't re-download different version."; \
	$(PIP) install --no-deps "photoholmes @ git+https://github.com/photoholmes/photoholmes.git"; \
	echo "Installing photoholmes's other dependencies (numpy, opencv, etc)..."; \
	$(PIP) install numpy matplotlib opencv-python scikit-learn scikit-image pydantic tqdm scipy pyyaml mpmath typer pillow wget ipykernel; \
	echo ""; \
	echo "$(GREEN)========================================$(NC)"; \
	echo "$(GREEN)✓ Photoholmes ready!$(NC)"; \
	echo "$(GREEN)========================================$(NC)"; \
	echo ""; \
	echo "CPU methods available: DQ, ZERO, Noisesniffer"; \
	echo "GPU methods: set SUSSCROFA_PHOTOHOLMES_GPU=1 and download weights"; \
	echo ""; \
	echo "Verify installation:"; \
	echo "  make photoholmes-verify"

photoholmes-verify:
	@echo "$(GREEN)Verifying photoholmes installation...$(NC)"
	@$(PYTHON) -c "from photoholmes.methods.factory import MethodFactory; print('✓ photoholmes library OK')" 2>/dev/null || echo "$(RED)✗ photoholmes not installed$(NC)"
	@$(PYTHON) -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "$(RED)✗ PyTorch not installed$(NC)"
	@$(PYTHON) -c "import jpegio; print('✓ jpegio OK')" 2>/dev/null || echo "$(RED)✗ jpegio not installed$(NC)"
	@$(PYTHON) -c "from photoholmes.methods.factory import MethodFactory; [exec('try:\\n method, pp = MethodFactory.load(m)\\n print(f\"  ✓ {m}: loaded\")\\nexcept Exception as e:\\n print(f\"  ✗ {m}: {e}\")') for m in ['dq', 'zero', 'noisesniffer']]" 2>/dev/null || echo "$(RED)Method loading failed$(NC)"

photoholmes-clean:
	@echo "$(YELLOW)Removing photoholmes...$(NC)"
	$(PIP) uninstall -y photoholmes 2>/dev/null || true
	@echo "$(GREEN)✓ Photoholmes removed$(NC)"
	@echo "$(YELLOW)Note: torch/torchvision left installed (may be used by other modules)$(NC)"

# ManTraNet Forgery Localization
mantranet-setup: venv
	@echo "$(GREEN)Setting up ManTraNet forgery localization...$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 1: Creating directory structure...$(NC)"
	@mkdir -p models/weights/mantranet ai_detection/mantranet/src
	@echo "$(GREEN)✓ Directories created$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 2: Downloading model architecture files...$(NC)"
	@if [ ! -f "ai_detection/mantranet/src/modern_mantranet.py" ]; then \
		echo "$(RED)✗ Model architecture missing - this should not happen!$(NC)" && exit 1; \
	else \
		echo "$(GREEN)✓ Model architecture present (modern TensorFlow 2.20/Keras 3.x implementation)$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Step 3: Installing TensorFlow into main venv...$(NC)"
	@INDEX_URL=$$($(SYSTEM_PYTHON) scripts/detect_system.py --index-url); \
	$(PIP) install --quiet --extra-index-url "$$INDEX_URL" "tensorflow>=2.10.0" && \
	echo "$(GREEN)✓ TensorFlow installed$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 4: Installing additional dependencies...$(NC)"
	@$(PIP) install --quiet "scipy>=1.4.0" "matplotlib>=3.0.0" && \
		echo "$(GREEN)✓ Dependencies installed$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 5: Downloading pretrained Keras model (~170MB)...$(NC)"
	@if [ ! -f "models/weights/mantranet/ManTraNet_Ptrain4.h5" ]; then \
		wget -q --show-progress -O models/weights/mantranet/ManTraNet_Ptrain4.h5 \
			https://raw.githubusercontent.com/ISICV/ManTraNet/master/pretrained_weights/ManTraNet_Ptrain4.h5 && \
		echo "$(GREEN)✓ Model downloaded$(NC)"; \
	else \
		echo "$(GREEN)✓ Model already exists ($(shell du -h models/weights/mantranet/ManTraNet_Ptrain4.h5 2>/dev/null | cut -f1))$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ ManTraNet ready!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "Setup created:"
	@echo "  • Model weights: models/weights/mantranet/ManTraNet_Ptrain4.h5"
	@echo "  • Architecture: ai_detection/mantranet/src/modern_mantranet.py (TensorFlow 2.20/Keras 3.x)"
	@echo "  • Python $(shell python --version 2>&1 | cut -d' ' -f2) + TensorFlow $(shell python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'N/A')"
	@echo ""
	@echo "Verify installation:"
	@echo "  make mantranet-verify"

mantranet-verify:
	@echo "$(GREEN)Verifying ManTraNet installation...$(NC)"
	@if [ -f "models/weights/mantranet/ManTraNet_Ptrain4.h5" ]; then \
		echo "$(GREEN)✓ Model file exists ($(shell du -h models/weights/mantranet/ManTraNet_Ptrain4.h5 2>/dev/null | cut -f1))$(NC)"; \
	else \
		echo "$(RED)✗ Model file not found$(NC)" && exit 1; \
	fi
	@$(PYTHON) -c "import tensorflow as tf; print('$(GREEN)✓ TensorFlow ' + tf.__version__ + '$(NC)')" 2>/dev/null || echo "$(RED)✗ TensorFlow not installed$(NC)"
	@$(PYTHON) -c "import scipy, matplotlib; print('$(GREEN)✓ Dependencies OK$(NC)')" 2>/dev/null || echo "$(RED)✗ Dependencies missing$(NC)"
	@echo "$(GREEN)✓ ManTraNet is ready$(NC)"

mantranet-clean:
	@echo "$(YELLOW)Removing ManTraNet model...$(NC)"
	@rm -rf models/weights/mantranet
	@echo "$(GREEN)✓ ManTraNet model removed$(NC)"
	@echo "$(YELLOW)Note: TensorFlow left installed (may be used by other modules)$(NC)"

# Research Content Analysis (Phase 1c)
research-setup: venv
	@echo "$(GREEN)Setting up research content analysis models...$(NC)"
	@echo ""
	@echo "$(YELLOW)Detecting system configuration...$(NC)"
	@INDEX_URL=$$($(SYSTEM_PYTHON) scripts/detect_system.py --index-url); \
	BACKEND=$$($(SYSTEM_PYTHON) scripts/detect_system.py --backend); \
	echo "Backend: $$BACKEND"; \
	echo "PyTorch Index: $$INDEX_URL"; \
	echo ""; \
	if $(PYTHON) -c "import torch" 2>/dev/null; then \
		echo "$(GREEN)✓ PyTorch already installed$(NC)"; \
	else \
		echo "$(YELLOW)Installing PyTorch ($$BACKEND)...$(NC)"; \
		echo "  This may take several minutes..."; \
		$(PIP) install --extra-index-url $$INDEX_URL torch torchvision; \
	fi
	@echo ""
	@echo "$(YELLOW)Downloading pretrained models (~395 MB)...$(NC)"
	@$(PYTHON) scripts/download_research_models.py
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Research content analysis ready!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "Models: FasterRCNN, KeypointRCNN, YuNet face detector"
	@echo "Enable in settings: ENABLE_RESEARCH_CONTENT_ANALYSIS = True"
	@echo ""
	@echo "Verify installation:"
	@echo "  make research-verify"

research-verify: venv
	@echo "$(GREEN)Verifying research content analysis...$(NC)"
	@echo ""
	@echo "$(YELLOW)1. Checking PyTorch...$(NC)"
	@$(PYTHON) -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || \
		(echo "  ✗ PyTorch not installed. Run: make research-setup" && exit 1)
	@$(PYTHON) -c "import torchvision; print(f'  ✓ torchvision {torchvision.__version__}')" 2>/dev/null || \
		(echo "  ✗ torchvision not installed. Run: make research-setup" && exit 1)
	@echo ""
	@echo "$(YELLOW)2. Checking OpenCV...$(NC)"
	@$(PYTHON) -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" 2>/dev/null || \
		(echo "  ✗ OpenCV not installed" && exit 1)
	@echo ""
	@echo "$(YELLOW)3. Checking pretrained model cache...$(NC)"
	@$(PYTHON) scripts/download_research_models.py --verify
	@echo "$(GREEN)✓ Research content analysis is ready$(NC)"

research-clean:
	@echo "$(YELLOW)Removing cached research models...$(NC)"
	@$(PYTHON) scripts/download_research_models.py --clean 2>/dev/null || \
		echo "$(YELLOW)Could not run cleanup script (torch not installed?)$(NC)"
	@echo "$(YELLOW)Note: torch/torchvision left installed (may be used by other modules)$(NC)"

# OpenCV Service (Container-based)
opencv-build:
	@echo "$(GREEN)Building OpenCV service container...$(NC)"
	@cd opencv_service && podman build -t sus-scrofa-opencv:latest .
	@echo "$(GREEN)✓ OpenCV service image built$(NC)"

opencv-start:
	@echo "$(GREEN)Starting OpenCV service...$(NC)"
	@podman network exists sus-scrofa-net 2>/dev/null || podman network create sus-scrofa-net
	@podman rm -f sus-scrofa-opencv 2>/dev/null || true
	@podman run -d \
		--replace \
		--name sus-scrofa-opencv \
		-p 8080:8080 \
		--network sus-scrofa-net \
		sus-scrofa-opencv:latest
	@sleep 2
	@echo "$(GREEN)✓ OpenCV service started on http://localhost:8080$(NC)"
	@echo ""
	@echo "Health check: curl http://localhost:8080/health"

opencv-stop:
	@echo "$(YELLOW)Stopping OpenCV service...$(NC)"
	@podman stop sus-scrofa-opencv 2>/dev/null || true
	@podman rm sus-scrofa-opencv 2>/dev/null || true
	@echo "$(GREEN)✓ OpenCV service stopped$(NC)"

opencv-logs:
	@podman logs -f sus-scrofa-opencv

opencv-restart: opencv-stop opencv-start

opencv-shell:
	@podman exec -it sus-scrofa-opencv /bin/bash

opencv-test:
	@echo "$(GREEN)Testing OpenCV service...$(NC)"
	@curl -s http://localhost:8080/health | python3 -m json.tool || echo "$(RED)Service not responding$(NC)"

opencv-clean: opencv-stop
	@echo "$(YELLOW)Removing OpenCV service image...$(NC)"
	@podman rmi sus-scrofa-opencv:latest 2>/dev/null || true
	@echo "$(GREEN)✓ OpenCV service cleaned$(NC)"
