.PHONY: help start stop status logs clean mongodb web processor dev run setup install check-deps venv reset-db fresh ai-setup ai-verify ai-clean photoholmes-setup photoholmes-verify photoholmes-clean detect-system

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
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make setup       - Complete setup (venv, deps, MongoDB, migrations)"
	@echo "  make fresh       - Fresh start (reset DBs, recreate everything)"
	@echo "  make venv        - Create Python virtual environment"
	@echo "  make install     - Install Python dependencies"
	@echo "  make check-deps  - Check system dependencies"
	@echo "  make mongodb     - Start MongoDB container (Podman)"
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
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

setup: check-deps install mongodb
	@echo ""
	@echo "$(YELLOW)Running database migrations...$(NC)"
	@$(MAKE) migrate
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Important: Create a superuser account$(NC)"
	@echo "  Run: make superuser"
	@echo ""
	@echo "Then start the application:"
	@echo "  Run: make run"
	@echo ""

mongodb:
	@echo "$(GREEN)Checking MongoDB container...$(NC)"
	@if podman ps -a --format "{{.Names}}" | grep -q "^sus-scrofa-mongodb$$"; then \
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
ai-setup:
	@echo "$(GREEN)Setting up AI detection module...$(NC)"
	@echo ""
	@cd ai_detection && $(MAKE) setup
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ AI detection ready!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "The AI detection plugin will now be available for image analysis."
	@echo ""
	@echo "Verify installation:"
	@echo "  make ai-verify"

ai-verify:
	@echo "$(GREEN)Verifying AI detection module...$(NC)"
	@cd ai_detection && $(MAKE) verify

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
	$(PIP) install jpegio>=0.4.0 torchmetrics torch_kmeans; \
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
