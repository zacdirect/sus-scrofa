.PHONY: help start stop status logs clean mongodb web processor dev run setup install check-deps venv reset-db fresh ai-setup ai-verify ai-clean

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
	@echo "$(GREEN)Ghiro Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make setup       - Complete setup (venv, deps, MongoDB, migrations)"
	@echo "  make fresh       - Fresh start (reset DBs, recreate everything)"
	@echo "  make venv        - Create Python virtual environment"
	@echo "  make install     - Install Python dependencies"
	@echo "  make check-deps  - Check system dependencies"
	@echo "  make mongodb     - Start MongoDB container (Podman)"
	@echo ""
	@echo "$(YELLOW)AI Detection (Optional):$(NC)"
	@echo "  make ai-setup    - Setup AI detection module (SPAI)"
	@echo "  make ai-verify   - Verify AI detection installation"
	@echo "  make ai-clean    - Remove AI detection environment"
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
	@echo "  make stop        - Stop all Ghiro services"
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
	@dpkg -l | grep -q "gir1.2-gexiv2-0.10" && echo "$(GREEN)✓ GExiv2 found$(NC)" || echo "$(YELLOW)⚠ GExiv2 not found (install gir1.2-gexiv2-0.10)$(NC)"
	@echo ""
	@dpkg -l | grep -q python3.13-dev && echo "$(GREEN)✓ Python dev headers found$(NC)" || echo "$(RED)✗ Python dev headers not found (install python3.13-dev)$(NC)"

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
	@echo "$(YELLOW)Installing PyGObject (for EXIF support)...$(NC)"
	@$(VENV)/pip install PyGObject || (echo "$(RED)PyGObject installation failed. Make sure you have:$(NC)" && \
		echo "  sudo apt-get install build-essential libcairo2-dev libgirepository-2.0-dev pkg-config python3.13-dev" && exit 1)
	@echo ""
	@echo "$(YELLOW)Verifying GExiv2 access from venv...$(NC)"
	@$(VENV)/python -c "from gi.repository import GExiv2; print('  ✓ GExiv2 version:', GExiv2.get_version())" 2>/dev/null || \
		(echo "$(RED)  ✗ Cannot import GExiv2. Make sure gir1.2-gexiv2-0.10 is installed:$(NC)" && \
		echo "    sudo apt-get install gir1.2-gexiv2-0.10" && exit 1)
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
	@if podman ps -a --format "{{.Names}}" | grep -q "^ghiro-mongodb$$"; then \
		if podman ps --format "{{.Names}}" | grep -q "^ghiro-mongodb$$"; then \
			echo "$(GREEN)✓ MongoDB container is already running$(NC)"; \
		else \
			echo "$(YELLOW)Starting existing MongoDB container...$(NC)"; \
			podman start ghiro-mongodb; \
		fi \
	else \
		echo "$(YELLOW)Creating new MongoDB container...$(NC)"; \
		podman run -d --name ghiro-mongodb \
			-p 27017:27017 \
			-v ghiro-mongodb-data:/data/db \
			docker.io/library/mongo:4.4; \
	fi
	@sleep 2
	@echo "$(GREEN)✓ MongoDB is ready$(NC)"

web: venv
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "$(RED)Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Starting Ghiro web server...$(NC)"
	@$(MANAGE) runserver 0.0.0.0:8000

processor: venv
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "$(RED)Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Starting Ghiro image processor...$(NC)"
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
	@echo "$(GREEN)Starting Ghiro in development mode...$(NC)"
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
	echo "  MongoDB:         Running on port 27017";

dev: run \
	echo ""; \
	echo "$(YELLOW)Press Ctrl+C to stop$(NC)"; \
	wait

status:
	@echo "$(GREEN)Ghiro Service Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)MongoDB Container:$(NC)"
	@if podman ps --format "{{.Names}}\t{{.Status}}" | grep ghiro-mongodb; then \
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

stop:
	@echo "$(YELLOW)Stopping Ghiro services...$(NC)"
	@-pkill -f "manage.py runserver" 2>/dev/null && echo "$(GREEN)✓ Web server stopped$(NC)" || echo "$(YELLOW)Web server not running$(NC)"
	@-pkill -f "manage.py process" 2>/dev/null && echo "$(GREEN)✓ Image processor stopped$(NC)" || echo "$(YELLOW)Image processor not running$(NC)"
	@echo "$(GREEN)✓ All services stopped$(NC)"
	@echo "$(YELLOW)Note: MongoDB container is still running. Use 'podman stop ghiro-mongodb' to stop it.$(NC)"

logs:
	@echo "$(GREEN)Recent MongoDB logs:$(NC)"
	@podman logs --tail 20 ghiro-mongodb 2>/dev/null || echo "$(RED)MongoDB container not found$(NC)"

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
	@echo "$(GREEN)Database reset complete!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Create superuser: make superuser"
	@echo "  2. Start development: make dev"
	@echo ""
	@echo "$(YELLOW)Tip: To also clear MongoDB analysis data:$(NC)"
	@echo "  podman rm -f ghiro-mongodb"
	@echo "  podman volume rm ghiro-mongodb-data"
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

# OpenCV Service (Container-based)
opencv-build:
	@echo "$(GREEN)Building OpenCV service container...$(NC)"
	@cd opencv_service && podman build -t ghiro-opencv:latest .
	@echo "$(GREEN)✓ OpenCV service image built$(NC)"

opencv-start:
	@echo "$(GREEN)Starting OpenCV service...$(NC)"
	@podman ps -a --filter name=ghiro-opencv -q | grep -q . && podman rm -f ghiro-opencv || true
	@podman run -d \
		--name ghiro-opencv \
		-p 8080:8080 \
		--network ghiro-net \
		ghiro-opencv:latest
	@echo "$(GREEN)✓ OpenCV service started on http://localhost:8080$(NC)"
	@echo ""
	@echo "Health check: curl http://localhost:8080/health"

opencv-stop:
	@echo "$(YELLOW)Stopping OpenCV service...$(NC)"
	@podman stop ghiro-opencv 2>/dev/null || true
	@podman rm ghiro-opencv 2>/dev/null || true
	@echo "$(GREEN)✓ OpenCV service stopped$(NC)"

opencv-logs:
	@podman logs -f ghiro-opencv

opencv-restart: opencv-stop opencv-start

opencv-shell:
	@podman exec -it ghiro-opencv /bin/bash

opencv-test:
	@echo "$(GREEN)Testing OpenCV service...$(NC)"
	@curl -s http://localhost:8080/health | python -m json.tool || echo "$(RED)Service not responding$(NC)"

opencv-clean: opencv-stop
	@echo "$(YELLOW)Removing OpenCV service image...$(NC)"
	@podman rmi ghiro-opencv:latest 2>/dev/null || true
	@echo "$(GREEN)✓ OpenCV service cleaned$(NC)" 2. Start development: make dev"
	@echo ""
	@echo "$(YELLOW)Tip: To also clear MongoDB analysis data:$(NC)"
	@echo "  podman rm -f ghiro-mongodb"
	@echo "  podman volume rm ghiro-mongodb-data"
	@echo ""
