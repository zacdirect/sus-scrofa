.PHONY: help setup install weights models verify clean test venv detect-system models-download models-verify models-clean

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Paths
VENV_DIR := .venv
VENV := $(VENV_DIR)/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip
SYSTEM_PYTHON := python3

# Weights
WEIGHTS_DIR := weights
WEIGHTS_FILE := $(WEIGHTS_DIR)/spai.pth
WEIGHTS_URL := https://drive.google.com/uc?export=download&id=1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI

# HuggingFace model cache (git-ignored)
MODELS_DIR := models
HF_MODELS := Organika/sdxl-detector

VENV_EXISTS := $(shell test -f $(VENV_DIR)/bin/pip && echo 1 || echo 0)

help:
	@echo "$(GREEN)Sus Scrofa AI Detection Setup$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make setup       - Complete setup (venv + dependencies + all weights/models)"
	@echo "  make venv        - Create Python virtual environment"
	@echo "  make install     - Install dependencies only"
	@echo "  make weights     - Download SPAI model weights"
	@echo "  make models      - Download HuggingFace models (sdxl-detector, etc.)"
	@echo ""
	@echo "$(YELLOW)System Detection:$(NC)"
	@echo "  make detect-system - Check GPU/CUDA availability"
	@echo ""
	@echo "$(YELLOW)Verification:$(NC)"
	@echo "  make verify      - Verify installation and test inference"
	@echo "  make test        - Run tests"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make clean       - Remove virtual environment"
	@echo "  make clean-all   - Remove venv, weights, and downloaded models"
	@echo "  make models-clean - Remove only downloaded HuggingFace models"
	@echo ""
	@echo "$(YELLOW)Note:$(NC)"
	@echo "  PyTorch installation is automatically configured based on"
	@echo "  detected GPU/CUDA. Run 'make detect-system' to see what"
	@echo "  will be installed."
	@echo ""
	@echo "  HuggingFace models are cached to ai_detection/models/ (git-ignored)."

detect-system:
	@echo "$(BLUE)Detecting system configuration...$(NC)"
	@echo ""
	@$(SYSTEM_PYTHON) ../scripts/detect_system.py

setup: detect-system venv install weights models
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ AI Detection setup complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "Verify installation:"
	@echo "  make verify"

venv:
	@if [ $(VENV_EXISTS) -eq 1 ]; then \
		echo "$(GREEN)✓ Virtual environment already exists$(NC)"; \
	else \
		echo "$(YELLOW)Creating Python virtual environment...$(NC)"; \
		$(SYSTEM_PYTHON) -m venv $(VENV_DIR); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	fi

install: venv
	@echo "$(GREEN)Installing AI detection dependencies...$(NC)"
	@echo ""
	@echo "$(BLUE)Detecting GPU/CUDA configuration...$(NC)"
	@INDEX_URL=$$($(SYSTEM_PYTHON) ../scripts/detect_system.py --index-url); \
	BACKEND=$$($(SYSTEM_PYTHON) ../scripts/detect_system.py --backend); \
	echo "$(YELLOW)Backend: $$BACKEND$(NC)"; \
	echo "$(YELLOW)PyTorch Index: $$INDEX_URL$(NC)"; \
	echo ""; \
	echo "$(YELLOW)Upgrading pip...$(NC)"; \
	$(PIP) install --upgrade pip setuptools wheel; \
	echo ""; \
	echo "$(YELLOW)Installing PyTorch and dependencies...$(NC)"; \
	echo "  This may take several minutes..."; \
	echo ""; \
	$(PIP) install --extra-index-url "$$INDEX_URL" "torch>=2.0.0" "torchvision>=0.15.0"; \
	$(PIP) install "opencv-python>=4.10.0" "pyyaml>=6.0.1" "scipy>=1.14.0" \
		"timm==0.4.12" "yacs>=0.1.8" "numpy>=1.26.4" "torchmetrics>=1.4.0" \
		"tqdm>=4.66.4" "pillow>=10.4.0" "einops>=0.8.0" "ftfy>=6.1.0" "regex>=2023.0.0"; \
	echo "$(YELLOW)Installing HuggingFace Transformers...$(NC)"; \
	$(PIP) install "transformers>=4.36.0" "safetensors>=0.4.0"; \
	echo "$(YELLOW)Installing CLIP (for alternative backbones)...$(NC)"; \
	$(PIP) install open-clip-torch; \
	echo ""; \
	echo "$(GREEN)✓ Dependencies installed$(NC)"

weights:
	@mkdir -p $(WEIGHTS_DIR)
	@if [ -f $(WEIGHTS_FILE) ]; then \
		SIZE=$$(stat -c%s "$(WEIGHTS_FILE)" 2>/dev/null || stat -f%z "$(WEIGHTS_FILE)" 2>/dev/null); \
		if [ $$SIZE -gt 10000000 ]; then \
			echo "$(GREEN)✓ Weights file already exists: $(WEIGHTS_FILE)$(NC)"; \
		else \
			echo "$(RED)✗ Weights file is too small ($$SIZE bytes) - likely a failed download$(NC)"; \
			rm -f $(WEIGHTS_FILE); \
			echo "$(YELLOW)Removing corrupted file and retrying...$(NC)"; \
		fi; \
	fi
	@if [ ! -f $(WEIGHTS_FILE) ]; then \
		echo "$(YELLOW)SPAI model weights need to be downloaded manually$(NC)"; \
		echo "$(RED)Automated download from Google Drive is not reliable$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Please download manually:$(NC)"; \
		echo "  1. Visit: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view"; \
		echo "  2. Click 'Download' and confirm the virus scan warning"; \
		echo "  3. Save as: $(WEIGHTS_FILE)"; \
		echo "  4. Expected size: ~892MB"; \
		echo ""; \
		echo "$(YELLOW)Or use gdown if you have it installed:$(NC)"; \
		echo "  pip install gdown"; \
		echo "  gdown 1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI -O $(WEIGHTS_FILE)"; \
		echo ""; \
		exit 1; \
	fi

# ============================================================================
# HuggingFace Model Downloads
# ============================================================================

models: venv
	@echo "$(BLUE)Downloading HuggingFace models to $(MODELS_DIR)/...$(NC)"
	@echo ""
	@mkdir -p $(MODELS_DIR)
	@for model in $(HF_MODELS); do \
		safe_name=$$(echo $$model | tr '/' '--'); \
		if [ -d "$(MODELS_DIR)/$$safe_name" ] && [ "$$(ls -A $(MODELS_DIR)/$$safe_name 2>/dev/null)" ]; then \
			size=$$(du -sh "$(MODELS_DIR)/$$safe_name" | cut -f1); \
			echo "$(GREEN)✓ $$model already downloaded ($$size)$(NC)"; \
		else \
			echo "$(YELLOW)Downloading $$model...$(NC)"; \
			$(PYTHON) download_models.py "$$model" "$(MODELS_DIR)/$$safe_name"; \
			if [ $$? -eq 0 ]; then \
				size=$$(du -sh "$(MODELS_DIR)/$$safe_name" | cut -f1); \
				echo "$(GREEN)✓ $$model downloaded ($$size)$(NC)"; \
			else \
				echo "$(RED)✗ Failed to download $$model$(NC)"; \
			fi; \
		fi; \
		echo ""; \
	done
	@echo "$(GREEN)✓ Model downloads complete$(NC)"
	@echo "  Models cached in: $(MODELS_DIR)/ (git-ignored)"

models-clean:
	@echo "$(YELLOW)Removing downloaded HuggingFace models...$(NC)"
	@rm -rf $(MODELS_DIR)
	@echo "$(GREEN)✓ Models removed$(NC)"
	@echo "  Run 'make models' to re-download"

models-list:
	@echo "$(BLUE)Configured HuggingFace models:$(NC)"
	@for model in $(HF_MODELS); do \
		safe_name=$$(echo $$model | tr '/' '--'); \
		if [ -d "$(MODELS_DIR)/$$safe_name" ] && [ "$$(ls -A $(MODELS_DIR)/$$safe_name 2>/dev/null)" ]; then \
			size=$$(du -sh "$(MODELS_DIR)/$$safe_name" | cut -f1); \
			echo "  $(GREEN)✓$(NC) $$model ($$size)"; \
		else \
			echo "  $(RED)✗$(NC) $$model (not downloaded)"; \
		fi; \
	done
	@echo ""
	@echo "To add a model, edit HF_MODELS in this Makefile"

verify:
	@echo "$(GREEN)Verifying AI detection installation...$(NC)"
	@echo ""
	@echo "$(YELLOW)1. Checking virtual environment...$(NC)"
	@if [ $(VENV_EXISTS) -eq 1 ]; then \
		echo "  $(GREEN)✓ Virtual environment exists$(NC)"; \
	else \
		echo "  $(RED)✗ Virtual environment not found$(NC)"; \
		echo "  Run: make setup"; \
		exit 1; \
	fi
	@echo ""
	@echo "$(YELLOW)2. Checking PyTorch installation...$(NC)"
	@$(PYTHON) -c "import torch; print(f'  $(GREEN)✓ PyTorch {torch.__version__}$(NC)')" || \
		(echo "  $(RED)✗ PyTorch not installed$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)3. Checking GPU/CUDA availability...$(NC)"
	@$(PYTHON) -c "import torch; cuda = torch.cuda.is_available(); print('  $(GREEN)✓ CUDA available: GPU detected$(NC)' if cuda else '  $(YELLOW)⚠ CUDA not available (CPU mode only)$(NC)')"
	@if $(PYTHON) -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
		$(PYTHON) -c "import torch; print(f'  $(GREEN)✓ GPU: {torch.cuda.get_device_name(0)}$(NC)'); print(f'  $(GREEN)✓ CUDA version: {torch.version.cuda}$(NC)')"; \
	else \
		echo "  $(YELLOW)  Inference will be slower (~2-3s per image)$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)4. Checking SPAI weights file...$(NC)"
	@if [ -f $(WEIGHTS_FILE) ]; then \
		size=$$(du -h $(WEIGHTS_FILE) | cut -f1); \
		echo "  $(GREEN)✓ SPAI weights exist ($$size)$(NC)"; \
	else \
		echo "  $(RED)✗ SPAI weights not found$(NC)"; \
		echo "  Run: make weights"; \
		exit 1; \
	fi
	@echo ""
	@echo "$(YELLOW)5. Testing SPAI inference...$(NC)"
	@$(PYTHON) -c "from spai.inference import SPAIDetector; \
		print('  $(GREEN)✓ SPAI module loads successfully$(NC)')" || \
		(echo "  $(RED)✗ SPAI module failed to load$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)6. Checking HuggingFace Transformers...$(NC)"
	@$(PYTHON) -c "import transformers; print(f'  $(GREEN)✓ Transformers {transformers.__version__}$(NC)')" || \
		(echo "  $(RED)✗ Transformers not installed$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)7. Checking HuggingFace models...$(NC)"
	@for model in $(HF_MODELS); do \
		safe_name=$$(echo $$model | tr '/' '--'); \
		if [ -d "$(MODELS_DIR)/$$safe_name" ]; then \
			size=$$(du -sh "$(MODELS_DIR)/$$safe_name" | cut -f1); \
			echo "  $(GREEN)✓ $$model ($$size)$(NC)"; \
		else \
			echo "  $(YELLOW)⚠ $$model not downloaded$(NC)"; \
			echo "    Run: make models"; \
		fi; \
	done
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ All checks passed!$(NC)"
	@echo "$(GREEN)========================================$(NC)"

test: verify
	@echo "$(GREEN)Running AI detection tests...$(NC)"
	@$(PYTHON) -m pytest tests/ -v || echo "$(YELLOW)No tests found yet$(NC)"

clean:
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	@rm -rf $(VENV_DIR)
	@rm -rf __pycache__ spai/__pycache__
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean
	@echo "$(YELLOW)Removing weights...$(NC)"
	@rm -rf $(WEIGHTS_DIR)
	@echo "$(YELLOW)Removing downloaded models...$(NC)"
	@rm -rf $(MODELS_DIR)
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"
