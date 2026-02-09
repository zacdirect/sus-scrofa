# Makefile for AI Detection Model Evaluation
#
# Usage:
#   make baseline              - Test current detectors and save baseline
#   make new-candidate NAME=x  - Create new candidate directory
#   make test-candidate NAME=x - Test a specific candidate
#   make compare NAME=x        - Compare candidate with baseline
#   make compare-all           - Compare all candidates
#   make integrate NAME=x      - Integrate candidate into main codebase
#   make clean                 - Remove generated files (keep results)
#   make clean-all             - Remove everything including results
#   make check-test-data       - Check test data setup

SHELL := /bin/bash
.PHONY: help baseline new-candidate test-candidate compare compare-all integrate clean clean-all setup-test-data check-test-data

# Configuration
PYTHON := python3
VENV_DIR := ../.venv
VENV_PYTHON := $(VENV_DIR)/bin/python
TEST_DATA_DIR := test_data
BASELINE_DIR := baseline
CANDIDATES_DIR := candidates
REPORTS_DIR := reports
FRAMEWORK_DIR := framework

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)AI Detection Model Evaluation$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make setup-test-data          Link test data directory"
	@echo "  make baseline                 Create baseline from current detectors"
	@echo ""
	@echo "$(GREEN)Evaluation:$(NC)"
	@echo "  make new-candidate NAME=x     Create new candidate model directory"
	@echo "  make test-candidate NAME=x    Test specific candidate model"
	@echo "  make compare NAME=x           Compare candidate with baseline"
	@echo "  make compare-all              Compare all candidates with baseline"
	@echo ""
	@echo "$(GREEN)Integration:$(NC)"
	@echo "  make integrate NAME=x         Integrate candidate into main codebase"
	@echo "  make validate NAME=x          Validate detector API compliance"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean                    Remove generated files (keep results)"
	@echo "  make clean-all                Remove everything including results"
	@echo "  make stats                    Show statistics for all models"
	@echo ""
	@echo "$(YELLOW)Example workflow:$(NC)"
	@echo "  1. make setup-test-data              # Link your test images"
	@echo "  2. make baseline                     # Establish current performance"
	@echo "  3. make new-candidate NAME=mymodel   # Create new model directory"
	@echo "  4. # Edit candidates/mymodel/detector.py"
	@echo "  5. make test-candidate NAME=mymodel  # Test your model"
	@echo "  6. make compare NAME=mymodel         # Compare with baseline"
	@echo "  7. make integrate NAME=mymodel       # If better, integrate it"

# ============================================================================
# Setup
# ============================================================================

setup-test-data:
	@echo "$(BLUE)Setting up test data directory...$(NC)"
	@if [ ! -d "$(TEST_DATA_DIR)" ]; then \
		echo "$(YELLOW)Test data directory not found.$(NC)"; \
		echo "Please copy your test images into $(TEST_DATA_DIR)/:"; \
		echo ""; \
		echo "  mkdir -p $(TEST_DATA_DIR)/{fake,edited,real}"; \
		echo "  cp /path/to/ai-generated/*.jpg $(TEST_DATA_DIR)/fake/"; \
		echo "  cp /path/to/edited/*.jpg       $(TEST_DATA_DIR)/edited/"; \
		echo "  cp /path/to/real-photos/*.jpg   $(TEST_DATA_DIR)/real/"; \
		echo ""; \
		echo "Required structure:"; \
		echo "  $(TEST_DATA_DIR)/"; \
		echo "  ├── fake/     # AI-generated images"; \
		echo "  ├── edited/   # Human-edited images"; \
		echo "  └── real/     # Authentic photos"; \
		echo ""; \
		echo "The directory is git-ignored so your test images stay local."; \
		exit 1; \
	fi
	@if [ ! -d "$(TEST_DATA_DIR)/fake" ] || [ ! -d "$(TEST_DATA_DIR)/real" ]; then \
		echo "$(RED)Error: Test data incomplete$(NC)"; \
		echo "Required subdirectories: fake/, edited/, real/"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Test data structure validated$(NC)"
	@echo "  Fake images:   $$(find $(TEST_DATA_DIR)/fake -type f | wc -l)"
	@echo "  Edited images: $$(find $(TEST_DATA_DIR)/edited -type f 2>/dev/null | wc -l)"
	@echo "  Real images:   $$(find $(TEST_DATA_DIR)/real -type f | wc -l)"

$(BASELINE_DIR):
	@mkdir -p $(BASELINE_DIR)

$(CANDIDATES_DIR):
	@mkdir -p $(CANDIDATES_DIR)

$(REPORTS_DIR):
	@mkdir -p $(REPORTS_DIR)

$(FRAMEWORK_DIR):
	@mkdir -p $(FRAMEWORK_DIR)

# ============================================================================
# Baseline Testing
# ============================================================================

baseline: setup-test-data $(BASELINE_DIR)
	@echo "$(BLUE)Running baseline tests on current detectors...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		$(VENV_PYTHON) -m framework.baseline_runner \
			--test-data $(TEST_DATA_DIR) \
			--output $(BASELINE_DIR) \
			--detectors metadata spai
	@echo "$(GREEN)✓ Baseline results saved to $(BASELINE_DIR)/$(NC)"
	@echo ""
	@$(MAKE) --no-print-directory show-baseline

show-baseline:
	@if [ -f "$(BASELINE_DIR)/summary.json" ]; then \
		echo "$(BLUE)Baseline Summary:$(NC)"; \
		$(VENV_PYTHON) -m framework.show_results $(BASELINE_DIR)/summary.json; \
	else \
		echo "$(YELLOW)No baseline results found. Run 'make baseline' first.$(NC)"; \
	fi

# ============================================================================
# Candidate Model Management
# ============================================================================

new-candidate: $(CANDIDATES_DIR)
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME not specified$(NC)"; \
		echo "Usage: make new-candidate NAME=my_model"; \
		exit 1; \
	fi
	@if [ -d "$(CANDIDATES_DIR)/$(NAME)" ]; then \
		echo "$(RED)Error: Candidate $(NAME) already exists$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Creating new candidate: $(NAME)$(NC)"
	@mkdir -p $(CANDIDATES_DIR)/$(NAME)/{weights,tests}
	@cp framework/templates/detector_template.py $(CANDIDATES_DIR)/$(NAME)/detector.py
	@cp framework/templates/requirements_template.txt $(CANDIDATES_DIR)/$(NAME)/requirements.txt
	@cp framework/templates/setup_template.sh $(CANDIDATES_DIR)/$(NAME)/setup.sh
	@chmod +x $(CANDIDATES_DIR)/$(NAME)/setup.sh
	@echo "# $(NAME)" > $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "## Model Information" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "- **Source**: [Add link]" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "- **Paper**: [Add citation]" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "- **License**: [Add license]" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "## Setup" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "\`\`\`bash" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "./setup.sh" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "\`\`\`" >> $(CANDIDATES_DIR)/$(NAME)/README.md
	@echo "$(GREEN)✓ Created candidate directory: $(CANDIDATES_DIR)/$(NAME)/$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Edit $(CANDIDATES_DIR)/$(NAME)/detector.py"
	@echo "  2. Add dependencies to $(CANDIDATES_DIR)/$(NAME)/requirements.txt"
	@echo "  3. Update $(CANDIDATES_DIR)/$(NAME)/setup.sh for model setup"
	@echo "  4. Run: make test-candidate NAME=$(NAME)"

validate: $(FRAMEWORK_DIR)
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME not specified$(NC)"; \
		echo "Usage: make validate NAME=my_model"; \
		exit 1; \
	fi
	@if [ ! -d "$(CANDIDATES_DIR)/$(NAME)" ]; then \
		echo "$(RED)Error: Candidate $(NAME) not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Validating detector API for $(NAME)...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		$(VENV_PYTHON) -m framework.validator \
			--detector $(CANDIDATES_DIR)/$(NAME)/detector.py
	@echo "$(GREEN)✓ Validation complete$(NC)"

test-candidate: setup-test-data
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME not specified$(NC)"; \
		echo "Usage: make test-candidate NAME=my_model"; \
		exit 1; \
	fi
	@if [ ! -d "$(CANDIDATES_DIR)/$(NAME)" ]; then \
		echo "$(RED)Error: Candidate $(NAME) not found$(NC)"; \
		echo "Run: make new-candidate NAME=$(NAME)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Testing candidate: $(NAME)$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		$(VENV_PYTHON) -m framework.candidate_tester \
			--name $(NAME) \
			--detector $(CANDIDATES_DIR)/$(NAME)/detector.py \
			--test-data $(TEST_DATA_DIR) \
			--output $(CANDIDATES_DIR)/$(NAME)/test_results.json
	@echo "$(GREEN)✓ Test results saved to $(CANDIDATES_DIR)/$(NAME)/test_results.json$(NC)"
	@echo ""
	@$(MAKE) --no-print-directory show-results NAME=$(NAME)

show-results:
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME not specified$(NC)"; \
		exit 1; \
	fi
	@if [ -f "$(CANDIDATES_DIR)/$(NAME)/test_results.json" ]; then \
		echo "$(BLUE)Results for $(NAME):$(NC)"; \
		$(VENV_PYTHON) -m framework.show_results $(CANDIDATES_DIR)/$(NAME)/test_results.json; \
	else \
		echo "$(YELLOW)No results found for $(NAME). Run 'make test-candidate NAME=$(NAME)' first.$(NC)"; \
	fi

# ============================================================================
# Comparison
# ============================================================================

compare: $(REPORTS_DIR)
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME not specified$(NC)"; \
		echo "Usage: make compare NAME=my_model"; \
		exit 1; \
	fi
	@if [ ! -f "$(BASELINE_DIR)/summary.json" ]; then \
		echo "$(RED)Error: No baseline found$(NC)"; \
		echo "Run: make baseline"; \
		exit 1; \
	fi
	@if [ ! -f "$(CANDIDATES_DIR)/$(NAME)/test_results.json" ]; then \
		echo "$(RED)Error: No test results for $(NAME)$(NC)"; \
		echo "Run: make test-candidate NAME=$(NAME)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Comparing $(NAME) with baseline...$(NC)"
	@source $(VENV_DIR)/bin/activate && \
		$(VENV_PYTHON) -m framework.comparator \
			--baseline $(BASELINE_DIR)/summary.json \
			--candidate $(CANDIDATES_DIR)/$(NAME)/test_results.json \
			--output $(REPORTS_DIR)/compare_$(NAME)_$$(date +%Y%m%d_%H%M%S).md
	@echo "$(GREEN)✓ Comparison report generated$(NC)"

compare-all: $(REPORTS_DIR)
	@if [ ! -f "$(BASELINE_DIR)/summary.json" ]; then \
		echo "$(RED)Error: No baseline found$(NC)"; \
		echo "Run: make baseline"; \
		exit 1; \
	fi
	@echo "$(BLUE)Comparing all candidates with baseline...$(NC)"
	@CANDIDATES=$$(find $(CANDIDATES_DIR) -name "test_results.json" | xargs dirname | xargs basename -a); \
	if [ -z "$$CANDIDATES" ]; then \
		echo "$(YELLOW)No candidates with test results found.$(NC)"; \
		exit 0; \
	fi; \
	source $(VENV_DIR)/bin/activate && \
		$(VENV_PYTHON) -m framework.comparator_all \
			--baseline $(BASELINE_DIR)/summary.json \
			--candidates $(CANDIDATES_DIR)/*/test_results.json \
			--output $(REPORTS_DIR)/comparison_all_$$(date +%Y%m%d_%H%M%S).md
	@echo "$(GREEN)✓ Comparison report generated in $(REPORTS_DIR)/$(NC)"

stats:
	@echo "$(BLUE)Model Evaluation Statistics$(NC)"
	@echo ""
	@if [ -f "$(BASELINE_DIR)/summary.json" ]; then \
		echo "$(GREEN)Baseline:$(NC)"; \
		$(VENV_PYTHON) -m framework.show_results $(BASELINE_DIR)/summary.json --compact; \
		echo ""; \
	fi
	@echo "$(GREEN)Candidates:$(NC)"
	@FOUND=0; \
	for result in $(CANDIDATES_DIR)/*/test_results.json; do \
		if [ -f "$$result" ]; then \
			MODEL=$$(dirname $$result | xargs basename); \
			echo "  $(YELLOW)$$MODEL:$(NC)"; \
			$(VENV_PYTHON) -m framework.show_results $$result --compact --indent 4; \
			echo ""; \
			FOUND=1; \
		fi; \
	done; \
	if [ $$FOUND -eq 0 ]; then \
		echo "  $(YELLOW)No candidate results found.$(NC)"; \
	fi

# ============================================================================
# Integration
# ============================================================================

integrate:
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME not specified$(NC)"; \
		echo "Usage: make integrate NAME=my_model"; \
		exit 1; \
	fi
	@if [ ! -d "$(CANDIDATES_DIR)/$(NAME)" ]; then \
		echo "$(RED)Error: Candidate $(NAME) not found$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(CANDIDATES_DIR)/$(NAME)/test_results.json" ]; then \
		echo "$(RED)Error: No test results for $(NAME)$(NC)"; \
		echo "Test the model first: make test-candidate NAME=$(NAME)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Integrating $(NAME) into main codebase...$(NC)"
	@echo ""
	@echo "$(YELLOW)This will:$(NC)"
	@echo "  1. Copy detector to ../detectors/$(NAME)_detector.py"
	@echo "  2. Copy weights to ../weights/$(NAME)/"
	@echo "  3. Add to orchestrator detector list"
	@echo ""
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ ! $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(YELLOW)Integration cancelled$(NC)"; \
		exit 0; \
	fi
	@# Copy detector
	@cp $(CANDIDATES_DIR)/$(NAME)/detector.py ../detectors/$(NAME)_detector.py
	@echo "$(GREEN)✓$(NC) Copied detector"
	@# Copy weights if they exist
	@if [ -d "$(CANDIDATES_DIR)/$(NAME)/weights" ] && [ "$$(ls -A $(CANDIDATES_DIR)/$(NAME)/weights)" ]; then \
		mkdir -p ../weights/$(NAME); \
		cp -r $(CANDIDATES_DIR)/$(NAME)/weights/* ../weights/$(NAME)/; \
		echo "$(GREEN)✓$(NC) Copied weights"; \
	fi
	@# Update orchestrator (manual step reminder)
	@echo ""
	@echo "$(YELLOW)Manual steps remaining:$(NC)"
	@echo "  1. Edit ../detectors/orchestrator.py"
	@echo "  2. Import: from .$(NAME)_detector import $(shell echo $(NAME) | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++) $$i=toupper(substr($$i,1,1)) tolower(substr($$i,2))}1' | sed 's/ //g')Detector"
	@echo "  3. Add to _load_detectors(): self._detectors.append($(shell echo $(NAME) | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++) $$i=toupper(substr($$i,1,1)) tolower(substr($$i,2))}1' | sed 's/ //g')Detector())"
	@echo "  4. Update requirements.txt with model dependencies"
	@echo "  5. Test full integration: make -C ../.. test"
	@echo ""
	@echo "$(GREEN)✓ Integration prepared$(NC)"

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	@find $(CANDIDATES_DIR) -name "*.pyc" -delete
	@find $(CANDIDATES_DIR) -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find $(FRAMEWORK_DIR) -name "*.pyc" -delete
	@find $(FRAMEWORK_DIR) -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-all: clean
	@echo "$(BLUE)Removing all results and reports...$(NC)"
	@rm -rf $(BASELINE_DIR)/*.json
	@rm -rf $(CANDIDATES_DIR)/*/test_results.json
	@rm -rf $(REPORTS_DIR)/*
	@echo "$(GREEN)✓ All results removed$(NC)"
	@echo "$(YELLOW)Note: Test data, weights, and detector code preserved$(NC)"

clean-candidate:
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME not specified$(NC)"; \
		echo "Usage: make clean-candidate NAME=my_model"; \
		exit 1; \
	fi
	@if [ ! -d "$(CANDIDATES_DIR)/$(NAME)" ]; then \
		echo "$(YELLOW)Candidate $(NAME) not found$(NC)"; \
		exit 0; \
	fi
	@echo "$(BLUE)Removing candidate: $(NAME)$(NC)"
	@rm -rf $(CANDIDATES_DIR)/$(NAME)
	@echo "$(GREEN)✓ Removed $(NAME)$(NC)"

# ============================================================================
# Development Helpers
# ============================================================================

.PHONY: list-candidates list-reports watch

list-candidates:
	@echo "$(BLUE)Available Candidates:$(NC)"
	@if [ -d "$(CANDIDATES_DIR)" ]; then \
		for dir in $(CANDIDATES_DIR)/*; do \
			if [ -d "$$dir" ]; then \
				NAME=$$(basename $$dir); \
				if [ -f "$$dir/test_results.json" ]; then \
					echo "  $(GREEN)✓$(NC) $$NAME (tested)"; \
				else \
					echo "  $(YELLOW)○$(NC) $$NAME (not tested)"; \
				fi; \
			fi; \
		done; \
	else \
		echo "  $(YELLOW)No candidates directory$(NC)"; \
	fi

list-reports:
	@echo "$(BLUE)Available Reports:$(NC)"
	@if [ -d "$(REPORTS_DIR)" ]; then \
		ls -1t $(REPORTS_DIR)/*.md 2>/dev/null | head -10 || echo "  $(YELLOW)No reports found$(NC)"; \
	else \
		echo "  $(YELLOW)No reports directory$(NC)"; \
	fi

watch:
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME not specified$(NC)"; \
		echo "Usage: make watch NAME=my_model"; \
		exit 1; \
	fi
	@echo "$(BLUE)Watching $(NAME) for changes...$(NC)"
	@echo "Press Ctrl+C to stop"
	@while true; do \
		inotifywait -q -e modify $(CANDIDATES_DIR)/$(NAME)/detector.py 2>/dev/null && \
		echo "" && \
		echo "$(YELLOW)Change detected, re-testing...$(NC)" && \
		$(MAKE) --no-print-directory test-candidate NAME=$(NAME); \
	done

check-test-data:
	@echo "$(BLUE)Checking test data setup...$(NC)"
	@echo ""
	@if [ -d "$(TEST_DATA_DIR)/fake" ]; then \
		count=$$(ls -1 $(TEST_DATA_DIR)/fake/ 2>/dev/null | wc -l); \
		echo "Fake images: $$count"; \
		if [ $$count -lt 10 ]; then \
			echo "  $(YELLOW)Warning: Recommended minimum is 10 images$(NC)"; \
		fi; \
	else \
		echo "$(RED)Error: $(TEST_DATA_DIR)/fake/ directory not found$(NC)"; \
	fi
	@if [ -d "$(TEST_DATA_DIR)/edited" ]; then \
		count=$$(ls -1 $(TEST_DATA_DIR)/edited/ 2>/dev/null | wc -l); \
		echo "Edited images: $$count"; \
		if [ $$count -lt 5 ]; then \
			echo "  $(YELLOW)Warning: Recommended minimum is 5 images$(NC)"; \
		fi; \
	else \
		echo "$(RED)Error: $(TEST_DATA_DIR)/edited/ directory not found$(NC)"; \
	fi
	@if [ -d "$(TEST_DATA_DIR)/real" ]; then \
		count=$$(ls -1 $(TEST_DATA_DIR)/real/ 2>/dev/null | wc -l); \
		echo "Real images: $$count"; \
		if [ $$count -lt 5 ]; then \
			echo "  $(YELLOW)Warning: Recommended minimum is 5 images$(NC)"; \
		fi; \
	else \
		echo "$(RED)Error: $(TEST_DATA_DIR)/real/ directory not found$(NC)"; \
	fi
	@echo ""
	@echo "See $(TEST_DATA_DIR)/README.md for setup instructions"
