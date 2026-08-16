# AUDIRE — one-command entrypoints.
# Every target is safe to re-run. Nothing here writes participant data into the repo.

SHELL := /bin/bash
PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin
PYTHON := $(BIN)/python
PYTEST := $(BIN)/pytest
HOST ?= 127.0.0.1
PORT ?= 8000

.DEFAULT_GOAL := help
.PHONY: help bootstrap lock data data-verify test test-fast lint typecheck audit \
        simulate train eval caption-eval reproduce run e2e figures clean package \
        check gates

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- setup

$(VENV)/pyvenv.cfg:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip setuptools wheel

bootstrap: $(VENV)/pyvenv.cfg ## Create the venv and install pinned dependencies
	$(BIN)/python -m pip install -e ".[api,data,dev]"
	@echo "bootstrap complete. ASR extras are optional: make bootstrap-asr"

bootstrap-asr: $(VENV)/pyvenv.cfg ## Additionally install the ASR backend (large download)
	$(BIN)/python -m pip install -e ".[asr]"

bootstrap-e2e: $(VENV)/pyvenv.cfg ## Additionally install browser E2E tooling
	$(BIN)/python -m pip install -e ".[e2e]"
	$(BIN)/playwright install chromium

lock: ## Regenerate the dependency lockfile from the current venv
	$(BIN)/python -m pip freeze --exclude-editable > requirements.lock
	@echo "wrote requirements.lock"

# --------------------------------------------------------------------------- data

data: ## Fetch every source that does not need an outstanding human step
	$(PYTHON) scripts/fetch_data.py all-permitted

data-list: ## Show source registry and acknowledgement status
	$(PYTHON) scripts/fetch_data.py list

data-primary: ## Fetch the CC BY-NC-ND primary corpus (needs AUDIRE_PRIMARY_DATA_USE_NOTIFIED=1)
	$(PYTHON) scripts/fetch_data.py korean_monosyllabic_speech

data-verify: ## Re-check every manifest against local files
	$(PYTHON) scripts/fetch_data.py verify

# --------------------------------------------------------------------------- quality

lint: ## Format check + lint
	$(BIN)/ruff format --check .
	$(BIN)/ruff check .

format: ## Apply formatting and safe lint fixes
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: ## Strict type checking
	$(BIN)/mypy

audit: ## Dependency vulnerability audit
	$(BIN)/python -m pip install -q pip-audit && $(BIN)/pip-audit --strict || true

test: ## Full test suite with branch coverage on the core modules
	$(PYTEST) --cov --cov-report=term-missing --cov-report=xml

test-fast: ## Test suite without slow/asr/e2e markers
	$(PYTEST) -m "not slow and not asr and not e2e"

check: lint typecheck test ## Everything CI runs

# --------------------------------------------------------------------------- research

simulate: ## Generate the synthetic listener/trial cohorts declared in experiments/configs
	$(PYTHON) -m audire.cli simulate --config experiments/configs/simulation_main.yaml

train: ## Fit the risk models on the pinned cohort
	$(PYTHON) -m audire.cli train --config experiments/configs/model_main.yaml

eval: ## RQ1 — listener-level ablation with calibration metrics
	$(PYTHON) -m audire.cli evaluate --config experiments/configs/ablation_rq1.yaml

caption-eval: ## RQ2/RQ3 — caption budget Pareto and personalized thresholds
	$(PYTHON) -m audire.cli caption-eval --config experiments/configs/caption_rq2.yaml

sensitivity: ## E5/E8 — calibration length, SNR and subgroup sweeps
	$(PYTHON) -m audire.cli sensitivity --config experiments/configs/sensitivity.yaml

figures: ## Regenerate every figure and table from recorded experiment artifacts
	$(PYTHON) -m audire.cli figures --all

reproduce: ## Full research reproduction: simulate -> train -> eval -> caption -> figures
	$(MAKE) simulate
	$(MAKE) train
	$(MAKE) eval
	$(MAKE) caption-eval
	$(MAKE) sensitivity
	$(MAKE) figures
	@echo "reproduction complete; see docs/RESULTS.md and experiments/artifacts/"

# --------------------------------------------------------------------------- application

run: ## Start the local API + web application
	$(BIN)/uvicorn apps.api.main:app --host $(HOST) --port $(PORT)

run-dev: ## Start with autoreload
	$(BIN)/uvicorn apps.api.main:app --host $(HOST) --port $(PORT) --reload

e2e: ## Browser end-to-end tests (requires: make bootstrap-e2e)
	$(PYTEST) tests/e2e -m e2e

smoke: ## CPU-only end-to-end pipeline smoke test
	$(PYTEST) tests/integration/test_pipeline_smoke.py -q

# --------------------------------------------------------------------------- misc

package: ## Build the distributable wheel and sdist
	$(BIN)/python -m pip install -q build && $(BIN)/python -m build

clean: ## Remove caches and regenerable artifacts (never touches data/raw or private/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@echo "clean complete (data/raw, data/processed and private/ were NOT touched)"
