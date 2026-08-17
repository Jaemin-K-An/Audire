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
        simulate eval eval-smoke caption-eval sensitivity model-compare reproduce run e2e \
        figures model asr-eval smoke clean package check

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

audit: ## Dependency vulnerability audit (fails on any known vulnerability)
	$(BIN)/python -m pip install -q pip-audit && $(BIN)/pip-audit --strict -r requirements.lock

test: ## Full test suite with branch coverage on the core modules
	$(PYTEST) --cov --cov-report=term-missing --cov-report=xml

test-fast: ## Test suite without slow/asr/e2e markers
	$(PYTEST) -m "not slow and not asr and not e2e"

check: lint typecheck test ## Everything CI runs

# --------------------------------------------------------------------------- research

simulate: ## Generate the synthetic cohorts declared in the main experiment config
	$(PYTHON) -m audire.cli simulate --config experiments/configs/rq1_main.yaml

eval: ## RQ1/RQ2/RQ3 — listener-level ablation, caption budgets and thresholds
	$(PYTHON) -m audire.cli evaluate --config experiments/configs/rq1_main.yaml

eval-smoke: ## Fast deterministic run used by CI
	$(PYTHON) -m audire.cli evaluate --config experiments/configs/smoke.yaml

caption-eval: ## RQ2/RQ3 — caption budget Pareto frontier and personalized thresholds
	$(PYTHON) -m audire.cli caption-eval --config experiments/configs/rq1_main.yaml

sensitivity: ## E11 — idiosyncrasy x calibration-length sweep
	$(PYTHON) -m audire.cli sensitivity --config experiments/configs/sensitivity.yaml

model-compare: ## E22 — candidate model families and calibration methods, listener-level
	$(PYTHON) -m audire.cli model-compare --config experiments/configs/e22_models.yaml

figures: ## Regenerate every figure and table from recorded experiment artifacts
	$(PYTHON) -m audire.cli figures --all

reproduce: ## Full research reproduction: eval -> sensitivity -> model-compare -> figures
	$(MAKE) eval
	$(MAKE) sensitivity
	$(MAKE) model-compare
	$(MAKE) figures
	@echo "reproduction complete; see docs/RESULTS.md and experiments/artifacts/"

# --------------------------------------------------------------------------- application

model: ## Fit the provenance-recorded synthetic deployment model into private/
	$(PYTHON) -m audire.cli build-model --config experiments/configs/rq1_main.yaml

asr-eval: ## E7 actual ASR WER/CER + timestamp regression on 10 fixed speakers
	AUDIRE_RUN_REAL_ASR=1 $(PYTHON) -m audire.cli asr-eval --config experiments/configs/asr_eval.yaml

run: ## Start the local API + web application (run `make model` first)
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
