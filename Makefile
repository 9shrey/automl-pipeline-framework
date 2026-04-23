# AutoML Pipeline Framework — developer one-liners.
# All targets assume `uv` is installed: https://docs.astral.sh/uv/

PY      ?= python
UV      ?= uv
PKG     := automl
SRC     := src/$(PKG)
TESTS   := tests

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  bootstrap         Create venv and install package + dev deps via uv"
	@echo "  install           Install package in editable mode"
	@echo "  fmt               Run black + ruff --fix"
	@echo "  lint              Run ruff + mypy"
	@echo "  test              Run unit tests with coverage"
	@echo "  test-fast         Run unit tests, skip slow/integration"
	@echo "  test-property     Run hypothesis property tests"
	@echo "  test-integration  Run integration tests"
	@echo "  cov               Run full test suite with coverage report"
	@echo "  demo-quickstart   Execute examples/01_quickstart.ipynb"
	@echo "  demo-warmstart    Execute examples/03_warm_start_demo.ipynb"
	@echo "  benchmark-fast    Run small OpenML benchmark"
	@echo "  benchmark-full    Run full OpenML benchmark"
	@echo "  clean             Remove build / cache / coverage artifacts"

.PHONY: bootstrap
bootstrap:
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"
	$(UV) run pre-commit install || true

.PHONY: install
install:
	$(UV) pip install -e ".[dev]"

.PHONY: fmt
fmt:
	$(UV) run ruff check --fix $(SRC) $(TESTS)
	$(UV) run black $(SRC) $(TESTS)

.PHONY: lint
lint:
	$(UV) run ruff check $(SRC) $(TESTS)
	$(UV) run mypy $(SRC)

.PHONY: test
test:
	$(UV) run pytest -q --cov=$(PKG) --cov-report=term-missing

.PHONY: test-fast
test-fast:
	$(UV) run pytest -q -m "not slow and not integration" $(TESTS)/unit

.PHONY: test-property
test-property:
	$(UV) run pytest -q $(TESTS)/property

.PHONY: test-integration
test-integration:
	$(UV) run pytest -q $(TESTS)/integration

.PHONY: cov
cov:
	$(UV) run pytest --cov=$(PKG) --cov-report=html --cov-report=term

.PHONY: demo-quickstart
demo-quickstart:
	$(UV) run jupyter nbconvert --to notebook --execute examples/01_quickstart.ipynb --output 01_quickstart.executed.ipynb

.PHONY: demo-warmstart
demo-warmstart:
	$(UV) run jupyter nbconvert --to notebook --execute examples/03_warm_start_demo.ipynb --output 03_warm_start_demo.executed.ipynb

.PHONY: benchmark-fast
benchmark-fast:
	$(UV) run python benchmarks/run_benchmark.py --config configs/fast.yaml

.PHONY: benchmark-full
benchmark-full:
	$(UV) run python benchmarks/run_benchmark.py --config configs/full.yaml

.PHONY: clean
clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
