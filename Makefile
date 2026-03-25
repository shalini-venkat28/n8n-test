.PHONY: setup run test test-cov lint lint-fix format check clean help

setup: ## Install development dependencies
	@echo "Installing dependencies..."
	uv sync --all-extras

run: ## Start the application
	@echo "Starting Neural Hub API..."
	uv run uvicorn neural_hub.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run pytest test suite
	@echo "Running tests..."
	uv run pytest -v

test-cov: ## Run tests with coverage report
	@clear || cls
	@echo "Running tests with coverage..."
	uv run coverage run -m pytest -v
	uv run coverage report
	uv run coverage xml
	uv run diff-cover coverage.xml --compare-branch=main || true

lint: ## Run Ruff linter
	@echo "Linting code..."
	uv run ruff check src tests

lint-fix: ## Auto-fix linting issues
	@echo "Fixing lint issues..."
	uv run ruff check --fix src tests

format: ## Format code with Ruff
	@echo "Formatting code..."
	uv run ruff format src tests

check: ## Run lint and format checks
	@echo "Running checks..."
	uv run ruff check src tests
	uv run ruff format --check src tests

clean: ## Remove cache files
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help: ## List all available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
