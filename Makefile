# Long-Running Agent Skill - Development Makefile
# Uses uv for fast Python package management

.PHONY: help install dev-install test lint format clean run-example run-universal check-uv

# Default target
help:
	@echo "Long-Running Agent Skill - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      - Install uv and create virtual environment"
	@echo "  dev-install  - Install development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  test         - Run tests with pytest"
	@echo "  lint         - Run linting with ruff"
	@echo "  format       - Format code with black and ruff"
	@echo "  check        - Run all checks (lint + test)"
	@echo ""
	@echo "Example Commands:"
	@echo "  run-universal - Run the universal example (works with any AI agent)"
	@echo "  run-example   - Run the complete example (legacy DeepAgents)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean        - Clean up generated files"
	@echo "  check-uv     - Check if uv is installed"

# Check if uv is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "❌ uv is not installed. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "✅ uv installed successfully"; \
	}
	@echo "✅ uv is available: $$(uv --version)"

# Install uv and create virtual environment
install: check-uv
	@echo "🚀 Setting up Long-Running Agent Skill development environment..."
	uv venv --python 3.11
	@echo "✅ Virtual environment created"
	@echo ""
	@echo "To activate the virtual environment:"
	@echo "  source .venv/bin/activate  # On Unix/macOS"
	@echo "  .venv\\Scripts\\activate     # On Windows"

# Install development dependencies
dev-install: check-uv
	@echo "📦 Installing development dependencies..."
	uv pip install -e ".[dev]"
	@echo "✅ Development dependencies installed"

# Run tests
test: check-uv
	@echo "🧪 Running tests..."
	uv run pytest -v --cov=long-running-agent --cov-report=term-missing
	@echo "✅ Tests completed"

# Run linting
lint: check-uv
	@echo "🔍 Running linting checks..."
	uv run ruff check .
	uv run mypy scripts/ --ignore-missing-imports
	@echo "✅ Linting completed"

# Format code
format: check-uv
	@echo "🎨 Formatting code..."
	uv run black .
	uv run ruff check --fix .
	@echo "✅ Code formatted"

# Run all checks
check: lint test
	@echo "✅ All checks passed!"

# Run universal example (agent-agnostic)
run-universal: check-uv
	@echo "🚀 Running universal example (works with any AI agent)..."
	uv run python scripts/universal_example.py

# Run complete example (legacy DeepAgents)
run-example: check-uv
	@echo "🚀 Running complete example (legacy DeepAgents)..."
	@echo "⚠️  Note: This requires DeepAgents. Installing..."
	uv pip install deepagents
	uv run python scripts/complete_example.py

# Clean up generated files
clean:
	@echo "🧹 Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ Cleanup completed"

# Development workflow
dev: install dev-install
	@echo "🎉 Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate virtual environment: source .venv/bin/activate"
	@echo "  2. Run universal example: make run-universal"
	@echo "  3. Run checks: make check"
	@echo "  4. Format code: make format"

# Quick development setup
quick: check-uv dev-install run-universal
	@echo "🚀 Quick development setup completed!"