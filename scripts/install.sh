#!/bin/bash
# Long-Running Agent Skill - Quick Installation Script
# Usage: curl -sSL https://raw.githubusercontent.com/agent-skills/long-running-agent/main/scripts/install.sh | bash

set -e

echo "🚀 Installing Long-Running Agent Skill..."
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ uv is available: $(uv --version)"

# Create project directory
PROJECT_DIR="${1:-long-running-agent-skill}"
echo "📁 Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Clone the repository
echo "📥 Downloading skill files..."
if command -v git &> /dev/null; then
    git clone https://github.com/agent-skills/long-running-agent.git .
else
    # Fallback: download as ZIP
    curl -L https://github.com/agent-skills/long-running-agent/archive/main.zip -o skill.zip
    unzip skill.zip
    mv long-running-agent-main/* .
    rm -rf long-running-agent-main skill.zip
fi

# Set up Python environment (optional)
echo "🐍 Setting up Python environment..."
uv venv --python 3.11
source .venv/bin/activate 2>/dev/null || true

# Install development dependencies (optional)
echo "📚 Installing development dependencies..."
uv pip install -e ".[dev]" 2>/dev/null || echo "⚠️ Development dependencies skipped (optional)"

# Run validation
echo "🔍 Validating installation..."
if [ -f "scripts/universal_example.py" ]; then
    echo "✅ Universal example found"
else
    echo "❌ Installation validation failed"
    exit 1
fi

echo ""
echo "🎉 Long-Running Agent Skill installed successfully!"
echo ""
echo "📖 Quick Start:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run example: python scripts/universal_example.py"
echo "  3. Read documentation: cat SKILL.md"
echo ""
echo "🔗 Integration:"
echo "  - Copy functions from scripts/universal_example.py into your AI agent"
echo "  - Follow patterns in references/ for your specific agent framework"
echo "  - Works with: Cursor, OpenCode, Claude, and custom agents"
echo ""
echo "📚 Documentation: https://github.com/agent-skills/long-running-agent"