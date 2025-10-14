#!/bin/bash

# AI BMT GUI Submitter MacOS ARM64 Python
# Standalone run script for distribution

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "AI BMT GUI Submitter - macOS ARM64"
echo "=================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is designed for macOS only."
    exit 1
fi

# Check if we're on ARM64
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This version is optimized for Apple Silicon (ARM64) Macs."
fi

# Set environment variables for Qt and libraries
export DYLD_FRAMEWORK_PATH="$SCRIPT_DIR/lib:$DYLD_FRAMEWORK_PATH"
export DYLD_LIBRARY_PATH="$SCRIPT_DIR/lib:$DYLD_LIBRARY_PATH"
export QT_PLUGIN_PATH="$SCRIPT_DIR/PlugIns"

# Add Python path
export PYTHONPATH="$SCRIPT_DIR/bin:$PYTHONPATH"

echo "Starting AI BMT GUI..."

# Try conda first, then system python
if command -v conda >/dev/null 2>&1; then
    echo "Using conda environment py38..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate py38 2>/dev/null || echo "Note: py38 environment not found, using base environment"
    python "$SCRIPT_DIR/main.py"
else
    echo "Using system python..."
    python3 "$SCRIPT_DIR/main.py"
fi