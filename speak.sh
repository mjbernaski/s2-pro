#!/usr/bin/env bash
# Launcher for speak.py - ensures conda base env is active

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Initialize conda if not already available
if ! command -v conda &>/dev/null; then
    CONDA_BASE="$HOME/miniconda3"
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    elif [ -f "$CONDA_BASE/Scripts/activate" ]; then
        source "$CONDA_BASE/Scripts/activate"
    else
        echo "ERROR: Cannot find conda installation at $CONDA_BASE"
        exit 1
    fi
fi

conda activate base

# Verify key dependencies
python -c "import loguru, torch" 2>/dev/null || {
    echo "Installing missing dependencies..."
    pip install loguru
}

cd "$SCRIPT_DIR"
python speak.py "$@"
