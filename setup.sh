#!/bin/bash
# Environment for the non-ergodic Mess3 experiment (CPU-friendly, no LLMs).
# Usage:  bash setup_nonergodic.sh   then   source .venv/bin/activate
set -e

PYTHON="${PYTHON:-python3}"
VENV=".venv"

echo "Creating virtual environment in ${VENV} ..."
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip

# CPU torch is plenty for this tiny model. (Drop the index-url line for a GPU build.)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib numba jupyter nbformat nbconvert ipykernel

echo ""
echo "Done. Now:"
echo "  source ${VENV}/bin/activate"
echo "  python experiments/nonergodic/train.py      # writes experiments/nonergodic/run/"
echo "  jupyter notebook plots/analyze.ipynb         # run all cells"