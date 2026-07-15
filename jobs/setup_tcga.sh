#!/bin/bash
#SBATCH --job-name=setup_tcga
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# ONE-TIME SETUP for the TCGA data build.
#
# Pulls a lightweight python:3.10-slim container and creates a lean venv with
# ONLY the data-pipeline deps (query GDC, download, render thumbnails, parse
# MAF). openslide-bin ships the OpenSlide C library as a wheel, so this needs
# no apt / no system libraries. The heavy training stack (torch, transformers,
# ...) is deliberately NOT installed -- the data build is CPU-only.
#
# ALL artifacts live under the repo's runtime/ dir (on $SCRATCH/Lustre). Nothing
# is written to $HOME or to any /scratch/groups/... group storage.
#
# Usage:
#   cd /path/to/stanf_pfm
#   mkdir -p logs
#   sbatch jobs/setup_tcga.sh
# =============================================================================

set -e

# ── Resolve the repo (project) dir robustly ──────────────────────────────────
# Under sbatch, Slurm runs a *spooled copy* (BASH_SOURCE unreliable); and when
# this script is invoked as a child of another job (smoke/proof), SLURM_SUBMIT_DIR
# may point at wherever THAT job was submitted from. So pick the first candidate
# that actually contains this pipeline's files. Override with MED_PROJECT_DIR.
_has_pipeline() { [ -f "$1/jobs/verify_tcga_env.py" ] && [ -f "$1/build_tcga_dataset.py" ]; }
PROJECT_DIR=""
for _c in "${MED_PROJECT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
          "${SLURM_SUBMIT_DIR:-}" \
          "$PWD"; do
    if [ -n "$_c" ] && _has_pipeline "$_c"; then PROJECT_DIR="$_c"; break; fi
done
if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: cannot locate the repo (need jobs/verify_tcga_env.py +" >&2
    echo "       build_tcga_dataset.py). Set MED_PROJECT_DIR=/path/to/stanf_pfm." >&2
    exit 1
fi
REPO_ROOT="$(dirname "$PROJECT_DIR")"                                                 # .../stanf_pfm
RUNTIME="${PFM_ROOT:-$REPO_ROOT/runtime}"                                             # .../stanf_pfm/runtime

TCGA_DATA_DIR="${PFM_TCGA_ROOT:-$RUNTIME/tcga}"   # data output (user scratch, never group)
SIF_STORE="$RUNTIME/containers"
SIF_IMAGE="${SIF_IMAGE:-tcga_build.sif}"
SIF="$SIF_STORE/$SIF_IMAGE"
VENV="$RUNTIME/venvs/tcga_build"
PIP_CACHE="$RUNTIME/cache/pip"
TMP="$RUNTIME/tmp"

echo "INFO: Repo root:     $REPO_ROOT"
echo "INFO: Runtime dir:   $RUNTIME"
echo "INFO: Container:     $SIF"
echo "INFO: Build venv:    $VENV"
echo "INFO: TCGA data dir: $TCGA_DATA_DIR"

mkdir -p "$SIF_STORE" "$PIP_CACHE" "$TMP" "$TCGA_DATA_DIR" logs

# Keep apptainer's own cache/scratch off $HOME during the pull.
export APPTAINER_CACHEDIR="$RUNTIME/cache/apptainer"
if [ -n "${L_SCRATCH:-}" ] && [ -d "${L_SCRATCH:-}" ]; then
    export APPTAINER_TMPDIR="$L_SCRATCH/apptainer_tmp"
else
    export APPTAINER_TMPDIR="$TMP"
fi
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ── Container ────────────────────────────────────────────────────────────────
TOOL=$(command -v apptainer || command -v singularity)
if [ ! -f "$SIF" ]; then
    echo "INFO: Pulling python:3.10-slim -> $SIF"
    "$TOOL" pull "$SIF" "docker://python:3.10-slim"
else
    echo "INFO: Container already exists: $SIF"
fi

# Data-pipeline deps only (NOT the training stack). openslide-bin = prebuilt lib.
# tifffile: read each slide's embedded thumbnail via HTTP Range (streaming path,
# src/data/tcga/slide_streamer.py) without downloading the whole SVS.
# opencv-python-headless: wheel-bundled libjpeg-turbo for the GIL-free, byte-level
# parallel patch decode (tcga/decode_patches.py) -- no system libs needed.
TCGA_DEPS="pandas>=2.0 numpy>=1.24 omegaconf>=2.3 requests>=2.31 pillow>=9.5 \
openslide-python>=1.4.3 openslide-bin>=4.0.0.11 tifffile>=2024.1.30 pyarrow tqdm \
opencv-python-headless>=4.8"

# ── Build the venv + verify, inside the container ────────────────────────────
"$TOOL" exec \
    -B "$PROJECT_DIR:/workspace" \
    -B "$RUNTIME:/runtime" \
    -B "$TCGA_DATA_DIR:/tcga_data" \
    --pwd /workspace \
    "$SIF" \
    bash -c "
    set -e
    export PIP_CACHE_DIR=/runtime/cache/pip
    export PYTHONPATH=/workspace

    if [ ! -d /runtime/venvs/tcga_build ]; then
        echo 'INFO: Creating venv at /runtime/venvs/tcga_build'
        python -m venv /runtime/venvs/tcga_build
    else
        echo 'INFO: venv already exists'
    fi
    source /runtime/venvs/tcga_build/bin/activate

    echo 'INFO: Python: '\$(which python)
    pip install --upgrade pip wheel setuptools
    echo 'INFO: Installing data-pipeline deps (no torch/training stack)...'
    pip install $TCGA_DEPS

    echo ''
    echo '=========================================='
    echo 'VERIFICATION'
    echo '=========================================='
    python /workspace/jobs/verify_tcga_env.py

    echo ''
    echo 'Testing data directory is writable...'
    touch /tcga_data/.write_test && rm /tcga_data/.write_test
    echo 'Writable: /tcga_data'

    echo ''
    echo 'INFO: Setup completed successfully!'
    echo 'INFO: Next step: sbatch jobs/final_setup.sh   (or jobs/final_setup_mini.sh)'
"
