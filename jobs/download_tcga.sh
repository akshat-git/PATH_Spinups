#!/bin/bash
#SBATCH --job-name=download_tcga
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
#  download_tcga.sh -- standalone "download all of TCGA" job.
#
#  Pre-downloads the FULL SVS of the stratified staged subset into the PERSISTENT
#  $SCRATCH cache at  PFM_TCGA_ROOT/slides/<file_id>/<filename>  (CPU-only, no GPU).
#  Runs steps: etl -> manifest -> download_svs_cache.
#
#  This is OPTIONAL and decoupled from training. If you run it, a later
#  jobs/final_setup.sh reuses these cached SVS (thumbnailing from disk, no
#  network) via the hybrid in tcga/slide_stager.py; any slide NOT pre-downloaded
#  is streamed on demand. If you DON'T run it, final_setup streams everything.
#  Idempotent + resumable: already-cached SVS are skipped, so re-submit to resume.
#
#  Size is the same subset knobs as the staged config:
#    DOWNLOAD_TARGET_GB=50   (default from configs/tcga_staged.yaml: ~50 GB / ~142 slides)
#    DOWNLOAD_MAX_FILES=N    (cap by slide COUNT instead of bytes)
#  For (nearly) everything, set a large DOWNLOAD_TARGET_GB -- but the full 4-project
#  corpus is multi-TB, so scope to what fits your $SCRATCH quota.
#
#  Submit:  mkdir -p logs && sbatch jobs/download_tcga.sh
#           DOWNLOAD_TARGET_GB=200 sbatch jobs/download_tcga.sh
#  Then:    sbatch jobs/final_setup.sh   (reuses the cache; streams anything missing)
# =============================================================================

set -uo pipefail

# ── Resolve the repo dir (must contain the pipeline files). Override MED_PROJECT_DIR.
_has_pipeline() { [ -f "$1/jobs/verify_tcga_env.py" ] && [ -f "$1/build_tcga_dataset.py" ]; }
PROJECT_DIR=""
for _c in "${MED_PROJECT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
          "${SLURM_SUBMIT_DIR:-}" \
          "$PWD"; do
    if [ -n "$_c" ] && _has_pipeline "$_c"; then PROJECT_DIR="$_c"; break; fi
done
if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: cannot locate the repo. Set MED_PROJECT_DIR=/path/to/stanf_pfm." >&2
    exit 1
fi
REPO_ROOT="$(dirname "$PROJECT_DIR")"
RUNTIME="${PFM_ROOT:-$REPO_ROOT/runtime}"
TCGA_DATA_DIR="${PFM_TCGA_ROOT:-$RUNTIME/tcga}"
CONFIG="${DOWNLOAD_CONFIG:-configs/tcga_staged.yaml}"

TOOL=$(command -v apptainer || command -v singularity)
SIF="$RUNTIME/containers/${SIF_IMAGE:-tcga_build.sif}"
VENV="$RUNTIME/venvs/tcga_build"

echo "============================================================"
echo " DOWNLOAD TCGA -> persistent SVS cache"
echo "   job:         ${SLURM_JOB_ID:-interactive}   node: $(hostname)"
echo "   repo:        $PROJECT_DIR"
echo "   cache dir:   $TCGA_DATA_DIR/slides"
echo "   config:      $CONFIG"
echo "   target_gb:   ${DOWNLOAD_TARGET_GB:-<config default>}   max_files: ${DOWNLOAD_MAX_FILES:-<none>}"
echo "   start:       $(date)"
echo "============================================================"

[ -n "$TOOL" ] || { echo "FATAL: apptainer/singularity not on PATH"; exit 1; }
if [ ! -f "$SIF" ] || [ ! -d "$VENV" ]; then
  echo "ERROR: data-build container/venv missing. Run: sbatch jobs/setup_tcga.sh"
  echo "         container: $SIF"; echo "         venv:      $VENV"
  exit 1
fi

mkdir -p "$TCGA_DATA_DIR" "$RUNTIME/tmp" logs

"$TOOL" exec \
  -B "$PROJECT_DIR:/workspace" -B "$RUNTIME:/runtime" -B "$TCGA_DATA_DIR:/tcga_data" \
  --pwd /workspace "$SIF" bash -c "
    set -e
    source /runtime/venvs/tcga_build/bin/activate
    export PYTHONPATH=/workspace
    export PIP_CACHE_DIR=/runtime/cache/pip
    export TMPDIR=/runtime/tmp
    CMD=\"python -m build_tcga_dataset --config $CONFIG --steps etl,manifest,download_svs_cache\"
    [ -n '${DOWNLOAD_TARGET_GB:-}' ] && CMD=\"\$CMD download.target_gb=${DOWNLOAD_TARGET_GB:-}\"
    [ -n '${DOWNLOAD_MAX_FILES:-}' ] && CMD=\"\$CMD download.max_files=${DOWNLOAD_MAX_FILES:-}\"
    echo \"INFO: \$CMD\"
    eval \$CMD
  " || { echo "DOWNLOAD FAILED (resumable -- resubmit to continue)"; exit 1; }

echo "============================================================"
echo " DOWNLOAD COMPLETE"
echo "   SVS cache:   $TCGA_DATA_DIR/slides   ($(du -sh "$TCGA_DATA_DIR/slides" 2>/dev/null | cut -f1 || echo n/a))"
echo "   next:        sbatch jobs/final_setup.sh   (reuses this cache; streams anything missing)"
echo "   end:         $(date)"
echo "============================================================"
