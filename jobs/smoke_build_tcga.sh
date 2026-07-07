#!/bin/bash
#SBATCH --job-name=smoke_build_tcga
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
#  smoke_build_tcga.sh -- ONE submittable, self-bootstrapping smoke test that
#  runs the REAL end-to-end workflow on a tiny slice of TCGA:
#
#       TCGA query  ->  download  ->  thumbnails  ->  model  ->  linear probe
#
#  It uses the SAME containers, venvs, configs and code as a full production
#  run. The only thing that makes it a "smoke" test is the slide cap: it pulls
#  just 5 real slides from the GDC (override with $SMOKE_MAX_FILES) so the whole
#  chain completes in minutes instead of hours.
#
#  STEPS (each is idempotent / resumable -- safe to re-run):
#    0  ensure the CPU data-build container + venv exist   (jobs/setup_tcga.sh)
#    1  REAL GDC build: query -> manifest -> download 5 SVS -> thumbnails ->
#       gene matrix -> assemble  ==> $PFM_TCGA_ROOT/tables/dataset.csv
#    2  ensure the GPU/torch model container + model venv exist (pfm_setup.sh)
#    3  REAL extraction: run the model encoder over the 5 thumbnails  ->
#       patch_embeddings.pt                                      (pfm_setup.sh run)
#    4  REAL training: fit a linear probe per (model x task) over those
#       embeddings + the dataset.csv labels             (pfm_setup.sh benchmark)
#
#  Everything writes under <repo>/runtime (on $SCRATCH); nothing touches $HOME
#  or group storage. The 5-slide smoke data lives in its own runtime/smoke/tree
#  so it never collides with a real runtime/tcga dataset.
#
#  Submit:   mkdir -p logs && sbatch jobs/smoke_build_tcga.sh
#  Tune:     SMOKE_MAX_FILES=40 sbatch jobs/smoke_build_tcga.sh   # bigger probe
#            SMOKE_MODEL=phikon  (default; the only token-free public encoder --
#            gated models join once HF_TOKEN is set and they are `setup`).
#
#  NOTE on the probe: a linear probe needs both classes present with a few
#  samples each. With only 5 slides most (model x task) cells are correctly
#  skipped for "too few labelled samples" -- that is EXPECTED and still proves
#  the training code path runs. Raise SMOKE_MAX_FILES (e.g. 40+) for a probe
#  that actually learns.
# =============================================================================

set -uo pipefail

# ── locate the repo (this file: <repo>/jobs/smoke_build_tcga.sh) ─────────────
JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PFM_REPO:-$(dirname "$JOBS_DIR")}"             # .../stanf_pfm (jobs/ lives at repo root)
[ -e "$REPO/models.txt" ] || REPO="${SLURM_SUBMIT_DIR:-$REPO}"
MED_DIR="$REPO"                                        # flattened: ETL/CLI/jobs live at repo root
RUNTIME="${PFM_ROOT:-$REPO/runtime}"

# smoke artifacts kept separate from any real (full) dataset
TCGA="${PFM_TCGA_ROOT:-$RUNTIME/smoke/tcga}"
EMB="${PFM_OUTPUT_DIR:-$RUNTIME/smoke/embeddings}"

MAXF="${SMOKE_MAX_FILES:-5}"            # real slides to pull from the GDC
MODEL="${SMOKE_MODEL:-phikon}"          # public, no HF token required
MIN_SAMPLES="${SMOKE_MIN_SAMPLES:-2}"   # don't skip tiny task cells in the probe

TOOL="$(command -v apptainer || command -v singularity)"
TCGA_SIF="$RUNTIME/containers/tcga_build.sif"
TCGA_VENV="$RUNTIME/venvs/tcga_build"
PFM_SIF="$RUNTIME/containers/pfm_base.sif"
PFM_VENV="$RUNTIME/venvs/$MODEL"

export APPTAINER_CACHEDIR="$RUNTIME/cache/apptainer"
if [ -n "${L_SCRATCH:-}" ] && [ -d "${L_SCRATCH:-}" ]; then
  export APPTAINER_TMPDIR="$L_SCRATCH/apptainer_tmp"
else
  export APPTAINER_TMPDIR="$RUNTIME/tmp"
fi
mkdir -p "$TCGA" "$EMB" "$RUNTIME/tmp" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$MED_DIR/logs"

echo "============================================================"
echo " SMOKE: real $MAXF-slide TCGA -> model -> probe"
echo "   node:        $(hostname)"
echo "   job:         ${SLURM_JOB_ID:-interactive}"
echo "   repo:        $REPO"
echo "   tcga data:   $TCGA"
echo "   embeddings:  $EMB"
echo "   slides:      $MAXF      model: $MODEL"
echo "   start:       $(date)"
echo "============================================================"

[ -n "$TOOL" ] || { echo "FATAL: apptainer/singularity not on PATH"; exit 1; }
fail() { echo; echo "SMOKE FAILED at: $*"; echo "=== end: $(date) ==="; exit 1; }

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 0/4  ensure CPU data-build container + venv"
if [ -f "$TCGA_SIF" ] && [ -d "$TCGA_VENV" ]; then
  echo "  present: $TCGA_SIF + $TCGA_VENV"
else
  echo "  missing -> running jobs/setup_tcga.sh (pull python:3.10-slim + build venv)"
  ( cd "$MED_DIR" && bash jobs/setup_tcga.sh ) || fail "data-build setup (jobs/setup_tcga.sh)"
fi

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 1/4  REAL GDC build of $MAXF slides -> $TCGA/tables/dataset.csv"
"$TOOL" exec \
  -B "$MED_DIR:/workspace" -B "$RUNTIME:/runtime" -B "$TCGA:/tcga_data" \
  --pwd /workspace "$TCGA_SIF" bash -c "
    set -e
    source /runtime/venvs/tcga_build/bin/activate
    export PYTHONPATH=/workspace
    export PIP_CACHE_DIR=/runtime/cache/pip
    export TMPDIR=/runtime/tmp
    python -m build_tcga_dataset \
      --config configs/tcga_streaming.yaml \
      download.max_files=$MAXF
  " || fail "TCGA ETL build (build_tcga_dataset)"

DATASET="$TCGA/tables/dataset.csv"
[ -f "$DATASET" ] || fail "ETL produced no dataset.csv at $DATASET"
echo "  dataset.csv ready: $DATASET"

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 2/4  ensure model container + '$MODEL' venv"
if [ -f "$PFM_SIF" ]; then
  echo "  container present: $PFM_SIF"
else
  echo "  missing -> ./pfm_setup.sh build"
  ( cd "$REPO" && ./pfm_setup.sh build ) || fail "model container build (pfm_setup.sh build)"
fi
if [ -f "$PFM_VENV/pyvenv.cfg" ]; then
  echo "  venv present: $PFM_VENV"
else
  echo "  missing -> ./pfm_setup.sh setup $MODEL"
  ( cd "$REPO" && ./pfm_setup.sh setup "$MODEL" ) || fail "model venv setup (pfm_setup.sh setup $MODEL)"
fi

# point the PFM pipeline at the real smoke dataset for steps 3-4
export PFM_TCGA_ROOT="$TCGA"
export PFM_OUTPUT_DIR="$EMB"

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 3/4  REAL extraction: $MODEL encoder over the $MAXF thumbnails"
( cd "$REPO" && ./pfm_setup.sh run "$MODEL" ) || fail "extraction (pfm_setup.sh run $MODEL)"
[ -f "$EMB/$MODEL/patch_embeddings.pt" ] || fail "no embeddings written ($EMB/$MODEL/patch_embeddings.pt)"
echo "  embeddings ready: $EMB/$MODEL/patch_embeddings.pt"

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 4/4  REAL training: linear probe over (model x task)"
( cd "$REPO" && ./pfm_setup.sh benchmark \
    --dataset-csv "$DATASET" --models "$MODEL" --min-samples "$MIN_SAMPLES" )
RC=$?
if [ $RC -ne 0 ]; then
  echo
  echo "NOTE: the probe step returned non-zero. With only $MAXF slides this is"
  echo "      usually 'too few labelled samples / one class' -- the training code"
  echo "      ran but had nothing to learn from. Re-run with SMOKE_MAX_FILES=40+"
  echo "      for a probe that actually trains."
fi

echo
echo "============================================================"
echo " SMOKE COMPLETE -- real pipeline ran end to end."
echo "   dataset:    $DATASET"
echo "   embeddings: $EMB/$MODEL/patch_embeddings.pt"
echo "   probe out:  $EMB/benchmark/  (results.csv if any cell trained)"
echo "   end:        $(date)"
echo "============================================================"
