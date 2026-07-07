#!/bin/bash
#SBATCH --job-name=tcga_final
#SBATCH --partition=gpu
#SBATCH -G 4
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
#  final_setup.sh -- the FINAL end-to-end run on GPU with two-tier staged data.
#
#  Data architecture:
#    * Persistent tier ($SCRATCH, PFM_TCGA_ROOT): a stratified ~50 GB (~10% of
#      TCGA) subset of FULL SVS slides, kept as a reusable cache in .../slides/.
#    * Ephemeral tier ($L_SCRATCH / $TMPDIR, node-local SSD): each SVS is copied
#      here just before it is thumbnailed, then evicted; the tiny thumbnails the
#      GPU trains on are also staged here so the DataLoader reads node-local.
#    * Async + resumable: downloads (to scratch) overlap thumbnailing across
#      slides; a cached SVS is never re-downloaded and an existing thumbnail is
#      skipped. So this DOWNLOADS THE SUBSET ONLY IF NEEDED -- otherwise it just
#      accesses the cache and runs training.
#
#  Flow:  ensure data container/venv -> (build 50 GB staged dataset IF absent)
#         -> ensure model container -> stage thumbnails to node-local
#         -> extract embeddings for every model (GPU) -> benchmark (train probes).
#
#  Submit:  mkdir -p logs && sbatch jobs/final_setup.sh
#  Knobs:   FINAL_TARGET_GB (default from config: 50)   PFM_TCGA_ROOT (cache loc)
# =============================================================================
set -uo pipefail

# ── locate the repo (dir with 'models.txt'); robust to bash & sbatch ────
resolve_repo() {
  local d
  for d in "${PFM_REPO:-}" \
           "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)" \
           "${SLURM_SUBMIT_DIR:-}" \
           "$(dirname "${SLURM_SUBMIT_DIR:-/nonexistent}")"; do
    [ -n "$d" ] && [ -e "$d/models.txt" ] && { echo "$d"; return 0; }
  done; return 1
}
REPO="$(resolve_repo)" || { echo "FATAL: cannot find repo (no 'models.txt'). Set PFM_REPO=/path/to/stanf_pfm"; exit 1; }
MED_DIR="$REPO"
RUNTIME="${PFM_ROOT:-$REPO/runtime}"
export PFM_PROJECT_DIR="$REPO" PFM_ROOT="$RUNTIME"

# ── persistent data on $SCRATCH: the full-SVS cache + thumbnails + dataset.csv ─
TCGA="${PFM_TCGA_ROOT:-$RUNTIME/tcga}"
EMB="${PFM_OUTPUT_DIR:-$RUNTIME/embeddings}"
export PFM_TCGA_ROOT="$TCGA" PFM_OUTPUT_DIR="$EMB"

# ── node-local staging area (fast SSD; wiped at job end) ─────────────────────
STAGE="${L_SCRATCH:-$RUNTIME/tmp}"
export TMPDIR="$STAGE/tmp"
mkdir -p "$TMPDIR" "$TCGA" "$EMB" "$RUNTIME/tmp" "$MED_DIR/logs"

# ── HF token (gated models) ──────────────────────────────────────────────────
if [ -f "$RUNTIME/.hf_token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$RUNTIME/.hf_token")"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; TOKSTATE="present (runtime/.hf_token)"
elif [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; TOKSTATE="present (env)"
else
  TOKSTATE="MISSING (gated models will be disregarded)"
fi

# ── apptainer scratch off $HOME (extract layers on node-local when available) ─
export APPTAINER_CACHEDIR="$RUNTIME/cache/apptainer"
if [ -n "${L_SCRATCH:-}" ] && [ -d "${L_SCRATCH:-}" ]; then
  export APPTAINER_TMPDIR="$L_SCRATCH/apptainer_tmp"
else
  export APPTAINER_TMPDIR="$RUNTIME/tmp"
fi
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

TOOL="$(command -v apptainer || command -v singularity)"
TCGA_SIF="$RUNTIME/containers/tcga_build.sif"
TCGA_VENV="$RUNTIME/venvs/tcga_build"
PFM_SIF="$RUNTIME/containers/pfm_base.sif"
DATASET="$TCGA/tables/dataset.csv"
CONFIG="${FINAL_CONFIG:-configs/tcga_staged.yaml}"

hr(){ echo "============================================================"; }
fail(){ echo; echo "FINAL SETUP ABORTED at: $*"; echo "=== end: $(date) ==="; exit 1; }

hr
echo " FINAL SETUP — staged full-SVS (~50 GB) -> all models -> train"
echo "   node:        $(hostname)"
echo "   job:         ${SLURM_JOB_ID:-interactive}"
echo "   repo:        $REPO"
echo "   scratch cache (PFM_TCGA_ROOT): $TCGA"
echo "   node-local staging (L_SCRATCH): ${L_SCRATCH:-<none; using $RUNTIME/tmp>}"
echo "   embeddings:  $EMB"
echo "   HF token:    $TOKSTATE"
echo "   start:       $(date)"
hr
[ -n "$TOOL" ] || fail "apptainer/singularity not on PATH"

# ── STEP 1: ensure CPU data-build container + venv ───────────────────────────
echo; echo "### STEP 1/5  ensure data-build container + venv"
if [ -f "$TCGA_SIF" ] && [ -d "$TCGA_VENV" ]; then
  echo "  present: $TCGA_SIF + $TCGA_VENV"
else
  echo "  missing -> bash jobs/setup_tcga.sh"
  ( cd "$MED_DIR" && bash jobs/setup_tcga.sh ) || fail "data-build setup (jobs/setup_tcga.sh)"
fi

# ── STEP 2: build/complete the staged dataset (resumable + hybrid) ───────────
# Always run the build -- it is idempotent and resumable: etl/manifest skip if
# present, existing thumbnails are skipped, a pre-downloaded SVS cache (from
# jobs/download_tcga.sh) is thumbnailed from disk, and any slide NOT pre-downloaded
# is streamed into node-local and evicted. So a PARTIAL dataset tops up to the
# target instead of being silently reused (the old short-circuit bug).
echo; echo "### STEP 2/5  build/complete staged dataset (reuse cached SVS if present, else stream)"
if [ -d "$TCGA/slides" ] && find "$TCGA/slides" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -q .; then
  echo "  pre-downloaded SVS cache present in $TCGA/slides -> thumbnail from disk where available, stream the rest"
else
  echo "  no SVS cache -> stream each slide into node-local (\$TMPDIR), thumbnail, evict (nothing large persists)"
fi
"$TOOL" exec \
  -B "$MED_DIR:/workspace" -B "$RUNTIME:/runtime" -B "$TCGA:/tcga_data" \
  ${L_SCRATCH:+-B "$L_SCRATCH"} \
  --pwd /workspace "$TCGA_SIF" bash -c "
    set -e
    source /runtime/venvs/tcga_build/bin/activate
    export PYTHONPATH=/workspace
    export PIP_CACHE_DIR=/runtime/cache/pip
    export TMPDIR=/runtime/tmp
    export L_SCRATCH='${L_SCRATCH:-}'   # so stage_process stages to the node-local SSD
    CMD=\"python -m build_tcga_dataset --config $CONFIG\"
    [ -n '${FINAL_TARGET_GB:-}' ] && CMD=\"\$CMD download.target_gb=${FINAL_TARGET_GB:-}\"
    echo \"INFO: \$CMD\"
    eval \$CMD
  " || fail "staged dataset build (build_tcga_dataset --config $CONFIG)"
[ -f "$DATASET" ] || fail "no dataset.csv produced at $DATASET"
echo "  dataset ready: $DATASET  ($(($(wc -l < "$DATASET") - 1)) rows)"

# ── STEP 3: ensure GPU/torch model container ─────────────────────────────────
echo; echo "### STEP 3/5  ensure model container (pfm_base.sif)"
if [ -f "$PFM_SIF" ]; then
  echo "  present: $PFM_SIF"
else
  echo "  missing -> pfm_setup.sh build"
  ( cd "$REPO" && bash pfm_setup.sh build ) || fail "model container build"
fi

# ── STEP 4: stage the persisted tiles scratch -> node-local SSD, feed the GPU ─
echo; echo "### STEP 4/5  stage tiles to node-local SSD ($STAGE) for the GPU"
PATCHES_SCRATCH="$TCGA/patches"   # persistent: tiled once by tile_slides, reused every run
# NB: count without `head` -- under `set -o pipefail`, `find | head` makes find die
# with SIGPIPE and the whole test evaluates false even when patches exist.
npatch=$(find "$PATCHES_SCRATCH" -mindepth 2 -name '*.jpg' 2>/dev/null | wc -l)
if [ "${npatch:-0}" -ge 1 ]; then
  # Stage tiles onto the node-local SSD once, then the GPU reads them locally. With 9
  # models each reading every patch, one Lustre->SSD copy beats 9 Lustre read passes.
  PATCHES_LOCAL="$STAGE/tcga_patches"
  mkdir -p "$PATCHES_LOCAL"
  echo "  staging $npatch tiles: $PATCHES_SCRATCH -> $PATCHES_LOCAL (node-local SSD)"
  ( cd "$PATCHES_SCRATCH" && tar -cf - . ) 2>/dev/null | ( cd "$PATCHES_LOCAL" && tar -xf - ) 2>/dev/null || true
  export PFM_PATCH_DIR="$PATCHES_LOCAL"
  # compute knobs (batch/workers/amp/prefetch) come from the config spec -> STEP 5.
  echo "  GPU reads node-local: $PFM_PATCH_DIR"
else
  THUMBS_LOCAL="$STAGE/thumbnails"
  mkdir -p "$THUMBS_LOCAL"
  # tiny files; copy the ones not already staged
  cp -n "$TCGA"/thumbnails/*.jpg "$THUMBS_LOCAL"/ 2>/dev/null || true
  n_local=$(find "$THUMBS_LOCAL" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
  if [ "$n_local" -ge 1 ]; then
    export PFM_PATCH_DIR="$THUMBS_LOCAL"
    echo "  staged $n_local thumbnails -> $THUMBS_LOCAL (PFM_PATCH_DIR points here)"
  else
    echo "  WARN: no thumbnails staged; extraction will fall back to $TCGA/thumbnails"
  fi
fi

# ── STEP 5: extract embeddings for every model, then benchmark (train) ───────
# Spread models across the allocated GPUs (one model/GPU). GPU count is detected from
# the allocation; the compute knobs (batch/workers/amp/prefetch) come from the config
# spec's `compute:` block -- nothing hardcoded here.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}; [ "$NGPU" -lt 1 ] && NGPU=1
export PFM_RUN_GPUS="$NGPU"
CPT="${SLURM_CPUS_PER_TASK:-1}"

# Read the compute spec from the config (via the tcga_build venv's omegaconf).
read -r SPEC_BATCH SPEC_WORKERS SPEC_AMP SPEC_PREFETCH < <(
  "$TOOL" exec -B "$MED_DIR:/workspace" -B "$RUNTIME:/runtime" --pwd /workspace "$TCGA_SIF" \
    bash -c "source /runtime/venvs/tcga_build/bin/activate 2>/dev/null; python tcga/print_compute_spec.py '$CONFIG'" 2>/dev/null)

# "-" is the printer's "unset" sentinel; "auto" means derive at runtime.
_is_int() { case "$1" in ''|*[!0-9]*) return 1;; *) return 0;; esac; }

# num_workers: spec integer, or 'auto'/'-' -> split the CPU budget across per-GPU procs
if ! _is_int "$SPEC_WORKERS"; then
  SPEC_WORKERS=$(( CPT / NGPU )); [ "$SPEC_WORKERS" -lt 1 ] && SPEC_WORKERS=1
fi
# Export only what the spec provides: batch/prefetch must be integers (never 'auto'/'-'),
# else leave unset so the runner's config default applies.
_is_int "$SPEC_BATCH"          && export PFM_BATCH_SIZE="${PFM_BATCH_SIZE:-$SPEC_BATCH}"
                                  export PFM_NUM_WORKERS="${PFM_NUM_WORKERS:-$SPEC_WORKERS}"
[ "$SPEC_AMP" != "-" ]         && export PFM_AMP_DTYPE="${PFM_AMP_DTYPE:-$SPEC_AMP}"
_is_int "$SPEC_PREFETCH"       && export PFM_PREFETCH_FACTOR="${PFM_PREFETCH_FACTOR:-$SPEC_PREFETCH}"

echo; echo "### STEP 5/5  extract across $NGPU GPU(s) — spec: batch=${PFM_BATCH_SIZE:-default} workers=${PFM_NUM_WORKERS:-default} amp=${PFM_AMP_DTYPE:-default} prefetch=${PFM_PREFETCH_FACTOR:-default}"
( cd "$REPO" && bash pfm_setup.sh run )       # per-model non-fatal; GPU-parallel when PFM_RUN_GPUS>1
echo; echo "  --- benchmark: train a linear probe per (model x task) ---"
( cd "$REPO" && bash pfm_setup.sh benchmark --dataset-csv "$DATASET" ) \
  || echo "  WARN: benchmark returned nonzero (often 'too few labelled samples')."

echo; hr
echo " FINAL SETUP COMPLETE"
echo "   dataset:    $DATASET"
echo "   SVS cache:  $TCGA/slides   ($(du -sh "$TCGA/slides" 2>/dev/null | cut -f1 || echo n/a))"
echo "   embeddings: $EMB/<model>/patch_embeddings.pt"
echo "   results:    $EMB/benchmark/results.csv"
echo "   end:        $(date)"
hr
