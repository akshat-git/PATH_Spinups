#!/bin/bash
#SBATCH --job-name=tcga_final
#SBATCH --partition=gpu
#SBATCH -G 8                 # 8 GPUs, data-parallel (PFM_RUN_MODE=shard): each model is split
                             # across all 8, so 8 GPUs actually speed it up (the work-queue is
                             # gated by the slowest single model; sharding is not).
#SBATCH -C GPU_MEM:80GB      # target the fastest available 8-GPU node. NOTE: Sherlock has NO
                             # B200; the newest is H100 (80GB, Hopper) -- on gpu that's the
                             # single 8-GPU node sh04-01n01. The code is GPU-count-agnostic, so
                             # it'll shard across 8 of WHATEVER it lands on (incl. B200 if ever
                             # added; bump this to the B200's GPU_MEM then). Drop -C for any GPU.
#SBATCH --nodes=1            # all 8 GPUs on ONE node (a process can't drive a GPU on another node)
#SBATCH --cpus-per-task=32   # ~4 dataloader workers/GPU across 8 GPUs (32/8); raise if the node has more
#SBATCH --mem=128G           # RAM = bounded prefetch window x 8 model procs (streaming, no preload);
                             # independent of dataset size. ~12G/proc x 8 shards + headroom.
#SBATCH --time=2-00:00:00    # full run = ALL ~2.26M patches x ~9 models; ~10-16h even with
                             # the work-queue scheduler, so use the gpu partition's 2-day max.
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
#  Flow:  ensure data container/venv -> build the dataset, tiling each SVS into
#         tissue PATCHES that PERSIST under $PFM_TCGA_ROOT/patches (tiled once,
#         reused every run) -> ensure model container + per-model venvs
#         -> stage the persisted patches to node-local -> extract embeddings for
#         every model (GPU) -> benchmark (train probes).
#
#  Submit:  mkdir -p logs && sbatch jobs/final_setup.sh
#  Knobs:   FINAL_CONFIG (default configs/tcga_tiled.yaml; set tcga_staged.yaml for
#           the coarse 1-thumbnail/slide path)   FINAL_TARGET_GB (config default: 50)
#           PFM_TCGA_ROOT (cache + persisted patches location)
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
# Default to the patch-TILED config: it cuts each SVS into tissue patches and PERSISTS
# them to $PFM_TCGA_ROOT/patches (tiled once, reused every run) so we never re-tile or
# fall back to the on-the-fly loader. Override with FINAL_CONFIG=configs/tcga_staged.yaml
# for the coarse 1-thumbnail/slide path instead.
CONFIG="${FINAL_CONFIG:-configs/tcga_tiled.yaml}"

hr(){ echo "============================================================"; }
fail(){ echo; echo "FINAL SETUP ABORTED at: $*"; echo "=== end: $(date) ==="; exit 1; }

hr
echo " FINAL SETUP — tile SVS -> persist patches -> all models -> train  ($CONFIG)"
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
# present, already-tiled slides are skipped, an SVS already cached on $SCRATCH is
# reused from disk, and any slide NOT cached is streamed into node-local and evicted.
# So a PARTIAL dataset tops up to the target instead of being silently reused (the
# old short-circuit bug).
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

# ── STEP 3: ensure GPU/torch model container + per-model venvs ───────────────
echo; echo "### STEP 3/5  ensure model container (pfm_base.sif) + per-model venvs"
if [ -f "$PFM_SIF" ]; then
  echo "  container present: $PFM_SIF"
else
  echo "  container missing -> pfm_setup.sh build"
  ( cd "$REPO" && bash pfm_setup.sh build ) || fail "model container build"
fi
# Create any model venvs that don't exist yet. pfm_setup's `setup` is idempotent: a
# model whose venv (pyvenv.cfg) already exists is skipped, so this is a fast no-op when
# the venvs are already built, and a full per-model install on a fresh runtime. Failures
# are non-fatal here -- the run step (STEP 5) is per-model, so a bad venv only drops that
# one model rather than aborting the whole job.
echo "  ensuring per-model venvs (pfm_setup.sh setup -- skips ones already built)"
( cd "$REPO" && bash pfm_setup.sh setup ) \
  || echo "  WARN: one or more model venvs failed to set up (see log above); continuing."

# ── STEP 4: stage per-slide TAR SHARDS scratch -> node-local SSD ─────────────
# STEP 2's pack_patches packs each slide's tiles into one patches_tar/<slide_id>.tar, so
# we stage ~N_slides big files instead of millions of tiny Lustre files (that metadata
# storm timed STEP 4 out). Extraction streams tar members; PFM_PATCH_STRIDE keeps every
# Nth tile (full run = 1 = all; the mini run exports MINI_FRACTION).
echo; echo "### STEP 4/5  stage patch tar-shards to node-local SSD ($STAGE)"
TARS_SCRATCH="$TCGA/patches_tar"
PATCHES_SCRATCH="$TCGA/patches"
INPUT_MODE=""
export PFM_PATCH_STRIDE="${PFM_PATCH_STRIDE:-1}"

# Loud, unmissable notice (stdout + .err) whenever the run degrades to any fallback path.
fallback_banner() {                        # $1 headline, $2 detail, $3 impact
  echo
  echo "!!! ===================================================================== !!!"
  echo "!!! FALLBACK: $1"
  echo "!!! $2"
  echo "!!! Impact: $3"
  echo "!!! ===================================================================== !!!"
  echo
  echo "FALLBACK: $1 -- $2" >&2
}

SSD_OK=0
if [ -n "${L_SCRATCH:-}" ] && [ -d "${L_SCRATCH:-}" ]; then SSD_OK=1; fi
[ "$SSD_OK" -eq 1 ] || echo "  NOTE: no node-local SSD (\$L_SCRATCH) here; staging target $STAGE is on Lustre."

ntars=$(find "$TARS_SCRATCH" -maxdepth 1 -name '*.tar' 2>/dev/null | wc -l)
if [ "${ntars:-0}" -ge 1 ]; then
  # Copy the ~N_slides tar-shards onto the node-local SSD (big sequential reads, no storm),
  # then the GPU streams tar members -> compute-bound. Also copy the .done sentinels (counts).
  TARS_LOCAL="$STAGE/patches_tar"
  mkdir -p "$TARS_LOCAL"
  echo "  staging $ntars slide tar-shards: $TARS_SCRATCH (Lustre) -> $TARS_LOCAL (node-local SSD)"
  cp -f "$TARS_SCRATCH"/*.tar  "$TARS_LOCAL"/ 2>/dev/null || true
  cp -f "$TARS_SCRATCH"/*.done "$TARS_LOCAL"/ 2>/dev/null || true
  nlocal=$(find "$TARS_LOCAL" -maxdepth 1 -name '*.tar' 2>/dev/null | wc -l)
  if [ "$SSD_OK" -eq 1 ] && [ "${nlocal:-0}" -ge "$ntars" ]; then
    export PFM_PATCH_DIR="$TARS_LOCAL"
    INPUT_MODE="tar-shards ($nlocal slides, node-local SSD)"
    echo "  OK: all $nlocal tar-shards on SSD; GPU reads node-local -> compute-bound: $PFM_PATCH_DIR"
  else
    export PFM_PATCH_DIR="$TARS_SCRATCH"
    INPUT_MODE="tar-shards ($ntars slides, SCRATCH/Lustre -- SSD staging $nlocal/$ntars) -- SLOW FALLBACK"
    fallback_banner "reading tar-shards from Lustre scratch, not the node-local SSD" \
      "$nlocal/$ntars shards reached the SSD ($STAGE)." \
      "reading tar-shards off Lustre -- slower, but sequential (not the old metadata storm)."
  fi
elif [ -d "$PATCHES_SCRATCH" ] && find "$PATCHES_SCRATCH" -mindepth 2 -name '*.jpg' 2>/dev/null | grep -q .; then
  # Patches exist but were never packed -> read LOOSE from Lustre (the storm-prone path).
  export PFM_PATCH_DIR="$PATCHES_SCRATCH"
  INPUT_MODE="LOOSE patches (scratch, unpacked) -- SLOW FALLBACK"
  fallback_banner "no packed tar-shards; reading LOOSE patches straight from Lustre" \
    "pack_patches (STEP 2) produced no $TARS_SCRATCH/*.tar." \
    "millions of tiny Lustre reads -- slow and timeout-prone (exactly what packing avoids)."
else
  # No patches at all -> thumbnails. Empty PFM_PATCH_DIR makes data.py resolve thumbnails.
  export PFM_PATCH_DIR=""
  INPUT_MODE="THUMBNAILS (loader reads $TCGA/thumbnails) -- FALLBACK"
  fallback_banner "using THUMBNAILS, not patch tiles" \
    "No tar-shards and no loose patches (tile_slides/pack_patches didn't run, or wrong FINAL_CONFIG)." \
    "the COARSE 1-image-per-slide path, NOT the GPU-bound tiled run."
fi
# compute knobs (batch/workers/amp/prefetch) come from the config spec -> STEP 5.
echo "  input mode: $INPUT_MODE  (stride 1/${PFM_PATCH_STRIDE})"

# ── STEP 5: extract embeddings for every model, then benchmark (train) ───────
# Spread models across the allocated GPUs (one model/GPU). GPU count is detected from
# the allocation; the compute knobs (batch/workers/amp/prefetch) come from the config
# spec's `compute:` block -- nothing hardcoded here.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}; [ "$NGPU" -lt 1 ] && NGPU=1
export PFM_RUN_GPUS="$NGPU"
# DATA-PARALLEL: shard every model across all $NGPU GPUs (makespan ~ total/NGPU, not gated by
# the slowest single model) -- this is what makes 8 GPUs pay off. Override PFM_RUN_MODE=queue
# for the one-model-per-GPU work queue instead.
export PFM_RUN_MODE="${PFM_RUN_MODE:-shard}"
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
echo "   input used: ${INPUT_MODE:-unknown}"
echo "   SVS cache:  $TCGA/slides   ($(du -sh "$TCGA/slides" 2>/dev/null | cut -f1 || echo n/a))"
echo "   embeddings: $EMB/<model>/patch_embeddings.pt"
echo "   results:    $EMB/benchmark/results.csv"
echo "   end:        $(date)"
case "${INPUT_MODE:-}" in
  *THUMBNAILS*)
    echo "   ⚠  RAN ON THUMBNAILS (fallback) -- NOT the tiled patch-level result."
    echo "      Re-run once tile_slides has persisted patches to $PATCHES_SCRATCH."
    echo "      ran on thumbnails, not patch tiles" >&2 ;;
  *"SLOW FALLBACK"*)
    echo "   ⚠  Ran on patch tiles but read from Lustre scratch, not the node-local SSD"
    echo "      -- extraction was likely I/O-bound, not compute-bound. Check SSD capacity"
    echo "      (\$L_SCRATCH) so the full patch set can stage locally next run."
    echo "      ran I/O-bound off Lustre, not compute-bound off SSD" >&2 ;;
esac
hr
