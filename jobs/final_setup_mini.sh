#!/bin/bash
#SBATCH --job-name=tcga_final_mini
#SBATCH --partition=gpu
#SBATCH -G 4
#SBATCH --nodes=1            # FIX: force all 4 GPUs onto ONE node. Without this, -G 4 let
                             # Slurm split the alloc across 2 nodes (2 GPUs each) and the
                             # single-node script only used the head node's 2 GPUs.
#SBATCH --cpus-per-task=20   # 5 dataloader workers/GPU (auto = cpus/GPUs = 20/4); fits the
                             # 20-core gpu nodes too, so it schedules on more of the pool.
#SBATCH --mem=128G           # 4 model procs now share ONE node (--nodes=1); prior 2-proc run
                             # OOM'd at 32G. 128G fits every gpu node (>=191G) with headroom.
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
#  final_setup_mini.sh -- IDENTICAL to final_setup.sh, but extraction runs on only
#  a 1/MINI_FRACTION SAMPLE (default 10%) of the persisted patch tiles. Same setup,
#  same models, same benchmark -- ~10x less GPU work, so it fits a short walltime.
#
#  Fully self-bootstrapping / failsafe (an external party can run this on a fresh
#  checkout): it builds every missing piece itself --
#    * data-build container + venv        (STEP 1, via jobs/setup_tcga.sh)
#    * the dataset: streams/tiles slides -> PATCHES persisted under
#      $PFM_TCGA_ROOT/patches, + gene labels + dataset.csv   (STEP 2)
#    * model container + one venv per model (STEP 3, via pfm_setup.sh, idempotent)
#  Nothing is assumed to pre-exist; already-built pieces are detected and reused.
#
#  The ONLY difference from final_setup.sh: STEP 4 stages a 1/MINI_FRACTION sample
#  of the tiles to the node-local SSD. Tiling in STEP 2 still persists the FULL set,
#  so final_setup.sh can later use all of them -- nothing is thrown away.
#
#  Flow:  ensure data container/venv -> build dataset (tile+persist ALL patches)
#         -> ensure model container + per-model venvs -> stage a 1/N SAMPLE of the
#         patches to node-local SSD -> extract (GPU) -> benchmark (train probes).
#
#  Submit:  mkdir -p logs && sbatch jobs/final_setup_mini.sh
#  Knobs:   MINI_FRACTION (default 10 -> use 1/10 of the patches)
#           FINAL_CONFIG (default configs/tcga_tiled.yaml; set tcga_staged.yaml for
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

# MINI: extraction uses only 1 in MINI_FRACTION of the persisted patches (default 10 = 10%).
# STEP 2 still tiles+persists the FULL set; only STEP 4's node-local sample is reduced. Clamp
# a bad value (non-integer or 0) back to 10 so the run can never divide-by-zero / stage all.
MINI_FRACTION="${MINI_FRACTION:-10}"
case "$MINI_FRACTION" in ''|*[!0-9]*|0) MINI_FRACTION=10;; esac

hr(){ echo "============================================================"; }
fail(){ echo; echo "FINAL SETUP MINI ABORTED at: $*"; echo "=== end: $(date) ==="; exit 1; }

hr
echo " FINAL SETUP MINI — tile SVS -> persist patches -> extract 1/$MINI_FRACTION -> train  ($CONFIG)"
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

# ── STEP 4: stage a 1/MINI_FRACTION SAMPLE of the tiles -> node-local SSD ─────
echo; echo "### STEP 4/5  stage a 1/$MINI_FRACTION sample of the tiles to node-local SSD ($STAGE)"
PATCHES_SCRATCH="$TCGA/patches"   # persistent: tiled once by tile_slides, reused every run
INPUT_MODE=""                     # set below: "patches" (intended) or a THUMBNAIL fallback
# NB: count without `head` -- under `set -o pipefail`, `find | head` makes find die
# with SIGPIPE and the whole test evaluates false even when patches exist.
npatch=$(find "$PATCHES_SCRATCH" -mindepth 2 -name '*.jpg' 2>/dev/null | wc -l)

# Loud, unmissable notice -- printed to BOTH the .out and the .err log -- whenever the
# run degrades to any fallback path (thumbnails instead of tiles, or Lustre reads
# instead of the node-local SSD). The user explicitly wants to be told about EVERY
# fallback: a silent one wastes an expensive GPU job.
fallback_banner() {                        # $1 headline, $2 detail, $3 impact
  echo
  echo "!!! ===================================================================== !!!"
  echo "!!! FALLBACK: $1"
  echo "!!! $2"
  echo "!!! Impact: $3"
  echo "!!! ===================================================================== !!!"
  echo
  echo "FALLBACK: $1 -- $2" >&2            # also surface in the .err log
}

# The staging technique only helps if there IS a node-local SSD to stage onto. In a GPU
# job $L_SCRATCH is the fast local NVMe; on a login node there is none and $STAGE falls
# back to Lustre (staging is then a no-op copy -- flag it).
SSD_OK=0
if [ -n "${L_SCRATCH:-}" ] && [ -d "${L_SCRATCH:-}" ]; then SSD_OK=1; fi
[ "$SSD_OK" -eq 1 ] || echo "  NOTE: no node-local SSD (\$L_SCRATCH) here; staging target $STAGE is on Lustre."

if [ "${npatch:-0}" -ge 1 ]; then
  # Stage only a 1/MINI_FRACTION SAMPLE of the tiles onto the node-local SSD, then the GPU
  # reads them from fast local NVMe. That is ~1/MINI_FRACTION the GPU work of the full run
  # while staying COMPUTE-bound (SSD reads), so it fits a short walltime.
  PATCHES_LOCAL="$STAGE/tcga_patches_mini"
  mkdir -p "$PATCHES_LOCAL"
  FILELIST="$STAGE/mini_patchlist.txt"
  # Deterministic 1/MINI_FRACTION sample across the SORTED patch list. Sorting groups a
  # slide's patches together, so every-Nth spreads the sample across ALL slides -- benchmark
  # still mean-pools each slide's kept patches into a slide-level vector for the probe.
  ( cd "$PATCHES_SCRATCH" && find . -mindepth 2 -name '*.jpg' 2>/dev/null | sort \
      | awk -v f="$MINI_FRACTION" 'f<=1 || NR % f == 0' ) > "$FILELIST"
  nsel=$(wc -l < "$FILELIST" 2>/dev/null); nsel=${nsel:-0}
  echo "  MINI: staging $nsel of $npatch tiles (1/$MINI_FRACTION): $PATCHES_SCRATCH -> $PATCHES_LOCAL (node-local SSD)"
  ( cd "$PATCHES_SCRATCH" && tar -cf - -T "$FILELIST" ) 2>/dev/null | ( cd "$PATCHES_LOCAL" && tar -xf - ) 2>/dev/null || true
  # Require a COMPLETE stage of the SAMPLE: an incomplete SSD copy would silently drop
  # sampled patches. If it fails, fall back to the FULL set on scratch (the sampled subset
  # lives only on the SSD, so it can't be cleanly re-read from Lustre) -- big + slow.
  nstaged=$(find "$PATCHES_LOCAL" -mindepth 2 -name '*.jpg' 2>/dev/null | wc -l)
  if [ "$SSD_OK" -eq 1 ] && [ "${nstaged:-0}" -ge "$nsel" ] && [ "${nstaged:-0}" -ge 1 ]; then
    export PFM_PATCH_DIR="$PATCHES_LOCAL"
    INPUT_MODE="patches ($nstaged of $npatch = 1/$MINI_FRACTION sample, node-local SSD)"
    echo "  OK: $nstaged sampled tiles on SSD; GPU reads node-local -> compute-bound: $PFM_PATCH_DIR"
  else
    export PFM_PATCH_DIR="$PATCHES_SCRATCH"
    INPUT_MODE="patches (FULL $npatch tiles, SCRATCH/Lustre -- mini SSD staging $nstaged/$nsel) -- SLOW FALLBACK"
    if [ "$SSD_OK" -ne 1 ]; then
      fallback_banner "no node-local SSD; reading the FULL patch set off Lustre (mini sample not staged)" \
        "No \$L_SCRATCH on this node, so the 1/$MINI_FRACTION sample could not be staged locally." \
        "extraction runs on ALL $npatch tiles off Lustre -- slow AND I/O-bound; the mini speedup is lost."
    else
      fallback_banner "mini SSD staging incomplete; reading the FULL patch set off Lustre" \
        "Only $nstaged/$nsel sampled tiles reached the SSD ($STAGE) -- likely full." \
        "extraction runs on ALL $npatch tiles off Lustre -- slow AND I/O-bound; the mini speedup is lost."
    fi
  fi
  # compute knobs (batch/workers/amp/prefetch) come from the config spec -> STEP 5.
else
  THUMBS_LOCAL="$STAGE/thumbnails"
  mkdir -p "$THUMBS_LOCAL"
  # tiny files; copy the ones not already staged
  cp -n "$TCGA"/thumbnails/*.jpg "$THUMBS_LOCAL"/ 2>/dev/null || true
  n_local=$(find "$THUMBS_LOCAL" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
  if [ "$n_local" -ge 1 ]; then
    export PFM_PATCH_DIR="$THUMBS_LOCAL"
    INPUT_MODE="THUMBNAILS ($n_local staged, node-local) -- FALLBACK"
  else
    INPUT_MODE="THUMBNAILS (loader reads $TCGA/thumbnails) -- FALLBACK"
  fi
  fallback_banner "using THUMBNAILS, not persisted patch tiles" \
    "No patch tiles found in $PATCHES_SCRATCH (tile_slides didn't finish, FINAL_CONFIG isn't the tiled config, or patches/ was purged -- see STEP 2)." \
    "the COARSE 1-image-per-slide path, NOT the GPU-bound tiled run the tiled config intends."
fi
echo "  input mode: $INPUT_MODE"

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
echo " FINAL SETUP MINI COMPLETE  (extraction on 1/$MINI_FRACTION of the tiles)"
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
