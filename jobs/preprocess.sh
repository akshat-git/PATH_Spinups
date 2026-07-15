#!/bin/bash
#SBATCH --job-name=tcga_prep
#SBATCH --partition=normal   # CPU-ONLY: tiling + packing + byte-level JPEG->raw decode. No GPU
                             # here -- this is exactly the work we pull OFF the GPU node so 8
                             # H100s aren't idle decoding. Run this first; the GPU job then just
                             # extracts (STEP 4/5) off the pre-decoded raw bins.
#SBATCH --cpus-per-task=32   # decode is byte-level parallel over a GIL-free thread pool
                             # (cv2/libjpeg-turbo). More cores = more patches decoded at once.
#SBATCH --mem=64G            # tiling holds one SVS + its tiles; decode holds one slide's
                             # compressed blobs + a bounded decoded window. Size-independent.
#SBATCH --time=1-00:00:00    # FULL scope: one-time tile of ~1400 slides (~5-10h, resumable) +
                             # decode of ~22.6M patches. MINI scope finishes in minutes. Under
                             # the 2-day 'normal' ceiling; raise toward 2-00:00:00 if WAN is slow.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
#  preprocess.sh -- CPU-only data preprocessing, split OUT of the GPU run.
#
#  Does the three CPU-bound, GPU-independent stages ONCE, all resumable per slide:
#    tile_slides   : stream/reuse each SVS -> Otsu tissue patches -> patches_tar/<sid>.tar
#    pack_patches  : migrate any legacy loose patches into the per-slide tars
#    decode_patches: expand each tar -> patches_raw/<sid>.bin  (raw uint8; BYTE-LEVEL parallel,
#                    GIL-free thread pool over libjpeg-turbo) so the GPU run does ZERO decode
#  ...plus download / gene_matrix / assemble so tables/dataset.csv (the labels) exists.
#
#  Everything is idempotent + resumable (per-slide .done + .part atomic renames): a slide
#  already tiled/packed/decoded is SKIPPED; a slide killed mid-write left only a .part and is
#  redone. So on GCP spot (CPUs preempted at will) a re-launch continues where it stopped.
#
#  Scope:  PREP_SCOPE=mini  -> stratified ~FINAL_TARGET_GB (default 50 GB, ~142 slides) subset
#          PREP_SCOPE=full  -> ALL ~1400 slides / ~500 GB (config target_gb: null)  [default]
#
#  Submit standalone (recommended -- offloads all CPU work before the GPU job):
#      mkdir -p logs && PREP_SCOPE=full sbatch jobs/preprocess.sh
#  Or it is CALLED automatically by final_setup{,_mini}.sh STEP 2 (verify-or-redo) as a
#  failsafe, so a GPU job on a fresh checkout still works with no separate prep run.
#
#  Knobs:  PREP_SCOPE (mini|full)   FINAL_TARGET_GB (mini's GB cap; default 50)
#          FINAL_CONFIG (default configs/tcga_tiled.yaml)   PFM_DECODE_WORKERS (decode threads)
#          PREP_SKIP_CONTAINER=1 (skip the container-ensure; set when called from a GPU job that
#                                 already ensured it)
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

TCGA="${PFM_TCGA_ROOT:-$RUNTIME/tcga}"
export PFM_TCGA_ROOT="$TCGA"
STAGE="${L_SCRATCH:-$RUNTIME/tmp}"
export TMPDIR="$STAGE/tmp"
mkdir -p "$TMPDIR" "$TCGA" "$RUNTIME/tmp" "$MED_DIR/logs"

# ── HF token (gated GDC/model access is not needed for tiling; harmless to export) ──
if [ -f "$RUNTIME/.hf_token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$RUNTIME/.hf_token")"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

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
DATASET="$TCGA/tables/dataset.csv"
CONFIG="${FINAL_CONFIG:-configs/tcga_tiled.yaml}"

# ── scope: mini caps the tiling to a stratified subset; full = all slides (config null) ──
PREP_SCOPE="${PREP_SCOPE:-full}"
if [ "$PREP_SCOPE" = "mini" ]; then
  export FINAL_TARGET_GB="${FINAL_TARGET_GB:-50}"
fi
# decode thread count: explicit env, else the allocation's cores, else 32.
export PFM_DECODE_WORKERS="${PFM_DECODE_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"

hr(){ echo "============================================================"; }
fail(){ echo; echo "PREPROCESS ABORTED at: $*"; echo "=== end: $(date) ==="; exit 1; }

hr
echo " PREPROCESS ($PREP_SCOPE) — tile -> pack -> DECODE(raw) -> labels   ($CONFIG)"
echo "   node:        $(hostname)"
echo "   job:         ${SLURM_JOB_ID:-inline}"
echo "   repo:        $REPO"
echo "   tcga root:   $TCGA"
echo "   scope:       $PREP_SCOPE${FINAL_TARGET_GB:+  (cap ${FINAL_TARGET_GB} GB)}"
echo "   decode thr:  $PFM_DECODE_WORKERS"
echo "   start:       $(date)"
hr
[ -n "$TOOL" ] || fail "apptainer/singularity not on PATH"

# ── ensure CPU data-build container + venv (unless the caller already did) ────
if [ "${PREP_SKIP_CONTAINER:-0}" = "1" ]; then
  echo "  (container-ensure skipped -- caller guarantees $TCGA_SIF + $TCGA_VENV)"
elif [ -f "$TCGA_SIF" ] && [ -d "$TCGA_VENV" ]; then
  echo "  data-build container present: $TCGA_SIF + $TCGA_VENV"
else
  echo "  data-build container missing -> bash jobs/setup_tcga.sh"
  ( cd "$MED_DIR" && bash jobs/setup_tcga.sh ) || fail "data-build setup (jobs/setup_tcga.sh)"
fi

# ── run the resumable build: tile_slides + pack_patches + decode_patches + labels ────
# The config's `steps:` already include decode_patches (after pack_patches). Every step is
# resumable, so a re-launch tops up whatever is missing and skips finished slides.
echo; echo "### building (tile -> pack -> decode -> labels), resumable per slide"
"$TOOL" exec \
  -B "$MED_DIR:/workspace" -B "$RUNTIME:/runtime" -B "$TCGA:/tcga_data" \
  ${L_SCRATCH:+-B "$L_SCRATCH"} \
  --pwd /workspace "$TCGA_SIF" bash -c "
    set -e
    source /runtime/venvs/tcga_build/bin/activate
    export PYTHONPATH=/workspace
    export PIP_CACHE_DIR=/runtime/cache/pip
    export TMPDIR=/runtime/tmp
    export L_SCRATCH='${L_SCRATCH:-}'
    export PFM_DECODE_WORKERS='${PFM_DECODE_WORKERS}'
    # Ensure the GIL-free SIMD decoder (cv2/libjpeg-turbo) is present -- self-heals a venv built
    # before opencv was added to setup_tcga.sh, so the byte-level decode runs its FAST path (not
    # the Pillow fallback). No-op once installed.
    python -c 'import cv2' 2>/dev/null || pip install --quiet 'opencv-python-headless>=4.8' || true
    CMD=\"python -m build_tcga_dataset --config $CONFIG\"
    [ -n '${FINAL_TARGET_GB:-}' ] && CMD=\"\$CMD download.target_gb=${FINAL_TARGET_GB:-}\"
    echo \"INFO: \$CMD\"
    eval \$CMD
  " || fail "preprocess build (build_tcga_dataset --config $CONFIG)"

# ── verify: what did preprocessing actually produce? ─────────────────────────
ntar=$(find "$TCGA/patches_tar" -maxdepth 1 -name '*.tar' 2>/dev/null | wc -l)
nraw=$(find "$TCGA/patches_raw" -maxdepth 1 -name '*.bin' 2>/dev/null | wc -l)
ndone=$(find "$TCGA/patches_raw" -maxdepth 1 -name '*.done' 2>/dev/null | wc -l)
echo; hr
echo " PREPROCESS ($PREP_SCOPE) COMPLETE"
echo "   dataset.csv:   $([ -f "$DATASET" ] && echo "yes ($(($(wc -l < "$DATASET") - 1)) rows)" || echo "MISSING")"
echo "   patches_tar:   $ntar slide tars"
echo "   patches_raw:   $nraw raw bins ($ndone done sentinels)  <- ZERO-decode GPU input"
echo "   end:           $(date)"
if [ "${nraw:-0}" -lt 1 ]; then
  echo "   ⚠  NO raw bins produced -- decode_patches did not run (check the build log above)."
  echo "      The GPU job will fall back to reading tar-shards and decoding JPEG on-GPU."
fi
hr
[ -f "$DATASET" ] || fail "no dataset.csv produced at $DATASET"
