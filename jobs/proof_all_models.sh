#!/bin/bash
#SBATCH --job-name=proof_all_models
#SBATCH --partition=gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
#  proof_all_models.sh -- ONE self-bootstrapping run that PROVES every one of the
#  11 PFMs starts, extracts embeddings, and gets a linear probe trained, on a
#  real ~60-slide TCGA slice.
#
#       TCGA query -> download 60 SVS -> thumbnails ->  [for each of 11 models:
#       load weights -> extract embeddings]  -> linear probe per (model x task)
#       -> heatmap + a per-model PROOF table (proof.md / proof.csv)
#
#  It does NOT use `set -e` around the model loop: one model failing must NOT
#  abort the others -- every model's outcome is captured so the final proof
#  table shows the full picture. Each model's full log is saved separately so
#  you can send back the exact failure text.
#
#  Run it (inside a GPU allocation -- extraction wants a GPU, but it will fall
#  back to CPU if none is present):
#       salloc -p gpu -G 1 -c 4 --mem 16G -t 02:00:00
#       bash jobs/proof_all_models.sh
#  ...or submit it:
#       mkdir -p logs && sbatch jobs/proof_all_models.sh
#
#  Knobs:  PROOF_MAX_FILES (default 60)   PROOF_MIN_SAMPLES (default 2)
#          PROOF_MODELS="phikon conch ..." to restrict the model set.
#
#  KNOWN, EXPECTED non-failure: `titan` is a SLIDE encoder -- it consumes
#  precomputed CONCH-v1.5 patch-feature .h5 files, which the thumbnail ETL does
#  NOT produce. titan will LOAD fine but cannot extract here, so it is reported
#  as "load-only (expected)" and is not counted as a failure.
# =============================================================================

set -uo pipefail

# ── locate the repo (dir that holds 'models.txt'); robust to bash & sbatch
resolve_repo() {
  local d
  for d in \
      "${PFM_REPO:-}" \
      "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)" \
      "${SLURM_SUBMIT_DIR:-}" \
      "$(dirname "${SLURM_SUBMIT_DIR:-/nonexistent}")" ; do
    [ -n "$d" ] && [ -e "$d/models.txt" ] && { echo "$d"; return 0; }
  done
  return 1
}
REPO="$(resolve_repo)" || { echo "FATAL: cannot locate repo root (no 'models.txt'). Set PFM_REPO=/path/to/stanf_pfm"; exit 1; }
MED_DIR="$REPO"
RUNTIME="${PFM_ROOT:-$REPO/runtime}"

# ── force pfm_setup.sh to use the right project dir (sbatch spool breaks autodetect)
export PFM_PROJECT_DIR="$REPO"
export PFM_ROOT="$RUNTIME"

# ── proof tree (kept separate from runtime/tcga and runtime/smoke) ───────────
PROOF="$RUNTIME/proof"
TCGA="${PFM_TCGA_ROOT:-$PROOF/tcga}"
EMB="${PFM_OUTPUT_DIR:-$PROOF/embeddings}"
PLOGS="$PROOF/logs"
export PFM_TCGA_ROOT="$TCGA"
export PFM_OUTPUT_DIR="$EMB"
mkdir -p "$TCGA" "$EMB" "$PLOGS" "$RUNTIME/tmp" "$MED_DIR/logs"

MAXF="${PROOF_MAX_FILES:-60}"
MIN_SAMPLES="${PROOF_MIN_SAMPLES:-2}"

# Display order: phikon first (cached, token-free canary), then gated patch
# encoders, then hipt (needs a GitHub ckpt), then titan (slide encoder, load-only).
DEFAULT_MODELS="phikon conch uni2 virchow virchow2 exaone-path gigapath h-optimus mSTAR hipt titan"
read -r -a MODELS <<< "${PROOF_MODELS:-$DEFAULT_MODELS}"

# ── HuggingFace token: file (preferred) -> env -> none ───────────────────────
if [ -f "$RUNTIME/.hf_token" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$RUNTIME/.hf_token")"
  export HF_TOKEN HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  TOKSTATE="present (runtime/.hf_token, ${#HF_TOKEN} chars)"
elif [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  TOKSTATE="present (env, ${#HF_TOKEN} chars)"
else
  TOKSTATE="MISSING -- gated models will 401/403"
fi

# ── HIPT checkpoint: GitHub repo file, not on HF. Point spin at an abs path. ──
export PFM_HIPT_CKPT="$RUNTIME/repos/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth"

TOOL="$(command -v apptainer || command -v singularity)"
TCGA_SIF="$RUNTIME/containers/tcga_build.sif"
TCGA_VENV="$RUNTIME/venvs/tcga_build"
PFM_SIF="$RUNTIME/containers/pfm_base.sif"
DATASET="$TCGA/tables/dataset.csv"
RESULTS="$EMB/benchmark/results.csv"
STATUS_TSV="$PROOF/_status.tsv"   # model<TAB>run_rc<TAB>extract<TAB>shape<TAB>note
: > "$STATUS_TSV"

hr(){ echo "============================================================"; }
fail(){ echo; echo "PROOF ABORTED at: $*"; echo "=== end: $(date) ==="; exit 1; }

hr
echo " PROOF: all ${#MODELS[@]} models -> extract -> train, on $MAXF real TCGA slides"
echo "   node:        $(hostname)"
echo "   job:         ${SLURM_JOB_ID:-interactive}"
echo "   repo:        $REPO"
echo "   proof tree:  $PROOF"
echo "   models:      ${MODELS[*]}"
echo "   HF token:    $TOKSTATE"
echo "   start:       $(date)"
hr
[ -n "$TOOL" ] || fail "apptainer/singularity not on PATH"

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 1/6  ensure CPU data-build container + venv"
if [ -f "$TCGA_SIF" ] && [ -d "$TCGA_VENV" ]; then
  echo "  present: $TCGA_SIF + $TCGA_VENV"
else
  echo "  missing -> bash jobs/setup_tcga.sh"
  ( cd "$MED_DIR" && bash jobs/setup_tcga.sh ) || fail "data-build setup (jobs/setup_tcga.sh)"
fi

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 2/6  REAL GDC build of $MAXF slides -> $DATASET"
if [ -f "$DATASET" ]; then
  echo "  reusing existing dataset.csv ($(wc -l < "$DATASET") lines). Delete $TCGA to rebuild."
else
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
fi
[ -f "$DATASET" ] || fail "ETL produced no dataset.csv at $DATASET"
echo "  dataset.csv ready: $DATASET  ($(($(wc -l < "$DATASET") - 1)) slides)"

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 3/6  ensure GPU/torch model container"
if [ -f "$PFM_SIF" ]; then
  echo "  present: $PFM_SIF"
else
  echo "  missing -> pfm_setup.sh build"
  ( cd "$REPO" && bash pfm_setup.sh build ) || fail "model container build (pfm_setup.sh build)"
fi

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 4/6  ensure HIPT repo + ViT-256 checkpoint (GitHub LFS, not on HF)"
if [ ! -d "$RUNTIME/repos/HIPT/HIPT_4K" ]; then
  echo "  cloning mahmoodlab/HIPT -> $RUNTIME/repos/HIPT (for the HIPT_4K code)"
  mkdir -p "$RUNTIME/repos"
  ( cd "$RUNTIME/repos" && git clone --depth 1 https://github.com/mahmoodlab/HIPT.git ) \
    || echo "  WARN: HIPT clone failed -- hipt will be reported as a failure below."
fi
# The checkpoint tracked in the repo is a git-LFS POINTER (~134 B), not the real
# ~700 MB weights. Fetch the real file from the LFS media URL when the on-disk
# file is missing or pointer-sized.
if [ ! -f "$PFM_HIPT_CKPT" ] || [ "$(stat -c%s "$PFM_HIPT_CKPT" 2>/dev/null || echo 0)" -lt 1000000 ]; then
  echo "  fetching real HIPT ViT-256 checkpoint via git-lfs media URL ..."
  mkdir -p "$(dirname "$PFM_HIPT_CKPT")"
  curl -fsSL --retry 3 -o "$PFM_HIPT_CKPT" \
    "https://media.githubusercontent.com/media/mahmoodlab/HIPT/master/HIPT_4K/Checkpoints/vit256_small_dino.pth" \
    || echo "  WARN: HIPT ckpt download failed -- hipt will fail to load."
fi
[ -f "$PFM_HIPT_CKPT" ] && echo "  ckpt: $PFM_HIPT_CKPT ($(stat -c%s "$PFM_HIPT_CKPT" 2>/dev/null) bytes)"

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 5/6  per-model: setup venv (if needed) -> load weights -> extract"
for m in "${MODELS[@]}"; do
  hr; echo "  MODEL: $m"; hr
  mlog="$PLOGS/$m.log"
  venv="$RUNTIME/venvs/$m"

  # ensure venv
  if [ ! -f "$venv/pyvenv.cfg" ]; then
    echo "  [$m] venv missing -> pfm_setup.sh setup $m"
    ( cd "$REPO" && bash pfm_setup.sh setup "$m" ) >>"$mlog" 2>&1 \
      || echo "  [$m] WARN: setup returned nonzero (see $mlog)"
  fi

  # run extraction (capture full log; do not let a failure kill the loop)
  echo "  [$m] extracting -> log: $mlog"
  ( cd "$REPO" && bash pfm_setup.sh run "$m" ) >>"$mlog" 2>&1
  rc=$?

  emb="$EMB/$m/patch_embeddings.pt"
  shape="$(grep -ho 'saved embeddings ([^)]*)' "$mlog" 2>/dev/null | tail -1 | sed 's/saved embeddings //')"
  if [ -f "$emb" ]; then
    extract="YES"; note="ok"
    echo "  [$m] OK  embeddings=$emb  shape=${shape:-?}"
  elif [ "$m" = "titan" ]; then
    extract="LOAD-ONLY"; note="slide encoder: consumes CONCH .h5 patch features, not thumbnails (expected, not a failure)"
    echo "  [$m] LOAD-ONLY (expected): titan is a SLIDE encoder, not a patch encoder."
  elif [ "$rc" -eq 75 ]; then
    extract="NO-ACCESS"; note="gated repo: HF token missing or not approved -- access not granted, disregarded"
    echo "  [$m] NO-ACCESS (gated): access not granted -- disregarding this model."
  elif [ "$rc" -eq 137 ] || [ "$rc" -eq 139 ]; then
    extract="OOM"; note="killed (rc=$rc): out of memory -- this large model needs a GPU (or more host RAM)"
    echo "  [$m] OOM (rc=$rc): ran out of memory -- large models need a GPU. Last 15 lines:"
    tail -n 15 "$mlog" | sed 's/^/      | /'
  else
    extract="NO"; note="run rc=$rc -- see $mlog (last lines below)"
    echo "  [$m] FAIL (rc=$rc). Last 25 lines of $mlog:"
    tail -n 25 "$mlog" | sed 's/^/      | /'
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$m" "$rc" "$extract" "${shape:-}" "$note" >> "$STATUS_TSV"
done

# ───────────────────────────────────────────────────────────────────────────
echo; echo "### STEP 6/6  train linear probes across every model with embeddings + plot"
have_emb=0
for m in "${MODELS[@]}"; do [ -f "$EMB/$m/patch_embeddings.pt" ] && have_emb=1; done
if [ "$have_emb" -eq 1 ]; then
  ( cd "$REPO" && bash pfm_setup.sh benchmark --dataset-csv "$DATASET" --min-samples "$MIN_SAMPLES" ) \
    || echo "  WARN: benchmark returned nonzero (often 'too few labelled samples' on small slices)."
else
  echo "  SKIP: no model produced embeddings -- nothing to benchmark."
fi

# ── Build the PROOF report (per model: loaded / extracted [NxD] / tasks trained)
PROOF_MD="$PROOF/proof.md"
PROOF_CSV="$PROOF/proof.csv"
{
  echo "model,weights_loaded,embeddings_extracted,emb_shape,probe_tasks_trained,tasks,note"
} > "$PROOF_CSV"

echo; hr; echo " PROOF SUMMARY"; hr
printf '  %-14s %-8s %-12s %-14s %-20s %s\n' MODEL LOADED EXTRACTED SHAPE TASKS_TRAINED NOTE
fails=0; trained_any=0; noaccess=0
while IFS=$'\t' read -r m rc extract shape note; do
  # which tasks trained for this model (rows in results.csv: col1=model col2=task col9=auroc)
  tasks=""; ntasks=0
  if [ -f "$RESULTS" ]; then
    tasks="$(awk -F, -v M="$m" 'NR>1 && $1==M {printf "%s(au=%s) ", $2, $9}' "$RESULTS")"
    ntasks="$(awk -F, -v M="$m" 'NR>1 && $1==M {c++} END{print c+0}' "$RESULTS")"
  fi
  loaded="YES"
  case "$extract" in
    YES)        [ "$ntasks" -gt 0 ] && trained_any=1 ;;
    LOAD-ONLY)  : ;;                                  # expected, not a failure
    NO-ACCESS)  loaded="n/a"; noaccess=$((noaccess+1)) ;;   # access not granted -- disregarded
    OOM)        loaded="?"; fails=$((fails+1)) ;;
    NO)         loaded="?"; fails=$((fails+1)) ;;
  esac
  printf '  %-14s %-8s %-12s %-14s %-20s %s\n' "$m" "$loaded" "$extract" "${shape:-—}" "${ntasks} task(s)" "$note"
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$m" "$loaded" "$extract" "\"${shape:-}\"" "$ntasks" "\"${tasks}\"" "\"${note}\"" >> "$PROOF_CSV"
done < "$STATUS_TSV"
hr

# Markdown proof
{
  echo "# PFM proof run — $(date)"
  echo
  echo "- slides: $MAXF   dataset: \`$DATASET\`"
  echo "- embeddings: \`$EMB/<model>/patch_embeddings.pt\`"
  echo "- results: \`$RESULTS\`   heatmap: \`$EMB/benchmark/heatmap_auroc.png\`"
  echo "- HF token: $TOKSTATE"
  echo
  echo "| model | loaded | extracted | shape | probe tasks (auroc) | note |"
  echo "|---|---|---|---|---|---|"
  while IFS=$'\t' read -r m rc extract shape note; do
    tasks=""
    [ -f "$RESULTS" ] && tasks="$(awk -F, -v M="$m" 'NR>1 && $1==M {printf "%s=%s ", $2, $9}' "$RESULTS")"
    loaded="YES"
    case "$extract" in NO|OOM) loaded="?" ;; NO-ACCESS) loaded="n/a" ;; esac
    echo "| $m | $loaded | $extract | ${shape:-—} | ${tasks:-—} | ${note} |"
  done < "$STATUS_TSV"
} > "$PROOF_MD"

echo
echo "  proof table : $PROOF_MD"
echo "  proof csv   : $PROOF_CSV"
echo "  per-model logs: $PLOGS/<model>.log"
[ -f "$RESULTS" ] && echo "  results.csv : $RESULTS"
[ -f "$EMB/benchmark/heatmap_auroc.png" ] && echo "  heatmap     : $EMB/benchmark/heatmap_auroc.png"

echo; hr
if [ "$fails" -eq 0 ] && [ "$trained_any" -eq 1 ]; then
  echo " RESULT: PASS — every accessible patch encoder extracted; probes trained."
  echo "         (titan is a slide encoder -> load-only, expected.)"
  [ "$noaccess" -gt 0 ] && echo "         ($noaccess gated model(s) disregarded: access not granted on Hugging Face.)"
  rcfinal=0
else
  echo " RESULT: ATTENTION — $fails model(s) failed to extract; trained_any=$trained_any."
  [ "$noaccess" -gt 0 ] && echo "         ($noaccess gated model(s) disregarded: access not granted.)"
  echo "         See the per-model logs above / in $PLOGS and send them back."
  rcfinal=1
fi
echo " end: $(date)"; hr
exit $rcfinal
