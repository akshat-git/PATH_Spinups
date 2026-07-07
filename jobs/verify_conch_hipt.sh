#!/bin/bash
#SBATCH --job-name=verify_conch_hipt
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
#  verify_conch_hipt.sh -- fast CPU check that conch and hipt now SET UP and LOAD.
#
#  conch (ViT-B) and hipt (ViT-256) are small and load on CPU, and this needs no
#  GPU and no TCGA dataset: it points the encoder input at an EMPTY dir, so the
#  runner loads the weights, prints "[<model>] model ready.", then stops cleanly
#  at "no images found". That is exactly the signal we want -- it proves the
#  venv install + imports + weight load work, in minutes.
#
#  Submit:  mkdir -p logs && sbatch jobs/verify_conch_hipt.sh
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

# ── isolated verify tree; EMPTY input dirs so runs are load-only ─────────────
VERIFY="$RUNTIME/verify_conch_hipt"
LOGS="$VERIFY/logs"
export PFM_PATCH_DIR="$VERIFY/empty_patches"    # no images here -> load-only
export PFM_TCGA_ROOT="$VERIFY/empty_tcga"       # thumbnails fallback also empty
export PFM_OUTPUT_DIR="$VERIFY/embeddings"
mkdir -p "$PFM_PATCH_DIR" "$PFM_TCGA_ROOT" "$PFM_OUTPUT_DIR" "$LOGS" "$MED_DIR/logs"

# ── HF token (conch is gated) ────────────────────────────────────────────────
if [ -f "$RUNTIME/.hf_token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$RUNTIME/.hf_token")"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  TOKSTATE="present (runtime/.hf_token)"
elif [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; TOKSTATE="present (env)"
else
  TOKSTATE="MISSING (conch is gated -> may 401/403)"
fi

TOOL="$(command -v apptainer || command -v singularity)"
PFM_SIF="$RUNTIME/containers/pfm_base.sif"

hr(){ echo "============================================================"; }
hr; echo " VERIFY conch + hipt (setup + load-only, CPU)"
echo "   repo:      $REPO"
echo "   HF token:  $TOKSTATE"
echo "   verify:    $VERIFY"
echo "   start:     $(date)"; hr
[ -n "$TOOL" ] || { echo "FATAL: apptainer/singularity not on PATH"; exit 1; }

# ── ensure the model container ───────────────────────────────────────────────
if [ ! -f "$PFM_SIF" ]; then
  echo "pfm_base.sif missing -> building"; ( cd "$REPO" && bash pfm_setup.sh build ) || { echo "FATAL: container build failed"; exit 1; }
fi

# ── ensure the HIPT checkpoint (GitHub repo, not on HF) ──────────────────────
export PFM_HIPT_CKPT="$RUNTIME/repos/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth"
if [ ! -d "$RUNTIME/repos/HIPT/HIPT_4K" ]; then
  echo "cloning mahmoodlab/HIPT (for the HIPT_4K code)"
  mkdir -p "$RUNTIME/repos"
  ( cd "$RUNTIME/repos" && git clone --depth 1 https://github.com/mahmoodlab/HIPT.git ) \
    || echo "WARN: HIPT clone failed -- hipt will report FAIL below."
fi
# repo ckpt is a git-LFS pointer (~134 B); fetch the real ~700 MB weights.
if [ ! -f "$PFM_HIPT_CKPT" ] || [ "$(stat -c%s "$PFM_HIPT_CKPT" 2>/dev/null || echo 0)" -lt 1000000 ]; then
  echo "fetching real HIPT ViT-256 checkpoint via git-lfs media URL ..."
  mkdir -p "$(dirname "$PFM_HIPT_CKPT")"
  curl -fsSL --retry 3 -o "$PFM_HIPT_CKPT" \
    "https://media.githubusercontent.com/media/mahmoodlab/HIPT/master/HIPT_4K/Checkpoints/vit256_small_dino.pth" \
    || echo "WARN: HIPT ckpt download failed -- hipt will fail to load."
fi

# ── per model: reinstall venv, then load-only run ────────────────────────────
declare -A RESULT
for m in conch hipt; do
  hr; echo " MODEL: $m"; hr
  mlog="$LOGS/$m.log"; : > "$mlog"

  echo "  [$m] setup (recreate venv + install deps) -> $mlog"
  ( cd "$REPO" && bash pfm_setup.sh setup "$m" ) >>"$mlog" 2>&1; srn=$?

  echo "  [$m] load-only run (loads weights, stops at 'no images') -> $mlog"
  ( cd "$REPO" && bash pfm_setup.sh run "$m" ) >>"$mlog" 2>&1; rc=$?

  if grep -q "\[$m\] model ready\." "$mlog"; then
    RESULT[$m]="LOADS OK  (weights loaded; setup rc=$srn, run rc=$rc)"
  elif grep -q "ACCESS NOT GRANTED" "$mlog"; then
    RESULT[$m]="NO-ACCESS (gated repo not authorized for this token) -- escalate on HF"
  else
    RESULT[$m]="FAIL      (setup rc=$srn, run rc=$rc) -- see $mlog"
  fi
  echo "  [$m] -> ${RESULT[$m]}"
  echo "  ---- last 25 lines of $mlog ----"; tail -n 25 "$mlog" | sed 's/^/     | /'
done

hr; echo " VERIFICATION SUMMARY"; hr
for m in conch hipt; do printf '  %-8s %s\n' "$m" "${RESULT[$m]}"; done
echo "  logs: $LOGS/<model>.log"
echo "  end:  $(date)"; hr