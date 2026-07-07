#!/bin/bash
#SBATCH --job-name=check_alloc
#SBATCH --partition=gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
#  check_alloc.sbatch -- minimal scheduling probe.
#
#  Requests the SMALLEST possible GPU allocation (1 GPU, 1 CPU, 2G, 2 min) and
#  simply reports whether Slurm actually gave us the CPU + GPU we asked for.
#  Purpose: prove we can get scheduled at all, and confirm the GPU flags
#  (-p gpu AND -G 1) are honoured -- before spending a real job's queue time.
#
#  CPU-only variant:  sbatch -p normal -G 0 pfm_common/check_alloc.sbatch
#                     (GPU checks below will just report "no GPU", which is fine)
#
#  Submit:  cd /path/to/stanf_pfm && mkdir -p logs && sbatch pfm_common/check_alloc.sbatch
# =============================================================================
set -uo pipefail

echo "============================================================"
echo " ALLOCATION CHECK"
echo "   job id:      ${SLURM_JOB_ID:-<none>}"
echo "   node:        $(hostname)"
echo "   partition:   ${SLURM_JOB_PARTITION:-<none>}"
echo "   start:       $(date)"
echo "============================================================"

# ── CPU ──────────────────────────────────────────────────────────────────────
REQ_CPUS="${SLURM_CPUS_PER_TASK:-?}"
GOT_CPUS="$(nproc --all 2>/dev/null || echo '?')"
AVAIL_CPUS="$(nproc 2>/dev/null || echo '?')"
echo
echo "CPU:"
echo "   requested (cpus-per-task): $REQ_CPUS"
echo "   visible to this task:      $AVAIL_CPUS  (node has $GOT_CPUS total)"
if [ "$REQ_CPUS" != "?" ] && [ "$AVAIL_CPUS" != "?" ] && [ "$AVAIL_CPUS" -ge "$REQ_CPUS" ]; then
  echo "   -> CPU OK (got at least what we requested)"
else
  echo "   -> CPU MISMATCH (check request vs. allocation)"
fi

# ── GPU ──────────────────────────────────────────────────────────────────────
echo
echo "GPU:"
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "   SLURM_GPUS=${SLURM_GPUS:-<unset>}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
  if [ "${NGPU:-0}" -ge 1 ]; then
    echo "   -> GPU OK ($NGPU visible):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/        /'
  else
    echo "   -> NO GPU visible (nvidia-smi ran but reported 0)"
  fi
else
  echo "   -> nvidia-smi not on PATH (CPU-only node or no GPU allocated)"
fi

echo
echo "============================================================"
echo " done: $(date)"
echo "============================================================"
