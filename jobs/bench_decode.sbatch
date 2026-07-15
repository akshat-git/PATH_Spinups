#!/bin/bash
#SBATCH --job-name=bench_decode
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Time the NEW optimized byte-level decoder on real tars (CPU only). Throwaway output.
set -uo pipefail
REPO="${PFM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNTIME="${PFM_ROOT:-$REPO/runtime}"
TCGA="${PFM_TCGA_ROOT:-$RUNTIME/tcga}"
SIF="$RUNTIME/containers/tcga_build.sif"
SCRATCHPAD="/tmp/claude-473530/-scratch-users-akshatg-PATH-Spinups/65f04ec2-40d8-4622-b489-ca1045a87ee5/scratchpad"
TOOL="$(command -v apptainer || command -v singularity)"
export APPTAINER_CACHEDIR="$RUNTIME/cache/apptainer"
export APPTAINER_TMPDIR="${L_SCRATCH:-$RUNTIME/tmp}/apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$REPO/logs"

echo "node=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  L_SCRATCH=${L_SCRATCH:-none}"
"$TOOL" exec -B "$REPO:/workspace" -B "$RUNTIME:/runtime" -B "$TCGA:/tcga_data" \
  -B "$SCRATCHPAD:/bench" ${L_SCRATCH:+-B "$L_SCRATCH"} \
  --pwd /workspace "$SIF" bash -c "
    set -e
    source /runtime/venvs/tcga_build/bin/activate
    export PYTHONPATH=/workspace
    export L_SCRATCH='${L_SCRATCH:-}'
    python -c 'import cv2' 2>/dev/null || pip install --quiet 'opencv-python-headless>=4.8' || true
    python /bench/bench_decode.py /tcga_data/patches_tar ${BENCH_SLIDES:-12} ${SLURM_CPUS_PER_TASK:-32}
  "
