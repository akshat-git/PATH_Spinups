#!/bin/bash
#SBATCH --job-name=tcga_build
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# TCGA DATASET BUILD PIPELINE
# =============================================================================
#
# Downloads TCGA slide images and builds the full dataset:
#   ETL → Manifest → Download → Process Slides → Gene Matrix → Assemble
#
# Data output: $PFM_TCGA_ROOT (default <repo>/runtime/tcga) -- on $SCRATCH, not group
#
# RESOURCE ESTIMATES:
#   - ETL + Manifest: ~5 min (GDC API queries)
#   - Download: 4-20+ hours (depends on # slides, ~2000 SVS files ~500GB)
#   - Process slides: 1-3 hours (SVS → 512x512 JPG thumbnails)
#   - Gene matrix + Assemble: ~10 min
#
# Prerequisites:
#   sbatch jobs/setup_tcga.sh   (one-time)
#
# Usage:
#   cd /path/to/compressed-perception
#   sbatch jobs/build_tcga_dataset.sh
#
#   # Or override steps (e.g., skip download if already done):
#   sbatch jobs/build_tcga_dataset.sh --steps etl,manifest,process_slides,gene_matrix,assemble
#
#   # Or limit files for testing:
#   sbatch jobs/build_tcga_dataset.sh --max-files 5
# =============================================================================

set -e

# Resolve the repo (project) dir robustly. Trusting SLURM_SUBMIT_DIR alone breaks
# when this runs as a child of another job or under sbatch spooling. Pick the
# first candidate that actually contains the pipeline files. Override: MED_PROJECT_DIR.
_has_pipeline() { [ -f "$1/jobs/verify_tcga_env.py" ] && [ -f "$1/build_tcga_dataset.py" ]; }
PROJECT_DIR=""
for _c in "${MED_PROJECT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
          "${SLURM_SUBMIT_DIR:-}" \
          "$PWD"; do
    if [ -n "$_c" ] && _has_pipeline "$_c"; then PROJECT_DIR="$_c"; break; fi
done
if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: cannot locate the repo (need jobs/verify_tcga_env.py +" >&2
    echo "       build_tcga_dataset.py). Set MED_PROJECT_DIR=/path/to/stanf_pfm." >&2
    exit 1
fi
REPO_ROOT="$(dirname "$PROJECT_DIR")"                                                 # .../stanf_pfm
RUNTIME="${PFM_ROOT:-$REPO_ROOT/runtime}"                                             # .../stanf_pfm/runtime
# Output goes under the repo's runtime/ dir by default (repo lives on $SCRATCH).
# Override with PFM_TCGA_ROOT, e.g. export PFM_TCGA_ROOT=$(pwd)/runtime/tcga
TCGA_DATA_DIR="${PFM_TCGA_ROOT:-$RUNTIME/tcga}"

# Parse optional args passed via: sbatch jobs/build_tcga_dataset.sh [args]
EXTRA_ARGS=""
STEPS_OVERRIDE=""
MAX_FILES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS_OVERRIDE="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --force)
            EXTRA_ARGS="$EXTRA_ARGS --force"
            shift
            ;;
        --dry-run)
            EXTRA_ARGS="$EXTRA_ARGS --dry-run"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================================"
echo "TCGA DATASET BUILD"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Project dir:   $PROJECT_DIR"
echo "Data dir:      $TCGA_DATA_DIR"
echo "Start time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Container setup (built one-time by jobs/setup_tcga.sh, lives under runtime/)
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="$RUNTIME/containers"
SIF_IMAGE="${SIF_IMAGE:-tcga_build.sif}"
SIF="$SIF_STORE/$SIF_IMAGE"
VENV="$RUNTIME/venvs/tcga_build"

if [ ! -f "$SIF" ] || [ ! -d "$VENV" ]; then
    echo "ERROR: data-build container/venv not found:"
    echo "         container: $SIF"
    echo "         venv:      $VENV"
    echo "Run setup first: sbatch jobs/setup_tcga.sh"
    exit 1
fi

# Ensure data directories exist
mkdir -p "$TCGA_DATA_DIR" "$RUNTIME/tmp" logs

# Run pipeline inside container
"$TOOL" exec \
    -B "$PROJECT_DIR:/workspace" \
    -B "$RUNTIME:/runtime" \
    -B "$TCGA_DATA_DIR:/tcga_data" \
    --pwd /workspace \
    "$SIF" \
    bash -c "
    set -e
    cd /workspace

    source /runtime/venvs/tcga_build/bin/activate
    export PYTHONPATH=/workspace:\$PYTHONPATH
    export PIP_CACHE_DIR=/runtime/cache/pip
    echo 'INFO: Virtual environment activated.'
    echo 'INFO: Python: '\$(which python)

    export TMPDIR=/runtime/tmp
    mkdir -p \$TMPDIR

    # ==========================================================================
    # Resource monitoring (background)
    # ==========================================================================
    START_TIME=\$(date +%s)
    SLURM_TIME_LIMIT_SEC=\$((24 * 60 * 60))

    monitor_resources() {
        while true; do
            sleep 600  # Log every 10 minutes
            CURRENT_TIME=\$(date +%s)
            ELAPSED=\$((CURRENT_TIME - START_TIME))
            REMAINING=\$((SLURM_TIME_LIMIT_SEC - ELAPSED))
            ELAPSED_H=\$((ELAPSED / 3600))
            ELAPSED_M=\$(((ELAPSED % 3600) / 60))
            REMAINING_H=\$((REMAINING / 3600))
            REMAINING_M=\$(((REMAINING % 3600) / 60))

            echo ''
            echo '============================================================'
            echo \"RESOURCE MONITOR - \$(date '+%Y-%m-%d %H:%M:%S')\"
            echo \"Time elapsed:   \${ELAPSED_H}h \${ELAPSED_M}m\"
            echo \"Time remaining: \${REMAINING_H}h \${REMAINING_M}m\"
            echo \"Disk usage (data dir): \$(du -sh /tcga_data 2>/dev/null | cut -f1)\"
            echo \"CPU/Memory: \$(free -h | grep Mem | awk '{print \$3 \"/\" \$2}')\"
            echo '============================================================'
        done
    }

    monitor_resources &
    MONITOR_PID=\$!
    cleanup() {
        kill \$MONITOR_PID 2>/dev/null || true
    }
    trap cleanup EXIT

    # ==========================================================================
    # Build CLI command
    # ==========================================================================
    CMD=\"python -m build_tcga_dataset\"
    CMD=\"\$CMD --config configs/tcga_streaming.yaml\"

    # Steps override
    STEPS_ARG='$STEPS_OVERRIDE'
    if [ -n \"\$STEPS_ARG\" ]; then
        CMD=\"\$CMD --steps \$STEPS_ARG\"
    fi

    # Max files override
    MAX_FILES_ARG='$MAX_FILES'
    if [ -n \"\$MAX_FILES_ARG\" ]; then
        CMD=\"\$CMD download.max_files=\$MAX_FILES_ARG\"
    fi

    # Extra args (--force, --dry-run)
    CMD=\"\$CMD $EXTRA_ARGS\"

    echo ''
    echo \"INFO: Running: \$CMD\"
    echo ''
    eval \$CMD

    # ==========================================================================
    # Summary
    # ==========================================================================
    END_TIME=\$(date +%s)
    TOTAL_ELAPSED=\$((END_TIME - START_TIME))
    TOTAL_H=\$((TOTAL_ELAPSED / 3600))
    TOTAL_M=\$(((TOTAL_ELAPSED % 3600) / 60))

    echo ''
    echo '============================================================'
    echo 'JOB COMPLETED'
    echo '============================================================'
    echo \"End time:       \$(date '+%Y-%m-%d %H:%M:%S')\"
    echo \"Total runtime:  \${TOTAL_H}h \${TOTAL_M}m\"
    echo \"Data directory: /tcga_data\"
    echo \"Disk usage:     \$(du -sh /tcga_data 2>/dev/null | cut -f1)\"
    echo '============================================================'
"
