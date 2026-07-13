#!/usr/bin/env bash
#
# =============================================================================
#  pfm_setup.sh  --  One-file containerized setup + runner for the pathology
#                    foundation models (PFM) in this directory, on the
#                    Stanford Sherlock HPC cluster.
# =============================================================================
#
#  WHAT THIS DOES
#  --------------
#  * Builds ONE Apptainer container (CUDA + PyTorch base) into $SCRATCH.
#  * Creates a SEPARATE Python virtual environment for EACH model folder,
#    stored in $SCRATCH, using the container's Python (apptainer exec).
#    -> every model's dependencies are isolated and fully containerized.
#  * Sends EVERY git clone, pip cache, HuggingFace cache and temp dir into
#    $SCRATCH (never $HOME, per Sherlock policy).
#  * Installs all dependencies for every model, then runs each model's
#    *_spin.py file.
#
#  SHARED INFRASTRUCTURE (pfm_common/)
#  -----------------------------------
#  Each model folder holds only a thin <model>_spin.py adapter that knows how to
#  build that one model + its transform and turn images into an embedding.
#  Everything reusable -- HF auth, TCGA data location, the batched extraction
#  loop, saving, and the downstream linear-probe trainer -- lives once in
#  pfm_common/ (project root) and is imported by every adapter. This script puts
#  the project dir on the container PYTHONPATH so `from pfm_common import ...`
#  works inside every venv.
#
#  DATA (env-driven; nothing hard-coded in the model scripts)
#  ----------------------------------------------------------
#    PFM_TCGA_ROOT   TCGA dataset root      (default: $GROUP_SCRATCH/datasets/tcga)
#    PFM_PATCH_DIR   dir of pre-tiled patch images (preferred encoder input)
#    PFM_OUTPUT_DIR  where embeddings land  (default: $PFM_ROOT/embeddings)
#    PFM_MAX_IMAGES / PFM_BATCH_SIZE        smoke-test / throughput knobs
#  If no images are found, a run still succeeds through imports + weight download
#  and stops cleanly at the missing-data step -- the intended behavior until real
#  TCGA tiles are staged.
#
#  TRAIN A DOWNSTREAM HEAD (frozen-encoder linear probe)
#  ----------------------------------------------------
#    ./pfm_setup.sh run uni2        # 1) extract embeddings -> $PFM_OUTPUT_DIR/uni2
#    ./pfm_setup.sh shell uni2      # 2) then, inside the venv:
#      python -m pfm_common.train_probe \
#        --embeddings $PFM_ROOT/embeddings/uni2/patch_embeddings.pt \
#        --labels /path/to/labels.csv     # CSV columns: path,label[,split]
#
#  BENCHMARK ALL MODELS ACROSS ALL TASKS (head-to-head comparison)
#  ---------------------------------------------------------------
#    ./pfm_setup.sh run                 # extract embeddings for every model
#    ./pfm_setup.sh benchmark \         # probe each (model x TCGA task) + compare
#        --dataset-csv /path/to/tcga/tables/dataset.csv
#    # -> $PFM_OUTPUT_DIR/benchmark/{results.csv,results.json,heatmap_auroc.png}
#    # Tasks: luad_vs_lusc lgg_vs_gbm kras tp53 egfr idh (pfm_common/tasks.py).
#    # Labels come from the TCGA ETL dataset.csv -- build with tcga/.
#
#  USAGE (run as a Slurm job -- do NOT run heavy steps on the login node)
#  ---------------------------------------------------------------------
#    # Everything (build container + all venvs + run every model) on a GPU node:
#    sbatch pfm_setup.sh
#
#    # Or step by step from an interactive session (sh_dev / salloc):
#    ./pfm_setup.sh build              # pull/build the base container only
#    ./pfm_setup.sh setup              # create venvs + install deps (all models)
#    ./pfm_setup.sh setup conch uni2   # ...only specific models
#    ./pfm_setup.sh run                # run every model's *_spin.py
#    ./pfm_setup.sh run titan          # ...only one model
#    ./pfm_setup.sh benchmark          # probe every (model x task) + compare/plot
#    ./pfm_setup.sh all                # build + setup + run (== sbatch default)
#    ./pfm_setup.sh shell virchow      # interactive shell inside a model's venv
#    ./pfm_setup.sh list               # list known models
#    ./pfm_setup.sh clean              # remove all venvs/repos from $SCRATCH
#
#  SECURITY NOTE
#  -------------
#  The *_spin.py files contain a hard-coded HuggingFace token. That token is
#  effectively public now and should be rotated at
#  https://huggingface.co/settings/tokens . Going forward, export it instead:
#      export HF_TOKEN=hf_xxx          # picked up automatically below
#  Do not commit real tokens to source control.
#
# =============================================================================

# ---- Slurm directives (used when submitted with `sbatch pfm_setup.sh`) -------
#SBATCH --job-name=pfm_setup
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Optionally constrain GPU memory, e.g.:  #SBATCH -C GPU_MEM:24GB

set -uo pipefail

# -----------------------------------------------------------------------------
# Configuration (override any of these via the environment)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the project dir robustly. Under `sbatch`, Slurm runs a spooled copy of
# this script, so SCRIPT_DIR is NOT the project; prefer the submission dir.
# Priority: explicit override -> Slurm submit dir (if it holds the project)
#           -> the script's own dir.
if [[ -n "${PFM_PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$PFM_PROJECT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -e "$SLURM_SUBMIT_DIR/models.txt" ]]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$SCRIPT_DIR"
fi

# All persistent, large, and scratch I/O lives under here. Kept inside the repo
# (which itself sits on $SCRATCH/Lustre, not $HOME) so code and its generated
# artifacts -- container, venvs, embeddings, data -- live together.
PFM_ROOT="${PFM_ROOT:-$PROJECT_DIR/runtime}"

CONTAINER_DIR="$PFM_ROOT/containers"
VENV_DIR="$PFM_ROOT/venvs"
REPO_DIR="$PFM_ROOT/repos"          # <- ALL git clones land here ($SCRATCH)
CACHE_DIR="$PFM_ROOT/cache"
# Process TMPDIR: prefer the node-local SSD ($L_SCRATCH, fast NVMe wiped at job end)
# so temp files never land on Lustre ($SCRATCH); fall back to $PFM_ROOT/tmp on the
# login node where there is no $L_SCRATCH.
if [[ -n "${L_SCRATCH:-}" && -d "${L_SCRATCH:-}" ]]; then
  TMP_DIR="$L_SCRATCH/tmp"
else
  TMP_DIR="$PFM_ROOT/tmp"
fi

# Base image. --nv passes the host NVIDIA driver through, so CUDA minor-version
# compatibility lets this run on Sherlock GPUs. Change the tag if you hit a
# "CUDA driver version is insufficient" error (e.g. drop to cuda11.8).
PFM_BASE_IMAGE="${PFM_BASE_IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime}"
SIF="$CONTAINER_DIR/pfm_base.sif"

# HuggingFace token: the gitignored $PFM_ROOT/.hf_token file is the SINGLE source of
# truth. NEVER hard-code a token in this script -- it is tracked in git, so a literal
# would leak the secret on commit/share. The file wins; a pre-set $HF_TOKEN env var is
# used only as a fallback when the file is absent. If neither exists, gated models are
# simply skipped.
if [ -f "$PFM_ROOT/.hf_token" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$PFM_ROOT/.hf_token")"
else
  HF_TOKEN="${HF_TOKEN:-}"
fi

# Redirect every cache / temp dir off of $HOME and onto $SCRATCH.
export HF_HOME="$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export XDG_CACHE_DIR="$CACHE_DIR/xdg"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export TORCH_HOME="$CACHE_DIR/torch"
export TMPDIR="$TMP_DIR"
# Force the node-local TMPDIR *inside* the container too -- apptainer can otherwise
# reset it. $L_SCRATCH is bind-mounted by arun, so this path is valid in-container.
export APPTAINERENV_TMPDIR="$TMP_DIR"
export SINGULARITYENV_TMPDIR="$TMP_DIR"
export PYTHONUNBUFFERED=1

# Make the shared pfm_common package importable inside every model venv. The
# project dir is bound in via arun (-B "$PROJECT_DIR"); putting it on the
# container PYTHONPATH lets each thin <model>_spin.py do `from pfm_common import
# ...` with no per-file sys.path hacks. apptainer forwards APPTAINERENV_* vars.
export APPTAINERENV_PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$APPTAINERENV_PYTHONPATH"

# Apptainer itself caches/builds in $HOME by default -- move it off $HOME.
# CACHEDIR holds the downloaded OCI layers; keep it on $SCRATCH so the ~3 GB
# pull is reused across builds.
export APPTAINER_CACHEDIR="$CACHE_DIR/apptainer"
# TMPDIR is where the image is extracted + squashed -- millions of tiny files,
# which is painfully slow on Lustre ($SCRATCH). Use the node-local SSD
# ($L_SCRATCH) when running inside a job; it is fast for small files and wiped
# at job end. Falls back to $SCRATCH when $L_SCRATCH is unavailable (login node).
if [[ -n "${L_SCRATCH:-}" && -d "${L_SCRATCH:-}" ]]; then
  export APPTAINER_TMPDIR="$L_SCRATCH/apptainer_tmp"
else
  export APPTAINER_TMPDIR="$TMP_DIR"
fi

# Set by ensure_git(): the host path bound into the container so its git binary
# (and shared libs) are reachable for `pip install git+...`. Empty until then.
GIT_BIND=""

# CA bundle path *inside the container* that the host git binary uses for TLS.
# Sherlock's git is compiled for the CentOS path /etc/pki/tls/certs/ca-bundle.crt,
# which the Ubuntu-based PyTorch image lacks -- without pointing it at the image's
# own bundle, `git clone https://...` fails with "error setting certificate file".
PFM_CONTAINER_CA="${PFM_CONTAINER_CA:-/etc/ssl/certs/ca-certificates.crt}"

# Ordered list of model folders (mirrors "models.txt").
MODELS=(conch uni2 virchow virchow2 titan gigapath mSTAR exaone-path phikon h-optimus hipt)

# Packages every model needs. matplotlib is for the downstream benchmark's
# model x task heatmap (pfm_common.plot_results); it degrades to CSV-only if absent.
COMMON_PKGS="huggingface_hub transformers timm pillow requests numpy matplotlib"

# -----------------------------------------------------------------------------
# Per-model dependency registry
#   model_pip_pkgs  <model>  -> extra pip packages (space separated)
#   model_pip_git   <model>  -> pip-installable git URLs (pip install git+...)
#   model_clone     <model>  -> "URL|editable"  clone into $SCRATCH (editable=0/1)
#   model_spin      <model>  -> path to the *_spin.py file (relative to folder)
# -----------------------------------------------------------------------------
model_pip_pkgs() {
  case "$1" in
    titan)       echo "einops einops-exts h5py" ;;
    exaone-path) echo "opencv-python-headless openslide-bin openslide-python pandas h5py omegaconf" ;;
    # HIPT's hipt_model_utils imports cv2/h5py/scipy/skimage/webdataset/einops/tqdm
    # at module top, so the ViT-256 loader needs them all present.
    hipt)        echo "opencv-python-headless h5py scipy scikit-image webdataset einops tqdm" ;;
    *)           echo "" ;;
  esac
}

model_pip_git() {
  # Install these straight from GitHub HTTPS zip archives (NOT git+), so no git
  # binary is needed inside the container. The host git is RHEL7 (old glibc) and
  # cannot run inside the newer container; binding it over /usr/bin previously
  # broke the container python (see ensure_git).
  case "$1" in
    conch) echo "https://github.com/mahmoodlab/CONCH/archive/refs/heads/main.zip" ;;
    mSTAR) echo "https://github.com/mSTAR-project/mSTAR/archive/refs/heads/main.zip" ;;
    # h-optimus loads via timm hf_hub (no pip package needed); hipt uses the cloned
    # HIPT repo on sys.path (hipt_spin) -- neither needs a git-based install.
    *)     echo "" ;;
  esac
}

model_clone() {
  case "$1" in
    gigapath)    echo "https://github.com/prov-gigapath/prov-gigapath|1" ;;
    exaone-path) echo "https://github.com/LG-AI-EXAONE/EXAONE-Path-2.5.git|0" ;;
    *)           echo "" ;;
  esac
}

model_spin() {
  # spin filename convention: <folder>_spin.py
  echo "models/$1/${1}_spin.py"
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { printf '\n\033[1;34m[pfm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[pfm:warn]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[pfm:err]\033[0m %s\n' "$*" >&2; }

APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"

# -----------------------------------------------------------------------------
# ensure_git : make `git` reachable ON THE HOST for host-side clones only.
#   git is used only by the host-side `git clone` for models that clone a repo
#   (gigapath, exaone-path). The in-container pip installs use HTTPS zip archives
#   (see model_pip_git), so NOTHING inside the container needs git.
#
#   IMPORTANT: we deliberately do NOT bind the host git into the container.
#   Sherlock's host git is built for RHEL7 (old glibc); the base image is newer,
#   so the host binary can't run inside it (ld.so "sym != NULL" abort). Worse,
#   binding the host bin dir (often /usr/bin) over the container's shadowed the
#   container's own /usr/bin and python -- `python -m venv` then picked up the
#   host python2.7 and failed with "libpython2.7.so.1.0: cannot open...". So
#   GIT_BIND stays empty and no APPTAINERENV_* git vars are exported.
# -----------------------------------------------------------------------------
ensure_git() {
  GIT_BIND=""   # never bind host git into the container (see note above)
  if ! command -v git >/dev/null 2>&1; then
    # `module`/`ml` may be undefined in a bare sbatch shell -- init Lmod first.
    if ! command -v module >/dev/null 2>&1; then
      for _init in /usr/share/lmod/lmod/init/bash /etc/profile.d/z00_lmod.sh; do
        [[ -r "$_init" ]] && { source "$_init"; break; }
      done
    fi
    module load system git 2>/dev/null || true
  fi
  local git_bin; git_bin="$(command -v git || true)"
  if [[ -z "$git_bin" ]]; then
    warn "no host git found -- host clones (gigapath, exaone-path) may fail."
    warn "add git to PATH or 'ml load system git' if those models are needed."
    return 0
  fi
  log "host git: $git_bin  (host clones only; not bound into the container)"
}

ensure_dirs() {
  mkdir -p "$CONTAINER_DIR" "$VENV_DIR" "$REPO_DIR" "$CACHE_DIR" "$TMP_DIR" \
           "$HF_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
           "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
}

check_apptainer() {
  if [[ -z "$APPTAINER_BIN" ]]; then
    err "apptainer/singularity not found on PATH."
    exit 1
  fi
}

on_login_node_warn() {
  # Heavy work on a login node violates Sherlock policy. Warn if not in a job.
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    warn "Not inside a Slurm job. Heavy build/install/run on the login node is"
    warn "discouraged. Use:  sbatch pfm_setup.sh   or   sh_dev -g 1   first."
  fi
}

# apptainer exec wrapper: binds $SCRATCH + project dir, enables GPU, threads env.
arun() {
  local workdir="$1"; shift
  # Also bind the node-local SSD ($L_SCRATCH) when present, so a staged
  # PFM_PATCH_DIR under $L_SCRATCH is visible inside the container.
  "$APPTAINER_BIN" exec --nv \
    -B "$SCRATCH" -B "$PROJECT_DIR" \
    ${L_SCRATCH:+-B "$L_SCRATCH"} \
    ${GIT_BIND:+-B "$GIT_BIND"} \
    --pwd "$workdir" \
    "$SIF" "$@"
}

venv_path()   { echo "$VENV_DIR/$1"; }
venv_python() { echo "$(venv_path "$1")/bin/python"; }
venv_pip()    { echo "$(venv_path "$1")/bin/pip"; }

# -----------------------------------------------------------------------------
# build : pull/build the single base container into $SCRATCH
# -----------------------------------------------------------------------------
cmd_build() {
  check_apptainer
  ensure_dirs
  on_login_node_warn
  if [[ -f "$SIF" ]]; then
    log "Base container already exists: $SIF  (delete it to rebuild)"
    return 0
  fi
  log "Building base container from $PFM_BASE_IMAGE"
  log "  -> $SIF"
  if ! "$APPTAINER_BIN" build "$SIF" "$PFM_BASE_IMAGE"; then
    warn "Plain build failed; retrying with --fakeroot ..."
    "$APPTAINER_BIN" build --fakeroot "$SIF" "$PFM_BASE_IMAGE"
  fi
  log "Container ready."
}

# -----------------------------------------------------------------------------
# setup_one : create venv + clone repos + install deps for one model
# -----------------------------------------------------------------------------
setup_one() {
  local m="$1"
  local folder="$PROJECT_DIR/models/$m"
  if [[ ! -d "$folder" ]]; then
    err "[$m] folder not found: $folder -- skipping"
    return 1
  fi

  log "=== setup: $m ==="
  local vpy vpip venv
  venv="$(venv_path "$m")"
  vpy="$(venv_python "$m")"
  vpip="$(venv_pip "$m")"

  # 0) idempotent skip: a venv whose pyvenv.cfg exists is treated as already set up --
  # just move on (this is what lets `final_setup.sh` re-run cheaply). The venv python is
  # a container-only symlink, so pyvenv.cfg -- not `-x python` -- is the host-visible
  # marker. Delete the venv dir (or run `pfm_setup.sh clean`) to force a fresh reinstall.
  if [[ -f "$venv/pyvenv.cfg" ]]; then
    log "[$m] venv already present -> $venv (skipping setup)"
    return 0
  fi

  # 1) create an isolated venv (reuses the container's torch via system packages)
  if [[ ! -x "$vpy" ]]; then
    log "[$m] creating venv -> $venv"
    arun "$PROJECT_DIR" python -m venv --system-site-packages "$venv" \
      || { err "[$m] venv creation failed"; return 1; }
  fi

  # 2) upgrade pip tooling
  arun "$PROJECT_DIR" "$vpip" install --upgrade pip setuptools wheel \
    || warn "[$m] pip self-upgrade failed (continuing)"

  # 3) clone any git repos into $SCRATCH (never into $HOME / project dir)
  local clone_spec; clone_spec="$(model_clone "$m")"
  local clone_path=""
  if [[ -n "$clone_spec" ]]; then
    local url="${clone_spec%%|*}"
    local editable="${clone_spec##*|}"
    local name; name="$(basename "${url%.git}")"
    clone_path="$REPO_DIR/$name"
    if [[ ! -d "$clone_path/.git" ]]; then
      log "[$m] cloning $url -> $clone_path"
      git clone --depth 1 "$url" "$clone_path" \
        || { err "[$m] git clone failed"; return 1; }
    else
      log "[$m] repo already cloned: $clone_path"
    fi
  fi

  # 4) install common + model-specific pip packages
  local extra; extra="$(model_pip_pkgs "$m")"
  log "[$m] installing pip packages"
  # shellcheck disable=SC2086
  arun "$PROJECT_DIR" "$vpip" install $COMMON_PKGS $extra \
    || { err "[$m] base pip install failed"; return 1; }

  # 5) install git-based pip packages
  local gits; gits="$(model_pip_git "$m")"
  if [[ -n "$gits" ]]; then
    log "[$m] installing from git: $gits"
    # shellcheck disable=SC2086
    arun "$PROJECT_DIR" "$vpip" install $gits \
      || { err "[$m] git pip install failed"; return 1; }
  fi

  # 6) editable install of a cloned repo, if requested
  if [[ -n "$clone_spec" ]]; then
    local editable="${clone_spec##*|}"
    if [[ "$editable" == "1" && -n "$clone_path" ]]; then
      log "[$m] editable install: $clone_path"
      arun "$PROJECT_DIR" "$vpip" install -e "$clone_path" \
        || warn "[$m] editable install failed (continuing)"
    fi
  fi

  log "[$m] setup complete."
  return 0
}

cmd_setup() {
  check_apptainer
  ensure_dirs
  ensure_git
  on_login_node_warn
  if [[ ! -f "$SIF" ]]; then
    err "Base container missing. Run: ./pfm_setup.sh build"
    exit 1
  fi
  local targets=("$@"); [[ ${#targets[@]} -eq 0 ]] && targets=("${MODELS[@]}")
  local ok=() bad=()
  for m in "${targets[@]}"; do
    if setup_one "$m"; then ok+=("$m"); else bad+=("$m"); fi
  done
  log "setup summary:  OK=[${ok[*]:-}]  FAILED=[${bad[*]:-}]"
  [[ ${#bad[@]} -eq 0 ]]
}

# -----------------------------------------------------------------------------
# run_one : execute a model's *_spin.py inside its venv
# -----------------------------------------------------------------------------
run_one() {
  local m="$1"
  local folder="$PROJECT_DIR/models/$m"
  local spin; spin="$PROJECT_DIR/$(model_spin "$m")"
  local vpy; vpy="$(venv_python "$m")"
  local venv; venv="$(venv_path "$m")"

  log "=== run: $m ==="
  # NB: the venv python is a symlink to the container's /opt/conda python, which
  # does not exist on the host filesystem -- so `-x "$vpy"` is always false here.
  # Check a real host-visible venv marker instead; the python resolves once we
  # exec inside the container.
  if [[ ! -f "$venv/pyvenv.cfg" ]]; then
    err "[$m] venv not found ($venv). Run setup first: ./pfm_setup.sh setup $m"
    return 1
  fi
  if [[ ! -f "$spin" ]]; then
    err "[$m] spin script not found: $spin"
    return 1
  fi
  if [[ ! -s "$spin" ]]; then
    warn "[$m] $spin is empty -- nothing to run yet (no-op success)."
    return 0
  fi

  log "[$m] python $spin"
  warn "[$m] note: reads real TCGA images from the configured paths (PFM_PATCH_DIR ->"
  warn "[$m] thumbnails). If none are staged yet, the run stops cleanly at that step."
  arun "$folder" "$vpy" "$spin"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    log "[$m] run finished cleanly (exit 0)."
  else
    warn "[$m] exited with code $rc (likely the missing-data step -- see log above)."
  fi
  return $rc
}

cmd_run() {
  check_apptainer
  on_login_node_warn
  local targets=("$@"); [[ ${#targets[@]} -eq 0 ]] && targets=("${MODELS[@]}")
  local rc_worst=0 rc

  # PFM_RUN_GPUS>1: run one model per GPU via a WORK QUEUE. A GPU takes the next
  # pending model the INSTANT it frees, instead of waiting for a whole "wave" to finish.
  # The old wave scheme blocked on the slowest model in each wave while the other GPUs
  # sat idle (e.g. virchow at 64 patches/s stalling 3 GPUs); the queue keeps every GPU
  # busy, so makespan drops to ~max(total_work/N, slowest_single_model).
  # Each model is pinned with CUDA_VISIBLE_DEVICES (0..N-1, relative to the SLURM alloc).
  # Completion is detected via a per-GPU rc-file: `kill -0` can't tell a finished child
  # from a zombie, so the wrapper writes run_one's exit code on exit and we poll for it.
  local ngpu="${PFM_RUN_GPUS:-1}"

  # PFM_RUN_MODE=shard: DATA-PARALLEL. Run models one at a time, but split each model's
  # slide-tars across ALL $ngpu GPUs (one process per GPU, disjoint slides), then merge the
  # per-shard slide embeddings. Every GPU works on every model, so makespan ~= total/ngpu
  # instead of being gated by the single slowest model -- this is what lets 8 GPUs actually
  # help (the work-queue below tops out at the slowest model's single-GPU time).
  if [[ "$ngpu" -gt 1 && "${PFM_RUN_MODE:-queue}" == "shard" ]]; then
    log "run: DATA-PARALLEL -- each of ${#targets[@]} models sharded across $ngpu GPUs"
    local ok=() bad=() g r mrc
    for m in "${targets[@]}"; do
      log "[$m] sharding across $ngpu GPUs (PFM_SHARD_COUNT=$ngpu)"
      local rcdir; rcdir="$(mktemp -d "${TMP_DIR:-/tmp}/pfm_shard.XXXXXX")"
      for ((g=0; g<ngpu; g++)); do
        ( CUDA_VISIBLE_DEVICES="$g" PFM_SHARD_INDEX="$g" PFM_SHARD_COUNT="$ngpu" \
            run_one "$m"; echo $? > "$rcdir/g$g.rc" ) &
      done
      wait                                   # all shards of THIS model
      mrc=0
      for ((g=0; g<ngpu; g++)); do
        r="$(cat "$rcdir/g$g.rc" 2>/dev/null || echo 1)"
        [[ "$r" -ne 0 ]] && { [[ "$r" -eq 75 || $mrc -ne 75 ]] && mrc=$r; }
      done
      rm -rf "$rcdir"
      if [[ $mrc -eq 0 ]]; then
        log "[$m] all $ngpu shards done -> merging slide embeddings"
        if arun "$PROJECT_DIR" "$(venv_python "$m")" -m pfm_common.merge_shards "$m" "$ngpu"; then
          ok+=("$m")
        else
          warn "[$m] merge failed"; bad+=("$m"); [[ $rc_worst -eq 0 ]] && rc_worst=1
        fi
      else
        warn "[$m] shard(s) exited rc=$mrc -- not merging (model dropped)"
        bad+=("$m"); [[ $rc_worst -ne 75 ]] && rc_worst=$mrc
      fi
    done
    log "run(data-parallel) summary:  OK=[${ok[*]:-}]  FAILED=[${bad[*]:-}]  (worst rc=$rc_worst)"
    return $rc_worst
  fi

  # PFM_RUN_GPUS>1 (default mode): run one model per GPU via a WORK QUEUE. A GPU takes the
  # next pending model the INSTANT it frees, instead of waiting for a whole "wave" to finish.
  # The old wave scheme blocked on the slowest model in each wave while the other GPUs sat
  # idle (e.g. virchow stalling 3 GPUs); the queue keeps every GPU busy, so makespan drops
  # to ~max(total_work/N, slowest_single_model). (For many GPUs / few models, PFM_RUN_MODE=
  # shard above is better -- it isn't gated by the slowest single model.)
  # Each model is pinned with CUDA_VISIBLE_DEVICES (0..N-1, relative to the SLURM alloc).
  # Completion is detected via a per-GPU rc-file: `kill -0` can't tell a finished child
  # from a zombie, so the wrapper writes run_one's exit code on exit and we poll for it.
  if [[ "$ngpu" -gt 1 ]]; then
    log "run: ${#targets[@]} models across $ngpu GPUs (work queue -- a GPU takes the next model the instant it frees)"
    local qn=${#targets[@]} qi=0 remaining=${#targets[@]}
    local rcdir; rcdir="$(mktemp -d "${TMPDIR:-/tmp}/pfm_run.XXXXXX")"
    local -a gpid=() gmodel=(); local g
    for ((g=0; g<ngpu; g++)); do gpid[$g]=""; gmodel[$g]=""; done
    while (( remaining > 0 )); do
      local did=0
      for ((g=0; g<ngpu; g++)); do
        # reap a GPU whose model finished (its rc-file appeared)
        if [[ -n "${gpid[$g]}" && -f "$rcdir/g$g.rc" ]]; then
          rc="$(cat "$rcdir/g$g.rc")"; rm -f "$rcdir/g$g.rc"
          wait "${gpid[$g]}" 2>/dev/null || true
          [[ "$rc" -ne 0 && $rc_worst -ne 75 ]] && rc_worst=$rc
          log "[${gmodel[$g]}] finished on GPU $g (rc=$rc)"
          gpid[$g]=""; gmodel[$g]=""; remaining=$(( remaining - 1 )); did=1
        fi
        # hand a free GPU the next queued model
        if [[ -z "${gpid[$g]}" && $qi -lt $qn ]]; then
          local m="${targets[$qi]}"; qi=$(( qi + 1 ))
          log "[$m] -> GPU $g"
          ( CUDA_VISIBLE_DEVICES="$g" run_one "$m"; echo $? > "$rcdir/g$g.rc" ) &
          gpid[$g]=$!; gmodel[$g]="$m"; did=1
        fi
      done
      (( did == 0 )) && sleep 5   # all GPUs busy, none finished -> poll (coarse, cheap)
    done
    rm -rf "$rcdir" 2>/dev/null || true
    log "run(work-queue) done across $ngpu GPUs (worst rc=$rc_worst)"
    return $rc_worst
  fi

  # serial (default; used by smoke/proof and single-GPU runs)
  local ok=() bad=()
  for m in "${targets[@]}"; do
    if run_one "$m"; then
      ok+=("$m")
    else
      rc=$?; bad+=("$m")
      # Propagate a meaningful exit code to the caller (the swallowed code is why
      # the proof saw rc=0 for every failure). Prefer 75 (no-access) so callers can
      # disregard a gated model; otherwise keep the last nonzero code (e.g. 137 OOM).
      if [[ $rc_worst -ne 75 ]]; then rc_worst=$rc; fi
    fi
  done
  log "run summary:  OK=[${ok[*]:-}]  NONZERO=[${bad[*]:-}]"
  return $rc_worst
}

# -----------------------------------------------------------------------------
# benchmark : train a linear probe per (model, task) over the extracted
#             embeddings and compare every PFM head-to-head.
#   ./pfm_setup.sh benchmark [--dataset-csv X] [--tasks ...] [extra benchmark args]
# Runs in any model venv (needs only torch+numpy+matplotlib); picks the first
# venv that exists. Labels come from the TCGA ETL dataset.csv -- build it with
# tcga/ (see its README) and pass --dataset-csv, or set
# $PFM_TCGA_ROOT so the default $PFM_TCGA_ROOT/tables/dataset.csv resolves.
# -----------------------------------------------------------------------------
cmd_benchmark() {
  check_apptainer
  on_login_node_warn
  # find a venv to run in (the benchmark is model-agnostic)
  local venv="" vpy=""
  for m in "${MODELS[@]}"; do
    if [[ -f "$(venv_path "$m")/pyvenv.cfg" ]]; then venv="$(venv_path "$m")"; vpy="$(venv_python "$m")"; break; fi
  done
  [[ -n "$venv" ]] || { err "no model venv found -- run setup first: ./pfm_setup.sh setup"; exit 1; }
  log "benchmarking all PFMs across all tasks (venv: $venv)"
  arun "$PROJECT_DIR" "$vpy" -m pfm_common.benchmark "$@" || return $?
  # auto-plot the comparison from the results it just wrote
  local results="${PFM_OUTPUT_DIR:-$PFM_ROOT/embeddings}/benchmark/results.csv"
  if [[ -f "$results" ]]; then
    arun "$PROJECT_DIR" "$vpy" -m pfm_common.plot_results --results "$results"
  fi
}

# -----------------------------------------------------------------------------
# shell : drop into an interactive shell with a model's venv activated
# -----------------------------------------------------------------------------
cmd_shell() {
  check_apptainer
  local m="${1:?usage: pfm_setup.sh shell <model>}"
  local venv; venv="$(venv_path "$m")"
  [[ -d "$venv" ]] || { err "no venv for $m -- run setup first"; exit 1; }
  ensure_git
  log "Entering container shell with $m venv. Type 'exit' to leave."
  "$APPTAINER_BIN" exec --nv -B "$SCRATCH" -B "$PROJECT_DIR" \
    ${GIT_BIND:+-B "$GIT_BIND"} --pwd "$PROJECT_DIR/models/$m" \
    "$SIF" bash --rcfile <(echo "source '$venv/bin/activate'; PS1='($m) \w \$ '")
}

cmd_list() {
  log "Known models (folder -> spin script):"
  for m in "${MODELS[@]}"; do printf '  %-14s %s\n' "$m" "$(model_spin "$m")"; done
  echo
  log "Paths:"
  printf '  container : %s\n  venvs     : %s\n  repos     : %s\n  caches    : %s\n' \
    "$SIF" "$VENV_DIR" "$REPO_DIR" "$CACHE_DIR"
}

cmd_clean() {
  warn "Removing venvs and cloned repos under $PFM_ROOT (container kept)."
  rm -rf "$VENV_DIR" "$REPO_DIR"
  log "Done. (Use 'rm -rf $PFM_ROOT' to remove everything including the container.)"
}

cmd_all() {
  cmd_build
  cmd_setup "$@"
  cmd_run "$@"
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
main() {
  local cmd="${1:-all}"; shift || true
  case "$cmd" in
    build)  cmd_build ;;
    setup)  cmd_setup "$@" ;;
    run)    cmd_run "$@" ;;
    benchmark) cmd_benchmark "$@" ;;
    all)    cmd_all "$@" ;;
    shell)  cmd_shell "$@" ;;
    list)   cmd_list ;;
    clean)  cmd_clean ;;
    -h|--help|help)
      sed -n '2,70p' "${BASH_SOURCE[0]}" ;;
    *)
      err "unknown command: $cmd"; cmd_list; exit 1 ;;
  esac
}

main "$@"
