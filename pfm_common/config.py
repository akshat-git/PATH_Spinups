"""Central configuration -- every path/knob is read from the environment so the
same code runs unchanged across all model venvs and on any node.

Override any of these via environment variables (the Slurm job / pfm_setup.sh
exports the important ones for you):

    HF_TOKEN            HuggingFace access token (used by hf_auth.login_hf)
    PFM_ROOT            root for all scratch I/O   (default: <repo>/runtime)
    PFM_TCGA_ROOT       TCGA dataset root          (default: $PFM_ROOT/tcga)
    PFM_PATCH_DIR       directory of pre-tiled patch images (best input source)
    PFM_SLIDE_FEATURES  .h5 of precomputed patch features (slide encoders: TITAN)
    PFM_OUTPUT_DIR      where embeddings are written (default: $PFM_ROOT/embeddings)
    PFM_MAX_IMAGES      cap number of images (0 = all; handy for smoke tests)
    PFM_BATCH_SIZE      dataloader batch size      (default: 8)
    PFM_NUM_WORKERS     dataloader workers         (default: 2)
"""
import os


def _env(name, default=""):
    v = os.environ.get(name, "")
    return v if v else default


def _int_env(name, default):
    """Parse an int env var, falling back to `default` if unset or non-numeric
    (e.g. a stray 'auto'/'-' sentinel) so one bad value can't crash every import."""
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


SCRATCH = _env("SCRATCH", "/tmp")
GROUP_SCRATCH = _env("GROUP_SCRATCH", "")

# All runtime artifacts (container, venvs, embeddings, data) live under the repo
# in <repo>/runtime so code and its outputs stay together on $SCRATCH. The repo
# root is two levels up from this file (pfm_common/config.py -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFM_ROOT = _env("PFM_ROOT", os.path.join(_REPO_ROOT, "runtime"))
TCGA_ROOT = _env("PFM_TCGA_ROOT", os.path.join(PFM_ROOT, "tcga"))
PATCH_DIR = _env("PFM_PATCH_DIR", "")
SLIDE_FEATURES = _env("PFM_SLIDE_FEATURES", "")
OUTPUT_DIR = _env("PFM_OUTPUT_DIR", os.path.join(PFM_ROOT, "embeddings"))

MAX_IMAGES = _int_env("PFM_MAX_IMAGES", 0)         # 0 = no cap
BATCH_SIZE = _int_env("PFM_BATCH_SIZE", 8)         # spec-driven in final_setup; this is the ad-hoc fallback
NUM_WORKERS = _int_env("PFM_NUM_WORKERS", 2)       # ""
AMP_DTYPE = _env("PFM_AMP_DTYPE", "auto")          # auto|float16|bfloat16|float32 (auto: bf16 if GPU supports, else fp16)
PREFETCH_FACTOR = _int_env("PFM_PREFETCH_FACTOR", 0)  # 0 = DataLoader default; else batches prefetched/worker


def device():
    """Return 'cuda' when a GPU is visible, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def output_dir_for(model_name):
    d = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return d
