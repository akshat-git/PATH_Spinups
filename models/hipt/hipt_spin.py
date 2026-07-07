"""HIPT (mahmoodlab) -- hierarchical ViT. This adapter runs the ViT-256 patch
encoder (256x256 tiles -> 384-d).

HIPT weights are NOT on the HF hub: the checkpoint `vit256_small_dino.pth` ships
in the GitHub repo (HIPT_4K/Checkpoints/). Point at it via:
    export PFM_HIPT_CKPT=/path/to/vit256_small_dino.pth
"""
import os
import sys

from pfm_common import runner

CKPT = os.environ.get(
    "PFM_HIPT_CKPT",
    os.path.join(os.environ.get("PFM_ROOT", ""), "repos", "HIPT",
                 "HIPT_4K", "Checkpoints", "vit256_small_dino.pth"),
)

# HIPT is not a pip package: the ViT-256 code lives in the cloned GitHub repo
# under HIPT_4K/. We need TWO dirs on sys.path: the repo root (so `import
# HIPT_4K.hipt_model_utils` resolves) AND HIPT_4K/ itself (because
# hipt_model_utils.py does `import vision_transformer` / `vision_transformer4k`,
# which are sibling modules inside HIPT_4K/). The repo root is three dirs above
# the checkpoint (.../repos/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth);
# fall back to $PFM_ROOT/repos/HIPT.
_HIPT_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(CKPT))))
if not os.path.isdir(os.path.join(_HIPT_REPO, "HIPT_4K")):
    _HIPT_REPO = os.path.join(os.environ.get("PFM_ROOT", ""), "repos", "HIPT")
for _p in (_HIPT_REPO, os.path.join(_HIPT_REPO, "HIPT_4K")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def load():
    from HIPT_4K.hipt_model_utils import get_vit256, eval_transforms
    from torchvision import transforms as T

    if not os.path.isfile(CKPT):
        raise FileNotFoundError(
            f"HIPT checkpoint not found: {CKPT}\n"
            "Download vit256_small_dino.pth from the HIPT repo and set PFM_HIPT_CKPT."
        )
    model = get_vit256(pretrained_weights=CKPT)
    # HIPT's eval_transforms() is only ToTensor + Normalize (NO resize), so raw
    # variable-size thumbnails can't be stacked into a batch. ViT-256 expects
    # 256x256 tiles -> resize to 256x256 before ToTensor/Normalize.
    transform = T.Compose([T.Resize((256, 256)), eval_transforms()])  # PIL -> Tensor[C,256,256]
    return model, transform


def embed(model, batch):
    return model(batch)  # [B, 384]


if __name__ == "__main__":
    runner.run_patch_encoder("hipt", load, embed, gated=False)
