"""CONCH (MahmoodLab) -- ViT-B/16 patch encoder. Thin adapter over pfm_common."""
import os
import sys

from pfm_common import runner

# This file lives in models/conch/, and the repo root is on PYTHONPATH (so
# `from pfm_common import ...` works). The folder is literally named 'conch', so
# `import conch` would resolve to models/conch/ instead of the pip-installed CONCH
# package, breaking `conch.open_clip_custom`. Drop models/conch/'s parent dirs from
# sys.path (pfm_common is already imported and cached above) and clear the stale
# 'conch' namespace so the installed package resolves.
_HERE = os.path.dirname(os.path.abspath(__file__))        # .../models/conch
_MODELS = os.path.dirname(_HERE)                          # .../models
_REPO = os.path.dirname(_MODELS)                          # repo root
sys.path = [p for p in sys.path
            if os.path.abspath(p or ".") not in (_HERE, _MODELS, _REPO)]
sys.modules.pop("conch", None)


def load():
    from conch.open_clip_custom import create_model_from_pretrained
    from pfm_common.hf_auth import get_token
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=get_token()
    )
    return model, preprocess


def embed(model, batch):
    return model.encode_image(batch, proj_contrast=False, normalize=False)


if __name__ == "__main__":
    runner.run_patch_encoder("conch", load, embed, gated=True)
