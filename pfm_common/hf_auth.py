"""Single place that handles HuggingFace authentication.

The token is read from the environment (HF_TOKEN / HUGGINGFACE_HUB_TOKEN) -- it is
NEVER hard-coded in a model script. pfm_setup.sh exports HF_TOKEN into the
container; locally just `export HF_TOKEN=hf_...` before running.
"""
import os


def get_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None


def login_hf(required=False):
    """Log in to the HF Hub if a token is present.

    Returns True if a login was performed. With required=True, raises when no
    token is found (use for gated models that cannot download anonymously).
    """
    token = get_token()
    if not token:
        if required:
            raise RuntimeError(
                "No HF token found. Export HF_TOKEN=hf_... (this model is gated)."
            )
        return False
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    except Exception as e:  # pragma: no cover - network/version dependent
        print(f"[hf_auth] login warning: {e}")
        return False
