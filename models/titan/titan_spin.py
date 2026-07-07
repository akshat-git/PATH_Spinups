"""TITAN (MahmoodLab) -- a *slide* encoder: it consumes precomputed patch features
(CONCH v1.5) + coordinates from an .h5 and produces one slide-level embedding.

This differs from the patch encoders, so it uses pfm_common for auth/paths/output
but its own slide-level loop. Provide features via:
    export PFM_SLIDE_FEATURES=/path/to/slide_features.h5
otherwise the first *.h5 found under $PFM_TCGA_ROOT is used.
"""
import os

from pfm_common import config, data, hf_auth


def main():
    import h5py
    import torch
    from transformers import AutoModel

    hf_auth.login_hf(required=True)
    dev = config.device()
    print(f"[titan] loading model (device={dev}) ...", flush=True)
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.eval().to(dev)
    print("[titan] model ready.", flush=True)

    h5_path = data.find_slide_feature_h5()
    if not h5_path:
        print(
            "[titan] Model loaded OK, but no patch-feature .h5 was found.\n"
            "[titan] Set PFM_SLIDE_FEATURES=/path/to/features.h5 (CONCH v1.5 features)\n"
            f"[titan] or place an .h5 under {config.TCGA_ROOT}, then re-run.",
            flush=True,
        )
        return
    print(f"[titan] using features: {h5_path}", flush=True)

    with h5py.File(h5_path, "r") as f:
        features = torch.from_numpy(f["features"][:]).to(dev)
        coords = torch.from_numpy(f["coords"][:]).to(dev)
        patch_size_lv0 = f["coords"].attrs["patch_size_level0"]

    with torch.autocast(dev, torch.float16), torch.inference_mode():
        slide_embedding = titan.encode_slide_from_patch_features(features, coords, patch_size_lv0)

    out_dir = config.output_dir_for("titan")
    out_path = os.path.join(out_dir, "slide_embedding.pt")
    torch.save({"model": "titan", "embedding": slide_embedding.float().cpu(),
                "source": h5_path}, out_path)
    print(f"[titan] saved slide embedding {tuple(slide_embedding.shape)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
