"""Virchow2 (paige-ai) -- ViT patch encoder with 4 register tokens.
Embedding = [CLS | mean(patch tokens[5:])] -> 2560-d."""
from pfm_common import runner


def load():
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2", pretrained=True,
        mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def embed(model, batch):
    import torch
    output = model(batch)                 # [B, 261, 1280]
    class_token = output[:, 0]            # [B, 1280]
    patch_tokens = output[:, 5:]          # [B, 256, 1280] (tokens 1-4 are registers)
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # [B, 2560]


if __name__ == "__main__":
    runner.run_patch_encoder("virchow2", load, embed, gated=True)
