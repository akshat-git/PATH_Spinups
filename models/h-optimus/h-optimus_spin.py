"""H-optimus-0 (bioptimus) -- ViT-g/14 patch encoder via timm. Output: 1536-d."""
from pfm_common import runner


def load():
    import timm
    from torchvision import transforms

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True,
        init_values=1e-5, dynamic_img_size=False,
    )
    # H-optimus uses dataset-specific normalization statistics.
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517),
        ),
    ])
    return model, transform


def embed(model, batch):
    return model(batch)  # [B, 1536]


if __name__ == "__main__":
    runner.run_patch_encoder("h-optimus", load, embed, gated=True)
