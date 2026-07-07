"""Prov-GigaPath (prov-gigapath) -- tile (patch) encoder via timm. Output: 1536-d.

Note: GigaPath also ships a slide encoder (gigapath.slide_encoder), which consumes
the tile embeddings produced here; that is a separate downstream step.
"""
from pfm_common import runner


def load():
    import timm
    from torchvision import transforms

    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return tile_encoder, transform


def embed(model, batch):
    return model(batch)  # [B, 1536]


if __name__ == "__main__":
    runner.run_patch_encoder("gigapath", load, embed, gated=True)
