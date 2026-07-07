"""mSTAR (Wangyh/mSTAR) -- ViT-L patch encoder via timm. Gated on HF."""
from pfm_common import runner


def load():
    import timm
    from torchvision import transforms

    model = timm.create_model(
        "hf-hub:Wangyh/mSTAR", pretrained=True,
        init_values=1e-5, dynamic_img_size=True,
    )
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, transform


def embed(model, batch):
    return model(batch)  # [B, 1024]


if __name__ == "__main__":
    runner.run_patch_encoder("mSTAR", load, embed, gated=True)
