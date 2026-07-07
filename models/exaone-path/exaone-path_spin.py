"""EXAONE-Path-2.5 (LG AI) -- patch encoder via transformers (trust_remote_code)."""
from pfm_common import runner

REPO_ID = "LGAI-EXAONE/EXAONE-Path-2.5"


def load():
    from transformers import AutoModel
    from torchvision import transforms

    model = AutoModel.from_pretrained(REPO_ID, component="patch", trust_remote_code=True)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def embed(model, batch):
    return model(batch)  # [B, C]


if __name__ == "__main__":
    runner.run_patch_encoder("exaone-path", load, embed, gated=True)
