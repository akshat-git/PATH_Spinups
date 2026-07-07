"""Phikon-v2 (owkin) -- ViT-L patch encoder via transformers. Public (no token needed)."""
from pfm_common import runner


def load():
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")

    def transform(img):
        # processor returns a batch dict; strip the batch dim -> [C, H, W]
        return processor(img, return_tensors="pt")["pixel_values"][0]

    return model, transform


def embed(model, batch):
    outputs = model(pixel_values=batch)
    return outputs.last_hidden_state[:, 0, :]  # CLS token -> [B, 1024]


if __name__ == "__main__":
    runner.run_patch_encoder("phikon", load, embed, gated=False)
