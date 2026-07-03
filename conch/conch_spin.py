from huggingface_hub import login

login(
    token=""
)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

import torch
from PIL import Image
from conch.open_clip_custom import create_model_from_pretrained

# load CONCH
model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
model.eval()

image = preprocess(Image.open("/path/to/your/image.png")).unsqueeze(0)

with torch.inference_mode():
    image_embedding = model.encode_image(image, proj_contrast=False, normalize=False)
