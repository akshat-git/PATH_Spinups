from huggingface_hub import login

login(
    token=""
)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

# load Prov-GigaPath tile encoder
model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
model.eval()

transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

image = Image.open("/path/to/your/image.png")
image = transform(image).unsqueeze(0)

with torch.inference_mode():
    output = model(image)
