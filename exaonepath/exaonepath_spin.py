from huggingface_hub import login

login(
    token=""
)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

import torch
from transformers import AutoModel
from PIL import Image

# load EXAONEPath
model = AutoModel.from_pretrained("LGAI-EXAONE/EXAONEPath1.0", trust_remote_code=True)
model.eval()

image = Image.open("/path/to/your/image.png")

with torch.inference_mode():
    output = model(image)
