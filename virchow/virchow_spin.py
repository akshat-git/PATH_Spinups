from huggingface_hub import login

login(
    token=""
)

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image

# need to specify MLP layer and activation function for proper init
model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()

transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

image = Image.open("/path/to/your/image.png")
image = transforms(image).unsqueeze(0)  # size: 1 x 3 x 224 x 224

output = model(image)  # size: 1 x 257 x 1280

class_token = output[:, 0]    # size: 1 x 1280
patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

# concatenate class token and average pool of patch tokens
embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image

# need to specify MLP layer and activation function for proper init
model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()

transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

image = Image.open("/path/to/your/image.png")
image = transforms(image).unsqueeze(0)  # size: 1 x 3 x 224 x 224

output = model(image)  # size: 1 x 257 x 1280

class_token = output[:, 0]    # size: 1 x 1280
patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

# concatenate class token and average pool of patch tokens
embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

