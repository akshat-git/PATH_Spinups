from huggingface_hub import login
import torch
login(
    token=""
)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

import h5py
from transformers import AutoModel

# load model
titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)

# load CONCH v1.5 demo features
h5_path = 'TCGA_demo_features/TCGA-RM-A68W-01Z-00-DX1.4E62E4F4-415C-46EB-A6C8-45BA14E82708.h5'
with h5py.File(h5_path, 'r') as file:
    features = torch.from_numpy(file['features'][:])
    coords = torch.from_numpy(file['coords'][:])
    patch_size_lv0 = file['coords'].attrs['patch_size_level0']

# extract slide embedding
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    slide_embedding = titan.encode_slide_from_patch_features(features, coords, patch_size_lv0)
