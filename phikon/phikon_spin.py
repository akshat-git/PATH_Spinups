import requests
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


# Load an image
image = Image.open(
    requests.get(
        "https://github.com/owkin/HistoSSLscaling/blob/main/assets/example.tif?raw=true",
        stream=True
    ).raw
)

# Load phikon-v2
processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
model = AutoModel.from_pretrained("owkin/phikon-v2")
model.eval()

# Process the image
inputs = processor(image, return_tensors="pt")

# Get the features
with torch.inference_mode():
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape

assert features.shape == (1, 1024)
