#!/usr/bin/env bash
set -e

# Install dependencies
pip install --user torch torchvision huggingface_hub
pip install --user transformers timm
pip install --user einops einops-exts
