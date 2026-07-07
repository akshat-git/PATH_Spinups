"""pfm_common -- shared infrastructure for every pathology foundation model.

The per-model ``<model>/<model>_spin.py`` files are thin adapters: each one only
knows how to (a) build its specific model + image transform and (b) turn a batch
of images into an embedding. Everything else -- HuggingFace auth, locating TCGA
data, the batched extraction loop, saving outputs, and downstream probe training
-- lives here so it is written once and reused by all models.

Typical adapter:

    from pfm_common import runner

    def load():
        model = ...            # build the model
        transform = ...        # PIL.Image -> Tensor[C, H, W]
        return model, transform

    def embed(model, batch):   # batch: Tensor[B, C, H, W] (on device)
        return model(batch)    # -> Tensor[B, D]

    runner.run_patch_encoder("mymodel", load, embed)
"""

from . import config, hf_auth, data, runner  # noqa: F401
from . import metrics, tasks  # noqa: F401  (downstream benchmark: metrics + TCGA task registry)

__all__ = ["config", "hf_auth", "data", "runner", "metrics", "tasks"]
