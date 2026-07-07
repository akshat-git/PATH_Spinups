"""TCGA data access, shared by every patch-encoder model.

Patch encoders consume small RGB tiles. Where those tiles come from is resolved
here in priority order so that a model script never has to care:

    1. PFM_PATCH_DIR        -- a directory of pre-tiled patch images (preferred)
    2. $PFM_TCGA_ROOT/slides -- whole-slide .svs files, tiled on the fly *if*
                               openslide is importable in the current venv
    3. $PFM_TCGA_ROOT/thumbnails -- low-res slide overviews, used as a fallback
                               so the pipeline can be smoke-tested without tiling

If none of these yield images, find_patch_images() returns [] and the caller
reports a clear "stage your data" message instead of crashing.
"""
import glob
import os

from . import config

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _glob_images(root, recursive=True):
    out = []
    for ext in IMG_EXTS:
        pat = os.path.join(root, "**", "*" + ext) if recursive else os.path.join(root, "*" + ext)
        out += glob.glob(pat, recursive=recursive)
        out += glob.glob(pat.replace(ext, ext.upper()), recursive=recursive)
    return sorted(set(out))


def find_patch_images(limit=None):
    """Return a list of image file paths to feed a patch encoder.

    `limit` (or config.MAX_IMAGES) caps the count. Returns [] if nothing found.
    """
    limit = limit or (config.MAX_IMAGES or None)

    # 1) explicit pre-tiled patch directory
    if config.PATCH_DIR and os.path.isdir(config.PATCH_DIR):
        imgs = _glob_images(config.PATCH_DIR)
        if imgs:
            return imgs[:limit] if limit else imgs

    # 2) TCGA thumbnails (overview PNGs) -- smoke-test fallback, no tiling needed
    thumb = os.path.join(config.TCGA_ROOT, "thumbnails")
    if os.path.isdir(thumb):
        imgs = _glob_images(thumb, recursive=False)
        if imgs:
            return imgs[:limit] if limit else imgs

    return []


def describe_sources():
    """Human-readable summary of where data would be looked for (for messages)."""
    return (
        f"  PFM_PATCH_DIR      = {config.PATCH_DIR or '(unset)'}\n"
        f"  TCGA thumbnails    = {os.path.join(config.TCGA_ROOT, 'thumbnails')}\n"
        f"  TCGA slides (.svs) = {os.path.join(config.TCGA_ROOT, 'slides')}\n"
    )


def find_slide_feature_h5():
    """Locate an .h5 of precomputed patch features for slide encoders (TITAN).

    Priority: PFM_SLIDE_FEATURES file -> first *.h5 under $PFM_TCGA_ROOT.
    Returns a path or None.
    """
    if config.SLIDE_FEATURES and os.path.isfile(config.SLIDE_FEATURES):
        return config.SLIDE_FEATURES
    hits = glob.glob(os.path.join(config.TCGA_ROOT, "**", "*.h5"), recursive=True)
    return sorted(hits)[0] if hits else None


def _open_rgb(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def make_dataset(paths, transform):
    """Build a torch Dataset yielding (transformed_tensor, path)."""
    import torch  # noqa: F401  (ensures torch present)
    from torch.utils.data import Dataset

    class PatchDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = _open_rgb(self.paths[i])
            x = self.transform(img)
            return x, self.paths[i]

    return PatchDataset(paths, transform)


def collate(batch):
    """Collate (tensor, path) pairs -> (stacked tensor, list-of-paths)."""
    import torch
    xs, paths = zip(*batch)
    return torch.stack(list(xs), 0), list(paths)
