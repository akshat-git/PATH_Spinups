"""TCGA data access, shared by every patch-encoder model.

Patch encoders consume small RGB tiles. Where those tiles come from is resolved
here in priority order so that a model script never has to care:

    1. PFM_PATCH_DIR        -- a directory of pre-tiled patch images (preferred)
    2. $PFM_TCGA_ROOT/thumbnails -- low-res slide overviews, used as a fallback
                               so the pipeline can still run without tiling

If neither yields images, resolve_patch_root() returns (None, _) and the caller
reports a clear "stage your data" message instead of crashing.

MEMORY MODEL -- constant in the dataset size N
-----------------------------------------------
Data is NEVER materialised in RAM as a whole. The tiles stay as compressed JPEGs
on the (node-local) disk; a streaming ``IterableDataset`` walks them one path at a
time and hands each decoded tile to a bounded DataLoader prefetch queue. Host RAM
therefore holds only the in-flight window
    num_workers x prefetch_factor x batch_size x bytes_per_tile
which is independent of N -- the SAME footprint whether there are 226k or 2.26M
tiles. Nothing here scales with the dataset size: not the pixel buffers (bounded
prefetch) and not the path index (streamed, never listed in full).

Correctness with multiple workers: every tile is produced by EXACTLY ONE worker.
Each worker walks the identical, deterministically-ordered path stream and keeps
only the paths at its stride (``global_index % num_workers == worker_id``), so the
union over workers is the full set with no duplicates and no omissions.
"""
import os

from . import config

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _iter_image_paths(root, recursive=True):
    """Yield image paths under ``root`` one at a time, in a DETERMINISTIC order,
    using O(1) memory in the dataset size (only one directory's names are held at
    a time). The fixed order is what lets independent workers shard the stream by
    stride and still cover every tile exactly once.
    """
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()                       # deterministic descent
            for fn in sorted(filenames):          # bounded: one dir's files (~per-slide)
                if fn.lower().endswith(IMG_EXTS):
                    yield os.path.join(dirpath, fn)
    else:
        with os.scandir(root) as it:
            names = sorted(e.name for e in it if e.is_file())
        for fn in names:
            if fn.lower().endswith(IMG_EXTS):
                yield os.path.join(root, fn)


def resolve_patch_root():
    """Return ``(root, recursive)`` for the first source that holds >=1 image --
    PFM_PATCH_DIR (recursive) then $PFM_TCGA_ROOT/thumbnails (flat) -- else
    ``(None, False)``. Non-emptiness is checked by pulling just the first path
    (O(1)); it never lists the directory."""
    if config.PATCH_DIR and os.path.isdir(config.PATCH_DIR):
        for _ in _iter_image_paths(config.PATCH_DIR, recursive=True):
            return config.PATCH_DIR, True
    thumb = os.path.join(config.TCGA_ROOT, "thumbnails")
    if os.path.isdir(thumb):
        for _ in _iter_image_paths(thumb, recursive=False):
            return thumb, False
    return None, False


def count_patch_images(root, recursive, limit=None):
    """Stream-count the images under ``root`` (O(1) memory). Honours ``limit``
    (or config.MAX_IMAGES; 0 = no cap) so the count matches what will be fed."""
    limit = limit or (config.MAX_IMAGES or None)
    n = 0
    for _ in _iter_image_paths(root, recursive):
        n += 1
        if limit and n >= limit:
            break
    return n


def find_patch_images(limit=None):
    """Backward-compatible helper: return a *list* of image paths (materialises
    it, so prefer the streaming dataset for the hot path). Honours ``limit`` or
    config.MAX_IMAGES (0 = all). Returns [] if nothing is found."""
    limit = limit or (config.MAX_IMAGES or None)
    root, recursive = resolve_patch_root()
    if root is None:
        return []
    out = []
    for i, p in enumerate(_iter_image_paths(root, recursive)):
        if limit and i >= limit:
            break
        out.append(p)
    return out


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
    import glob
    if config.SLIDE_FEATURES and os.path.isfile(config.SLIDE_FEATURES):
        return config.SLIDE_FEATURES
    hits = glob.glob(os.path.join(config.TCGA_ROOT, "**", "*.h5"), recursive=True)
    return sorted(hits)[0] if hits else None


def _open_rgb(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def make_streaming_dataset(transform, limit=None):
    """Build a streaming ``IterableDataset`` yielding (transformed_tensor, path).

    RAM is independent of the dataset size: tiles are decoded one at a time inside
    the DataLoader's bounded prefetch window, never preloaded. With ``num_workers``
    > 0 the path stream is sharded by stride so every tile is produced by exactly
    one worker (no duplicates, no omissions). Honours ``limit``/config.MAX_IMAGES
    (0 = all) as an EXACT global cap applied in the deterministic order.
    """
    import torch  # noqa: F401  (ensures torch present)
    from torch.utils.data import IterableDataset, get_worker_info

    limit = limit or (config.MAX_IMAGES or None)
    root, recursive = resolve_patch_root()

    class PatchIterableDataset(IterableDataset):
        def __init__(self):
            self.root = root
            self.recursive = recursive
            self.transform = transform
            self.limit = limit

        def __iter__(self):
            if self.root is None:
                return
            info = get_worker_info()
            wid = info.id if info is not None else 0
            nw = info.num_workers if info is not None else 1
            for i, path in enumerate(_iter_image_paths(self.root, self.recursive)):
                if self.limit and i >= self.limit:   # exact global cap (all workers agree on i)
                    break
                if (i % nw) != wid:                  # this tile belongs to another worker
                    continue
                yield self.transform(_open_rgb(path)), path

    return PatchIterableDataset()


def collate(batch):
    """Collate (tensor, path) pairs -> (stacked tensor, list-of-paths)."""
    import torch
    xs, paths = zip(*batch)
    return torch.stack(list(xs), 0), list(paths)
