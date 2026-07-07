"""Tile a full TCGA SVS into tissue-only patches (the patch-level preprocessing).

Turns each whole-slide image into many small patches so downstream extraction is
GPU-bound (100k+ forward passes) rather than CPU/IO-bound (1 thumbnail/slide).

Standard WSI tiling: lay a grid over the slide at the working level, keep only
grid cells that contain enough TISSUE (a downsampled saturation/brightness mask
drops the white glass background), and cut each kept cell to a patch. Depends only
on numpy + PIL + openslide (all in the tcga_build venv) -- no cv2/skimage.

The patches are the furnace fuel: written to node-local, consumed batch-by-batch by
the GPU, then discarded. Only the per-slide/patch EMBEDDINGS persist, never the raw
patches or SVS.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _otsu_threshold(values):
    """Otsu's method: the 0-255 threshold that maximizes between-class variance of
    a histogram. Pure numpy (no cv2/skimage). Returns the threshold as an int."""
    hist = np.bincount(values.ravel().astype(np.intp), minlength=256)[:256].astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127
    idx = np.arange(256)
    w0 = np.cumsum(hist)                 # weight of "background" class (<= t)
    w1 = total - w0
    isum = np.cumsum(hist * idx)
    total_mu = isum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, isum / w0, 0.0)
        mu1 = np.where(w1 > 0, (total_mu - isum) / w1, 0.0)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2  # between-class variance per threshold
    sigma_b[(w0 == 0) | (w1 == 0)] = 0
    return int(np.argmax(sigma_b))


def _tissue_mask(thumb_rgb, dark_min=15):
    """Boolean tissue mask over a downsampled RGB thumbnail (numpy [H,W,3], uint8).

    Tissue is colored (nonzero saturation); white glass background is near-zero
    saturation. We compute a cheap saturation proxy (max-min across channels) and
    pick the tissue/background cut with **Otsu** per-slide, so it adapts to each
    slide's staining/brightness instead of a fixed threshold. A dark-pixel guard
    drops ink/pen artifacts. cv2/skimage-free.
    """
    arr = thumb_rgb.astype(np.int16)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = (mx - mn).astype(np.uint8)    # saturation proxy
    thr = _otsu_threshold(sat)
    return (sat > thr) & (mx > dark_min)


def tile_slide(svs_path, out_dir, slide_id, patch_size=256, level=0,
               tissue_thresh=0.10, thumb_max_dim=2048, jpeg_quality=85):
    """Cut ``svs_path`` into tissue patches under ``out_dir/<slide_id>/``.

    Grid step = ``patch_size`` at ``level``. **Every** cell whose tissue fraction (from
    the Otsu mask) exceeds ``tissue_thresh`` is kept -- no cap, no subsampling. Each kept
    cell is read at ``level`` and saved as ``<slide_id>__x<X>_y<Y>.jpg`` (X,Y are level-0
    coords, so patches carry their provenance).

    Returns the number of patches written. Resumable: if the slide dir already holds
    patches, returns that count without re-tiling.
    """
    import openslide

    out_dir = Path(out_dir) / slide_id
    existing = sorted(out_dir.glob("*.jpg")) if out_dir.exists() else []
    if existing:
        return len(existing)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(svs_path))
    try:
        level = min(level, slide.level_count - 1)
        W, H = slide.level_dimensions[level]
        ds_l = slide.level_downsamples[level]            # level -> level-0 scale

        # Downsampled thumbnail for the tissue mask; map grid cells onto it.
        scale = max(W, H) / float(thumb_max_dim)
        scale = max(scale, 1.0)
        tw, th = int(W / scale), int(H / scale)
        thumb = slide.get_thumbnail((tw, th)).convert("RGB")
        mask = _tissue_mask(np.asarray(thumb))
        mh, mw = mask.shape
        # thumbnail px per patch-side (patch_size at `level` -> thumbnail scale)
        pm = max(1, int(round(patch_size / scale)))

        nx, ny = W // patch_size, H // patch_size
        candidates = []
        for gy in range(ny):
            for gx in range(nx):
                ty0, tx0 = int(gy * pm), int(gx * pm)
                cell = mask[ty0:min(ty0 + pm, mh), tx0:min(tx0 + pm, mw)]
                if cell.size and cell.mean() >= tissue_thresh:
                    candidates.append((gx, gy))

        written = 0
        for gx, gy in candidates:
            x0 = int(gx * patch_size * ds_l)             # level-0 coords for read_region
            y0 = int(gy * patch_size * ds_l)
            patch = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB")
            patch.save(out_dir / f"{slide_id}__x{x0}_y{y0}.jpg", "JPEG", quality=jpeg_quality)
            written += 1
    finally:
        slide.close()

    logger.info("tiled %s -> %d tissue patches (level=%d, %dpx)", slide_id, written,
                level, patch_size)
    return written
