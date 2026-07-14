"""TCGA data access, shared by every patch-encoder model.

Patch encoders consume small RGB tiles. Where those tiles come from is resolved here
in priority order so a model script never has to care:

    1. PFM_PATCH_DIR holding per-slide `.tar` shards  (preferred -- see MEMORY MODEL)
    2. PFM_PATCH_DIR holding loose patch image files   (fallback)
    3. $PFM_TCGA_ROOT/thumbnails loose images          (last-resort fallback)

If none yield images, resolve_patch_root() returns (None, ...) and the caller reports a
clear "stage your data" message instead of crashing.

MEMORY MODEL -- constant in the dataset size N
-----------------------------------------------
Tiles are NEVER materialised in RAM as a whole. A streaming ``IterableDataset`` hands one
decoded tile at a time to a bounded DataLoader prefetch queue, so host RAM holds only the
in-flight window (num_workers x prefetch_factor x batch_size x bytes_per_tile) -- the same
whether there are 226k or 22M tiles.

WHY TAR SHARDS
--------------
Tiling persists ~16k tiny JPGs/slide (~2.26M files). Per-run staging/reading of that many
files is a Lustre metadata storm (it timed the job out in STEP 4). ``tcga/pack_patches``
packs each slide into one ``patches_tar/<slide_id>.tar``; here we stream tar MEMBERS. That
turns millions of metadata ops into ~N_slides big sequential reads.

Correctness with multiple workers:
  * tar mode  -- each worker takes a disjoint set of tars (``tar_index % num_workers ==
    worker_id``) and streams them whole, so every tile is produced by exactly one worker.
  * loose mode -- a deterministic global walk, kept at stride, then survivors sharded by
    worker. Either way: no duplicates, no omissions.

PFM_PATCH_STRIDE keeps every Nth tile (1 = all; the mini run sets 10 for a 1/10 sample).
"""
import io
import os

from . import config

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ── loose-file streaming (fallback: thumbnails, or unpacked patches) ─────────────
def _iter_image_paths(root, recursive=True):
    """Yield image paths under ``root`` one at a time in a DETERMINISTIC order, using
    O(1) memory in N (only one directory's names held at a time)."""
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for fn in sorted(filenames):
                if fn.lower().endswith(IMG_EXTS):
                    yield os.path.join(dirpath, fn)
    else:
        with os.scandir(root) as it:
            names = sorted(e.name for e in it if e.is_file())
        for fn in names:
            if fn.lower().endswith(IMG_EXTS):
                yield os.path.join(root, fn)


# ── tar-shard streaming (preferred) ──────────────────────────────────────────────
def _iter_tars(tars_dir):
    """Sorted per-slide `.tar` shard paths under ``tars_dir`` (deterministic order)."""
    with os.scandir(tars_dir) as it:
        names = sorted(e.name for e in it if e.is_file() and e.name.endswith(".tar"))
    for n in names:
        yield os.path.join(tars_dir, n)


def _dir_has_tars(d):
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".tar"):
                return True
    return False


def _balanced_shard_assign(tars, shard_count):
    """Assign each tar to a GPU shard so total PATCHES per shard are balanced (LPT greedy),
    using the per-slide counts in the `<tar>.done` sentinels (pack/tiler wrote them). Sharding
    by tar *index* gives uneven work because slides vary a lot in patch count; this balances by
    weight instead. Deterministic (every process computes the same map). Missing sentinel ->
    weight 1. Returns {tar_path: shard_index}."""
    weighted = []
    for t in tars:
        try:
            n = int(open(t[:-4] + ".done").read().strip())
        except (OSError, ValueError):
            n = 1
        weighted.append((n, t))
    weighted.sort(key=lambda x: (-x[0], x[1]))        # heaviest first, path tie-break
    loads = [0] * shard_count
    assign = {}
    for n, t in weighted:
        g = min(range(shard_count), key=lambda i: (loads[i], i))
        assign[t] = g
        loads[g] += n
    return assign


def resolve_patch_root():
    """Return ``(mode, root, recursive)``:
        ('tars',  dir, None)  -- PFM_PATCH_DIR holds per-slide .tar shards
        ('loose', dir, bool)  -- PFM_PATCH_DIR / thumbnails hold loose images
        (None, None, None)    -- nothing found
    Non-emptiness is checked cheaply (first tar / first image), never a full listing."""
    pd = config.PATCH_DIR
    if pd and os.path.isdir(pd):
        if _dir_has_tars(pd):
            return "tars", pd, None
        for _ in _iter_image_paths(pd, recursive=True):
            return "loose", pd, True
    thumb = os.path.join(config.TCGA_ROOT, "thumbnails")
    if os.path.isdir(thumb):
        for _ in _iter_image_paths(thumb, recursive=False):
            return "loose", thumb, False
    return None, None, None


def count_patch_images(mode, root, recursive, stride=1):
    """Total tiles that WILL be fed (after stride), O(1) memory. In tar mode uses the
    `.done` sentinel counts pack_patches wrote (falls back to reading tar headers)."""
    import tarfile
    stride = max(1, stride)
    total = 0
    if mode == "tars":
        for tarpath in _iter_tars(root):
            n = None
            done = tarpath[:-4] + ".done"
            try:
                n = int(open(done).read().strip())
            except (OSError, ValueError):
                with tarfile.open(tarpath) as tf:
                    n = sum(1 for m in tf.getmembers() if m.isfile())
            total += (n + stride - 1) // stride          # ceil(n/stride) kept per slide
    elif mode == "loose":
        i = 0
        for _ in _iter_image_paths(root, recursive):
            if i % stride == 0:
                total += 1
            i += 1
    return total


def find_patch_images(limit=None):
    """Backward-compatible helper: a *list* of loose image paths (materialises it).
    Returns [] in tar mode or when nothing is found. Prefer the streaming dataset."""
    limit = limit or (config.MAX_IMAGES or None)
    mode, root, recursive = resolve_patch_root()
    if mode != "loose":
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
    Priority: PFM_SLIDE_FEATURES file -> first *.h5 under $PFM_TCGA_ROOT. Or None."""
    import glob
    if config.SLIDE_FEATURES and os.path.isfile(config.SLIDE_FEATURES):
        return config.SLIDE_FEATURES
    hits = glob.glob(os.path.join(config.TCGA_ROOT, "**", "*.h5"), recursive=True)
    return sorted(hits)[0] if hits else None


def _open_rgb(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def make_streaming_dataset(transform, stride=None):
    """Streaming ``IterableDataset`` yielding (transformed_tensor, path). RAM is
    independent of N (bounded prefetch, never preloaded). ``stride`` (or
    config.PATCH_STRIDE) keeps every Nth tile. Worker sharding guarantees each tile is
    produced by exactly one worker (no dup/drop). tar mode when shards exist, else loose."""
    import tarfile

    from PIL import Image
    from torch.utils.data import IterableDataset, get_worker_info

    stride = max(1, int(stride if stride is not None else config.PATCH_STRIDE))
    limit = config.MAX_IMAGES or None
    # Data-parallel sharding across GPU PROCESSES (config, set per-GPU by the runner), on top
    # of DataLoader-WORKER sharding inside each process. shard_count==1 => no process sharding.
    shard_index = config.SHARD_INDEX
    shard_count = max(1, config.SHARD_COUNT)
    mode, root, recursive = resolve_patch_root()
    # Balance tars across GPU shards by patch count (not tar count) so no shard straggles.
    tar_shard = _balanced_shard_assign(list(_iter_tars(root)), shard_count) \
        if (mode == "tars" and shard_count > 1) else None

    class PatchIterableDataset(IterableDataset):
        def __iter__(self):
            if root is None:
                return
            info = get_worker_info()
            wid = info.id if info is not None else 0
            nw = info.num_workers if info is not None else 1
            if mode == "tars":
                yield from self._iter_tars(wid, nw)
            else:
                yield from self._iter_loose(wid, nw)

        def _iter_tars(self, wid, nw):
            # Two levels: this GPU process takes the tars assigned to its shard (a DISJOINT,
            # patch-count-balanced set of whole slides); among THOSE, worker wid takes every
            # nw-th. Union over all (process, worker) pairs = every tar exactly once.
            j = 0
            for tarpath in _iter_tars(root):
                if tar_shard is not None and tar_shard.get(tarpath, 0) != shard_index:
                    continue
                if (j % nw) == wid:
                    with tarfile.open(tarpath, "r") as tf:
                        mi = 0
                        for m in tf:              # sequential -> streaming, O(1) memory
                            if not m.isfile() or not m.name.lower().endswith(IMG_EXTS):
                                continue
                            if (mi % stride) == 0:
                                f = tf.extractfile(m)
                                if f is not None:
                                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                                    yield transform(img), m.name
                            mi += 1
                j += 1

        def _iter_loose(self, wid, nw):
            # Fallback (thumbnails / unpacked). Assign each stride-survivor to exactly one
            # (process, worker) consumer: total = shard_count*nw, my id = shard_index*nw+wid.
            total = shard_count * nw
            me = shard_index * nw + wid
            kept = 0
            for i, path in enumerate(_iter_image_paths(root, recursive)):
                if limit and i >= limit:
                    break
                if (i % stride) != 0:             # 1/stride global sample (deterministic)
                    continue
                if (kept % total) == me:
                    yield transform(_open_rgb(path)), path
                kept += 1

    return PatchIterableDataset()


def collate(batch):
    """Collate (tensor, path) pairs -> (stacked tensor, list-of-paths)."""
    import torch
    xs, paths = zip(*batch)
    return torch.stack(list(xs), 0), list(paths)
