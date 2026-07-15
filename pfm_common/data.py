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

# Raw pre-decoded patches (tcga/decode_patches.py): patches_raw/<slide>.bin = N contiguous
# RAW_HW x RAW_HW x 3 uint8 patches, ZERO decode at train time (memmap + slice). Preferred
# input when present -- it removes libjpeg from the GPU run entirely.
RAW_HW = 256
RAW_BYTES = RAW_HW * RAW_HW * 3


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


# ── raw pre-decoded streaming (preferred when present) ───────────────────────────
def _iter_bins(bins_dir):
    """Sorted per-slide `.bin` paths under ``bins_dir`` (deterministic order)."""
    with os.scandir(bins_dir) as it:
        names = sorted(e.name for e in it if e.is_file() and e.name.endswith(".bin"))
    for n in names:
        yield os.path.join(bins_dir, n)


def _dir_has_bins(d):
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".bin"):
                return True
    return False


def _bin_patch_count(bin_path):
    """Patches in one raw bin: the `.done` sentinel if present, else filesize / RAW_BYTES."""
    try:
        return int(open(bin_path[:-4] + ".done").read().strip())
    except (OSError, ValueError):
        try:
            return os.path.getsize(bin_path) // RAW_BYTES
        except OSError:
            return 0


def _under(path, base):
    """True if `path` is inside directory `base` (both made absolute)."""
    if not base:
        return False
    a = os.path.abspath(path)
    b = os.path.abspath(base)
    return a == b or a.startswith(b + os.sep)


def _rawcache_dir(source_root):
    """Resolve the node-local SSD dir to stream raw bins through, or None to read Lustre direct.
    None when: disabled (PFM_RAWCACHE=off), no dir configured, the dir can't be made, or the
    source bins are ALREADY node-local (no point copying SSD->SSD)."""
    mode = str(config.RAWCACHE).lower()
    if mode in ("0", "off", "false", "no", "none"):
        return None
    base = config.RAWCACHE_DIR
    if not base or _under(source_root, config.LSCRATCH):   # source already on the SSD -> read direct
        return None
    try:
        os.makedirs(base, exist_ok=True)
        return base
    except OSError:
        return None


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
        ('raw',   dir, None)  -- PFM_PATCH_DIR holds per-slide .bin (pre-decoded; ZERO decode)
        ('tars',  dir, None)  -- PFM_PATCH_DIR holds per-slide .tar shards
        ('loose', dir, bool)  -- PFM_PATCH_DIR / thumbnails hold loose images
        (None, None, None)    -- nothing found
    Priority raw > tars > loose (raw is fastest to feed). Non-emptiness is checked cheaply
    (first bin / tar / image), never a full listing."""
    pd = config.PATCH_DIR
    if pd and os.path.isdir(pd):
        if _dir_has_bins(pd):
            return "raw", pd, None
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
    if mode == "raw":
        for binpath in _iter_bins(root):
            n = _bin_patch_count(binpath)
            total += (n + stride - 1) // stride          # ceil(n/stride) kept per slide
    elif mode == "tars":
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
    # Balance whole slides across GPU shards by patch count (not file count) so no shard
    # straggles -- same LPT logic for tar shards and raw bins (both have <slide>.done counts).
    if shard_count > 1 and mode == "tars":
        file_shard = _balanced_shard_assign(list(_iter_tars(root)), shard_count)
    elif shard_count > 1 and mode == "raw":
        file_shard = _balanced_shard_assign(list(_iter_bins(root)), shard_count)
    else:
        file_shard = None

    class PatchIterableDataset(IterableDataset):
        def __iter__(self):
            if root is None:
                return
            info = get_worker_info()
            wid = info.id if info is not None else 0
            nw = info.num_workers if info is not None else 1
            if mode == "raw":
                yield from self._iter_raw(wid, nw)
            elif mode == "tars":
                yield from self._iter_tars(wid, nw)
            else:
                yield from self._iter_loose(wid, nw)

        def _iter_raw(self, wid, nw):
            # Pre-decoded: memmap each <slide>.bin and slice patch i as uint8[H,W,3] -- NO libjpeg.
            # Sharding mirrors tar mode (this GPU takes its balanced slide set; among those, worker
            # wid takes every nw-th slide), so every patch is produced exactly once. The yielded
            # name is "<sid>__<i>.raw" so runner._sid() recovers the slide_id.
            #
            # STREAMING SSD CACHE: the raw bins live on Lustre (too big for the SSD in full). This
            # worker copies the NEXT slide's bin onto the node-local SSD while it reads the current
            # one (depth-1 prefetch), memmaps from the SSD (pages -> DRAM on access), then DELETES
            # the SSD copy -- a bounded ~2-bin window per worker, never the whole 4 TB. If the SSD
            # is (near) full the copy is skipped and that slide is memmapped straight from Lustre
            # (still zero-decode). See _rawcache_dir / config.RAWCACHE*.
            import numpy as np
            import shutil
            from concurrent.futures import ThreadPoolExecutor

            myslides = []
            j = 0
            for binpath in _iter_bins(root):
                if file_shard is not None and file_shard.get(binpath, 0) != shard_index:
                    continue
                if (j % nw) == wid:
                    myslides.append(binpath)
                j += 1
            if not myslides:
                return

            cache_dir = _rawcache_dir(root)
            stager = ThreadPoolExecutor(max_workers=1) if cache_dir else None

            def _stage(src):
                """Copy one Lustre bin -> SSD (atomic); return local path, or None to read Lustre
                direct (SSD full / copy failed / size mismatch). Self-limiting on free space."""
                need = _bin_patch_count(src) * RAW_BYTES
                try:
                    if need <= 0 or shutil.disk_usage(cache_dir).free < int(need * 1.15):
                        return None                       # leave headroom; fall back to Lustre
                    dst = os.path.join(cache_dir, os.path.basename(src))
                    tmp = "%s.tmp%d" % (dst, os.getpid())
                    shutil.copyfile(src, tmp)             # sequential Lustre read -> SSD write
                    if os.path.getsize(tmp) != need:
                        os.remove(tmp)
                        return None
                    os.replace(tmp, dst)                  # atomic: a partial copy never looks whole
                    return dst
                except OSError:                            # ENOSPC / transient FS error -> Lustre
                    return None

            def _evict(path):
                if path and cache_dir and _under(path, cache_dir):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            fut = stager.submit(_stage, myslides[0]) if stager else None
            try:
                for k, src in enumerate(myslides):
                    local = fut.result() if fut else None
                    # kick off the NEXT slide's copy so it overlaps this slide's reads (depth 1)
                    fut = stager.submit(_stage, myslides[k + 1]) \
                        if (stager and k + 1 < len(myslides)) else None
                    sid = os.path.basename(src)[:-4]
                    n = _bin_patch_count(src)              # count from Lustre .done (canonical)
                    read_path = local or src               # SSD copy if staged, else Lustre direct
                    if n <= 0:
                        _evict(local)
                        continue
                    mm = np.memmap(read_path, dtype=np.uint8, mode="r",
                                   shape=(n, RAW_HW, RAW_HW, 3))
                    try:
                        for i in range(0, n, stride):      # 1/stride sample, same as tar mode
                            # np.array (not asarray) COPIES the 192 KB patch into DRAM, decoupled
                            # from the memmap -- so del mm + evict can't invalidate an in-flight patch.
                            img = Image.fromarray(np.array(mm[i]))     # DRAM copy, no decode
                            yield transform(img), "%s__%d.raw" % (sid, i)
                    finally:
                        del mm
                        _evict(local)                      # free the SSD window immediately
            finally:
                if stager:
                    stager.shutdown(wait=False)

        def _iter_tars(self, wid, nw):
            # Two levels: this GPU process takes the tars assigned to its shard (a DISJOINT,
            # patch-count-balanced set of whole slides); among THOSE, worker wid takes every
            # nw-th. Union over all (process, worker) pairs = every tar exactly once.
            j = 0
            for tarpath in _iter_tars(root):
                if file_shard is not None and file_shard.get(tarpath, 0) != shard_index:
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
