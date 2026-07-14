"""Pre-decode per-slide JPEG patch tars into raw uint8 bins (CPU preprocessing).

Removes JPEG decode from the GPU run: each ``patches_tar/<slide>.tar`` is decoded ONCE
here into ``patches_raw/<slide>.bin`` -- N contiguous PATCH_HW x PATCH_HW x 3 uint8 patches
-- plus a ``<slide>.done`` sentinel (the patch count). The GPU dataset then memory-maps the
bin and slices patches with NO libjpeg (see pfm_common/data.py raw mode). Pixels are
bit-identical to decoding the JPEG, so embeddings are unchanged.

Parallelism (the read/decode/write pipeline): a PROCESS POOL over SLIDES saturates all
cores on the CPU-bound decode. Each worker streams one tar (one sequential READ), decodes
its members (byte-level: each JPEG is an independent stream), and writes one bin (one
sequential WRITE). With many slides in flight across the pool, the READS and WRITES of some
slides overlap the DECODES of others -- so disk I/O and CPU never idle waiting on each other,
which is the "reading/writing parallelized separately from decode" property. Resumable +
atomic per slide (write to .part, rename; skip a slide whose .bin + .done already exist).
"""
import io
import logging
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

PATCH_HW = 256                      # decoded patch side (must match patches.patch_size)
PATCH_BYTES = PATCH_HW * PATCH_HW * 3
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _decode_one_slide(args):
    """Decode one tar -> one raw .bin (+ .done). Runs in a worker process."""
    import numpy as np
    from PIL import Image

    tar_path, out_dir = args
    sid = os.path.basename(tar_path)[:-4]
    out_bin = os.path.join(out_dir, sid + ".bin")
    out_done = os.path.join(out_dir, sid + ".done")
    if os.path.exists(out_bin) and os.path.exists(out_done):
        try:
            return sid, int(open(out_done).read().strip()), "skipped"
        except ValueError:
            pass                    # corrupt sentinel -> re-decode
    tmp = out_bin + ".part"
    n = 0
    with tarfile.open(tar_path, "r") as tf, open(tmp, "wb", buffering=1 << 22) as w:
        for m in tf:                # sequential stream over the tar
            if not m.isfile() or not m.name.lower().endswith(_IMG_EXTS):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            a = np.asarray(img, dtype=np.uint8)
            if a.shape != (PATCH_HW, PATCH_HW, 3):   # guard: tiles should be exactly PATCH_HW
                a = np.asarray(img.resize((PATCH_HW, PATCH_HW)), dtype=np.uint8)
            w.write(np.ascontiguousarray(a).tobytes())
            n += 1
    os.replace(tmp, out_bin)        # atomic: a partial .part never looks finished
    with open(out_done, "w") as d:
        d.write(str(n))
    return sid, n, "decoded"


def decode_all(tars_dir, raw_dir, workers=None):
    """Decode every ``tars_dir/<slide>.tar`` -> ``raw_dir/<slide>.bin`` (+.done). Resumable.
    Returns a counts dict."""
    os.makedirs(raw_dir, exist_ok=True)
    tars = sorted(os.path.join(tars_dir, f) for f in os.listdir(tars_dir) if f.endswith(".tar")) \
        if os.path.isdir(tars_dir) else []
    workers = workers or os.cpu_count() or 4
    counts = {"decoded": 0, "skipped": 0, "patches": 0, "failed": 0, "slides": len(tars)}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_decode_one_slide, (t, raw_dir)) for t in tars]
        for fut in as_completed(futs):
            try:
                _sid, n, status = fut.result()
                counts[status] += 1
                counts["patches"] += n
            except Exception as e:                    # one bad slide never kills the pool
                counts["failed"] += 1
                logger.error("decode failed: %s", e)
    logger.info("decode_patches: %d decoded, %d already, %d patches, %d failed (%d slides, %d workers)",
                counts["decoded"], counts["skipped"], counts["patches"], counts["failed"],
                counts["slides"], workers)
    return counts
