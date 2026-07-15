"""Pre-decode per-slide JPEG patch tars into raw uint8 bins (CPU preprocessing).

Removes JPEG decode from the GPU run: each ``patches_tar/<slide>.tar`` is decoded ONCE
here into ``patches_raw/<slide>.bin`` -- N contiguous PATCH_HW x PATCH_HW x 3 uint8 patches
-- plus a ``<slide>.done`` sentinel (the patch count). The GPU dataset then memory-maps the
bin and slices patches with ZERO decode at train time (see pfm_common/data.py raw mode). The
raw pixels are bit-identical to decoding the JPEG, so embeddings are unchanged. Raw is ~14x
larger than JPEG on disk (~4 TB for the full run) -- that is the deliberate trade: disk is
cheap, GPU-starvation is not.

BYTE-LEVEL PARALLELISM (the point of this module)
-------------------------------------------------
A JPEG patch is an independent byte stream; decoding it (Huffman + IDCT + upsample) is pure
CPU. We saturate every core with a THREAD pool -- NOT one-process-per-core -- because the
decoder releases the GIL for the whole native decompress, so K threads decode K patches at
once on shared memory with no pickling/copy. Decoder preference (all GIL-free, all bundle
libjpeg-turbo's SIMD kernels, first importable wins):

    1. PyTurboJPEG  (thin libjpeg-turbo binding; fastest, needs system libturbojpeg)
    2. cv2.imdecode (OpenCV; wheel-bundled libjpeg-turbo, zero system deps -- the default)
    3. Pillow       (also drops the GIL around its C decode; slowest, always available)

READ / DECODE / WRITE are three overlapping stages, not one serial loop: a prefetch thread
does the sequential tar READ for slide N+1 while the shared thread pool DECODES slide N's
patches and the main thread streams the raw bytes to disk (sequential WRITE). So disk I/O and
CPU never idle waiting on each other -- "reading and writing parallelized separately from
decode." Resumable + atomic per slide: write ``<slide>.bin.part``, fsync, atomic rename, then
``<slide>.done``; a slide with ``.bin``+``.done`` is skipped, a slide killed mid-write left
only ``.part`` (never mistaken for done) and is redone next run.
"""
import io
import logging
import os
import queue
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

PATCH_HW = 256                      # decoded patch side (must match patches.patch_size)
PATCH_BYTES = PATCH_HW * PATCH_HW * 3
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_CHUNK = 2048                       # patches decoded in flight per slide (bounds RAM ~= _CHUNK*192KB)


def _build_decoder():
    """Return (name, decode_fn) where decode_fn(bytes) -> C-contiguous uint8[H,W,3] RGB.
    Picks the fastest GIL-free decoder importable in this environment (see module docstring)."""
    import numpy as np

    # 1. libjpeg-turbo direct binding -- fastest when the system lib is present.
    try:
        from turbojpeg import TJPF_RGB, TurboJPEG
        _tj = TurboJPEG()

        def _dec_turbo(buf):
            return np.ascontiguousarray(_tj.decode(buf, pixel_format=TJPF_RGB))

        return "PyTurboJPEG", _dec_turbo
    except Exception:
        pass

    # 2. OpenCV -- wheel-bundled libjpeg-turbo, no system deps, GIL released in imdecode.
    try:
        import cv2
        cv2.setNumThreads(0)              # we own the parallelism (threads over slides), not cv2
        _frombuffer = np.frombuffer

        def _dec_cv2(buf):
            a = cv2.imdecode(_frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)   # BGR uint8, GIL-free
            return np.ascontiguousarray(a[:, :, ::-1])                       # -> RGB, contiguous

        return "cv2", _dec_cv2
    except Exception:
        pass

    # 3. Pillow -- always available; also drops the GIL around the C decompress, just slower.
    from PIL import Image

    def _dec_pil(buf):
        return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"), dtype=np.uint8)

    return "Pillow", _dec_pil


def _read_tar_blobs(tar_path):
    """Sequential READ: pull every image member's compressed bytes out of one tar, in order.
    JPEG bytes are small (~10-30 KB), so the whole slide's compressed payload fits easily in
    RAM; the big raw arrays are produced downstream and streamed straight to disk."""
    blobs = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:                        # one sequential pass over the tar
            if not m.isfile() or not m.name.lower().endswith(_IMG_EXTS):
                continue
            f = tf.extractfile(m)
            if f is not None:
                blobs.append(f.read())
    return blobs


def _prefetch_slides(tars, raw_dir):
    """Yield (sid, tar_path, blobs_or_None) with a 1-deep READ-AHEAD thread: the next slide's
    tar is read off disk while the current slide is still decoding+writing. Already-done slides
    are yielded with blobs=None (a skip signal) without reading them."""
    q = queue.Queue(maxsize=1)

    def _reader():
        for t in tars:
            sid = os.path.basename(t)[:-4]
            out_bin = os.path.join(raw_dir, sid + ".bin")
            out_done = os.path.join(raw_dir, sid + ".done")
            if os.path.exists(out_bin) and os.path.exists(out_done):
                q.put((sid, t, None))       # resumable skip -- don't even read it
                continue
            try:
                q.put((sid, t, _read_tar_blobs(t)))
            except Exception as e:          # a corrupt tar shouldn't kill the pipeline
                q.put((sid, t, e))
        q.put(None)                         # sentinel: no more slides

    threading.Thread(target=_reader, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def _write_slide(blobs, out_bin, decode_fn, pool):
    """Decode all of one slide's patches on the shared thread pool (byte-level parallel, order
    preserved) and stream them to a single raw .bin. Atomic + resumable: writes .part, fsyncs,
    renames, drops a .done with the patch count. Returns the patch count."""
    import numpy as np

    tmp = out_bin + ".part"
    n = 0
    with open(tmp, "wb", buffering=1 << 22) as w:
        for i in range(0, len(blobs), _CHUNK):           # bound in-flight decoded RAM
            chunk = blobs[i:i + _CHUNK]
            for a in pool.map(decode_fn, chunk):         # K threads decode K patches at once
                if a.shape != (PATCH_HW, PATCH_HW, 3):   # guard: tiles should already be PATCH_HW
                    from PIL import Image
                    a = np.asarray(Image.fromarray(a).resize((PATCH_HW, PATCH_HW)), dtype=np.uint8)
                w.write(np.ascontiguousarray(a, dtype=np.uint8).tobytes())
                n += 1
        w.flush()
        os.fsync(w.fileno())                             # durable before we call it done
    os.replace(tmp, out_bin)                             # atomic: a partial .part never looks done
    with open(out_bin[:-4] + ".done", "w") as d:
        d.write(str(n))
    return n


def decode_all(tars_dir, raw_dir, workers=None):
    """Decode every ``tars_dir/<slide>.tar`` -> ``raw_dir/<slide>.bin`` (+ .done). Resumable
    (skips slides already done), atomic per slide, byte-level parallel over ``workers`` decode
    threads. Returns a counts dict."""
    os.makedirs(raw_dir, exist_ok=True)
    tars = sorted(os.path.join(tars_dir, f) for f in os.listdir(tars_dir) if f.endswith(".tar")) \
        if os.path.isdir(tars_dir) else []
    workers = workers or os.cpu_count() or 4
    decoder_name, decode_fn = _build_decoder()
    counts = {"decoded": 0, "skipped": 0, "patches": 0, "failed": 0, "slides": len(tars)}
    logger.info("decode_patches: %d slides -> %s via %s, %d decode threads",
                len(tars), raw_dir, decoder_name, workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sid, tar_path, blobs in _prefetch_slides(tars, raw_dir):
            if blobs is None:                            # already done (resumable skip)
                out_done = os.path.join(raw_dir, sid + ".done")
                try:
                    counts["patches"] += int(open(out_done).read().strip())
                except (OSError, ValueError):
                    pass
                counts["skipped"] += 1
                continue
            if isinstance(blobs, Exception):             # unreadable tar
                counts["failed"] += 1
                logger.error("decode: unreadable tar %s: %s", tar_path, blobs)
                continue
            try:
                n = _write_slide(blobs, os.path.join(raw_dir, sid + ".bin"), decode_fn, pool)
                counts["decoded"] += 1
                counts["patches"] += n
            except Exception as e:                       # one bad slide never kills the run
                counts["failed"] += 1
                logger.error("decode failed for %s: %s", sid, e)

    logger.info("decode_patches: %d decoded, %d already, %d patches, %d failed (%d slides, %s, %d threads)",
                counts["decoded"], counts["skipped"], counts["patches"], counts["failed"],
                counts["slides"], decoder_name, workers)
    return counts
