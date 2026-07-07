"""Two-tier staged loading of FULL TCGA slides.

Persistent tier ($SCRATCH): a stratified ~N GB subset of full SVS files, kept as
a reusable on-scratch cache (the "10% of TCGA" the user wants resident).

Ephemeral tier ($L_SCRATCH / $TMPDIR, node-local SSD): each SVS is copied here
just before it is read, thumbnailed with openslide, then evicted -- so openslide
gets fast local random access and local disk never holds more than ``workers``
slides at once.

Async + resumable:
  * download (scratch) and stage+thumbnail overlap ACROSS slides (a thread pool
    keeps several slides in flight: some downloading, others thumbnailing);
  * a slide whose thumbnail already exists is skipped entirely;
  * a slide whose SVS is already cached on scratch is not re-downloaded.

So the "final setup" downloads the subset only if necessary; otherwise it simply
accesses the cached SVS / existing thumbnails and proceeds to training.
"""

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


def _download_svs(file_id, filename, cache_dir, token=None, session=None, chunk=1 << 20,
                  retries=4):
    """Download a full SVS to ``cache_dir/<file_id>/<filename>`` on $SCRATCH.

    Resumable-skip: if the file is already present and non-empty, returns it
    without re-downloading. Writes to a .part file and renames on completion so
    an interrupted download is never mistaken for a finished one.

    Retries on transient network failures (streaming a full SVS over the WAN can
    ``IncompleteRead`` / drop the connection); the partial .part is discarded and the
    download restarts, with exponential-ish backoff. Raises after ``retries`` attempts.
    """
    sess = session or requests.Session()
    headers = {"X-Auth-Token": token} if token else {}
    d = Path(cache_dir) / file_id
    d.mkdir(parents=True, exist_ok=True)
    out = d / (filename or f"{file_id}.svs")
    if out.exists() and out.stat().st_size > 0:
        return out
    tmp = out.with_name(out.name + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with sess.get(f"{GDC_DATA_URL}/{file_id}", headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for c in r.iter_content(chunk_size=chunk):
                        if c:
                            f.write(c)
            tmp.rename(out)
            return out
        except Exception as e:                       # transient: IncompleteRead, ConnectionError, timeout
            last_err = e
            try:
                tmp.unlink()                         # discard the partial before retrying
            except OSError:
                pass
            if attempt < retries:
                logger.warning("download %s attempt %d/%d failed (%s) -- retrying",
                               file_id, attempt, retries, e)
                time.sleep(2 * attempt)
    raise last_err


def _evict(path):
    """Remove a staged file or its per-slide dir from node-local temp. Best-effort."""
    p = Path(path)
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    except OSError:
        pass


def _persist_thumbnail(local_jpg, out_jpg):
    """Copy a node-local thumbnail to its durable $SCRATCH home, then drop the
    node-local copy. Runs OFF the hot path (a background pool) so the slow Lustre
    write never stalls the download+thumbnail of the next slide."""
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_jpg, out_jpg)
    _evict(local_jpg)


def _render_thumbnail(svs_path, out_jpg, size):
    """Render an openslide thumbnail of ``svs_path`` to ``out_jpg`` (on $SCRATCH).

    Reads the SVS from wherever it already lives (node-local stage or scratch
    cache); does not copy or evict anything itself."""
    import openslide

    slide = openslide.OpenSlide(str(svs_path))
    thumb = slide.get_thumbnail(tuple(size))
    slide.close()
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(out_jpg, "JPEG", quality=85)


def _thumbnail_via_stage(svs_path, stage_dir, out_jpg, size):
    """Copy the SVS to node-local ``stage_dir``, render an openslide thumbnail to
    ``out_jpg`` (on $SCRATCH), then evict the local copy. Bounded local disk."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    local = stage_dir / Path(svs_path).name
    shutil.copy2(svs_path, local)
    try:
        _render_thumbnail(local, out_jpg, size)
    finally:
        try:
            local.unlink()
        except OSError:
            pass


@dataclass
class StageResult:
    df: object            # DataFrame with a 'jpg_path' column added
    downloaded: int = 0   # SVS fetched to the scratch cache this run
    reused: int = 0       # SVS already cached on scratch (not re-downloaded)
    processed: int = 0    # thumbnails produced this run (downloaded + reused)
    skipped: int = 0      # thumbnail already existed -> nothing to do
    failed: int = 0


def acquire_stage_process(df, cache_dir, stage_dir, thumbnails_dir, size=(512, 512),
                          workers=6, token=None, stream_to_local=False,
                          file_id_col="file_id", filename_col="filename",
                          slide_id_col="slide_id"):
    """Ensure a full-SVS subset is thumbnailed via a node-local staging area,
    concurrently and resumably. Only the tiny thumbnails ever persist on $SCRATCH.

    Two acquisition modes for the full SVS:

    * ``stream_to_local=False`` (two-tier cache): download the SVS to a reusable
      persistent cache on $SCRATCH (``cache_dir``), then copy it to node-local
      ``stage_dir`` to thumbnail, evicting only the node-local copy. The ~50 GB
      of SVS stays resident on scratch for fast re-runs.
    * ``stream_to_local=True`` (stream through TMPDIR): download the SVS DIRECTLY
      into node-local ``stage_dir`` (a fast SSD, e.g. $L_SCRATCH/$TMPDIR),
      thumbnail it in place, then evict it. Nothing full-size persists -- the run
      never relies on a pre-downloaded scratch cache; the thumbnails are the only
      lasting artifact.

    Per slide (bounded thread pool, so downloads and thumbnailing overlap):
      thumbnail exists?                 -> skip
      stream_to_local?                  -> stream SVS to node-local, thumbnail, evict
      else SVS cached on scratch?       -> stage->thumbnail  (reused)
      else                              -> download to scratch, stage->thumbnail (downloaded)

    Adds a ``jpg_path`` column and returns a StageResult.
    """
    cache_dir = Path(cache_dir)
    thumbnails_dir = Path(thumbnails_dir)
    stage_dir = Path(stage_dir)
    if not stream_to_local:
        cache_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    jpg_by_id = {}
    counts = {"downloaded": 0, "reused": 0, "processed": 0, "skipped": 0, "failed": 0}

    # Background pool that copies finished node-local thumbnails to their durable
    # $SCRATCH home. In stream_to_local mode the hot path (download SVS -> node-local,
    # thumbnail node-local, evict SVS) hands the tiny JPG here and moves straight on
    # to the next slide, so the slow Lustre write to the tcga folder overlaps with
    # the next download instead of blocking it.
    persist_pool = ThreadPoolExecutor(max_workers=4) if stream_to_local else None
    persist_futs = []

    def work(row):
        sid = row[slide_id_col]
        fid = row[file_id_col]
        out = thumbnails_dir / f"{sid}.jpg"
        if out.exists():
            return sid, out, "skipped", False, None
        if not fid:
            return sid, None, "failed", False, None
        try:
            # HYBRID, "use the download if available": if the full SVS is already in
            # the persistent $SCRATCH cache (populated by jobs/download_tcga.sh),
            # thumbnail from it with no network -- regardless of stream_to_local.
            cached = (cache_dir / fid).is_dir() and any((cache_dir / fid).iterdir())
            if cached:
                svs = _download_svs(fid, row.get(filename_col), cache_dir, token, session)  # returns cached path
                _thumbnail_via_stage(svs, stage_dir, out, size)
                return sid, out, "processed", False, None
            if stream_to_local:
                # Not pre-downloaded -> stream the full SVS into node-local temp,
                # thumbnail it LOCALLY (no persistent copy), evict. The node-local JPG
                # is returned so the main loop queues an async copy to $SCRATCH.
                svs = _download_svs(fid, row.get(filename_col), stage_dir, token, session)
                local_jpg = stage_dir / f"{sid}.jpg"
                try:
                    _render_thumbnail(svs, local_jpg, size)
                finally:
                    _evict(svs.parent)
                return sid, out, "processed", True, local_jpg
            # two-tier cache mode: download to the persistent cache, then stage+thumbnail
            svs = _download_svs(fid, row.get(filename_col), cache_dir, token, session)
            _thumbnail_via_stage(svs, stage_dir, out, size)
            return sid, out, "processed", True, None
        except Exception as e:
            logger.error("stage/process failed for %s (%s): %s", sid, fid, e)
            return sid, None, "failed", False, None

    rows = [r for _, r in df.iterrows()]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in as_completed([ex.submit(work, r) for r in rows]):
            sid, out, status, downloaded, local_jpg = fut.result()
            jpg_by_id[sid] = out
            if status == "processed":
                counts["processed"] += 1
                counts["downloaded" if downloaded else "reused"] += 1
                if local_jpg is not None and persist_pool is not None:
                    persist_futs.append(persist_pool.submit(_persist_thumbnail, local_jpg, out))
            elif status == "skipped":
                counts["skipped"] += 1
            else:
                counts["failed"] += 1

    # Drain the background persistence so every thumbnail is on $SCRATCH (durable +
    # resumable) and no node-local copy is left behind before we return.
    if persist_pool is not None:
        for f in as_completed(persist_futs):
            try:
                f.result()
            except Exception as e:
                logger.error("thumbnail persist failed: %s", e)
        persist_pool.shutdown(wait=True)

    logger.info("stage_process: %d processed (%d downloaded, %d reused-cache), "
                "%d already-thumbnailed, %d failed",
                counts["processed"], counts["downloaded"], counts["reused"],
                counts["skipped"], counts["failed"])

    df = df.copy()
    df["jpg_path"] = df[slide_id_col].map(lambda s: jpg_by_id.get(s))
    return StageResult(df=df, downloaded=counts["downloaded"], reused=counts["reused"],
                       processed=counts["processed"], skipped=counts["skipped"],
                       failed=counts["failed"])


def predownload_svs(df, cache_dir, workers=6, token=None, file_id_col="file_id",
                    filename_col="filename"):
    """Download the FULL SVS of every row into the persistent $SCRATCH cache at
    ``cache_dir/<file_id>/<filename>``, concurrently and resumably (a file already
    present is not re-downloaded).

    This is the "download all of TCGA" path (jobs/download_tcga.sh). Run it once and
    a later staged run reuses these cached SVS (thumbnailing from disk, no network)
    via the hybrid in ``acquire_stage_process``; slides NOT pre-downloaded are
    streamed on demand instead. Returns a counts dict.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    counts = {"downloaded": 0, "reused": 0, "failed": 0}

    def work(row):
        fid = row[file_id_col]
        if not fid:
            return "failed"
        try:
            already = (cache_dir / fid).is_dir() and any((cache_dir / fid).iterdir())
            _download_svs(fid, row.get(filename_col), cache_dir, token, session)
            return "reused" if already else "downloaded"
        except Exception as e:
            logger.error("predownload failed for %s: %s", fid, e)
            return "failed"

    rows = [r for _, r in df.iterrows()]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in as_completed([ex.submit(work, r) for r in rows]):
            counts[fut.result()] += 1

    logger.info("predownload_svs: %d downloaded, %d already-cached, %d failed",
                counts["downloaded"], counts["reused"], counts["failed"])
    return counts


def acquire_tile_process(df, cache_dir, stage_dir, patches_dir, patch_size=256,
                         level=0, tissue_thresh=0.10, thumb_max_dim=2048, jpeg_quality=85,
                         workers=6, token=None, stream_to_local=True, file_id_col="file_id",
                         filename_col="filename", slide_id_col="slide_id"):
    """Acquire each SVS (reuse cached, else stream to node-local) and TILE it into
    tissue patches under ``patches_dir/<slide_id>/`` -- the patch-level preprocessing
    that makes extraction GPU-bound.

    Fallback chain per slide (resumable): patches already present -> skip (load them,
    no re-tile); else SVS pre-downloaded in the $SCRATCH cache -> tile from it (no
    download); else (stream_to_local) stream the SVS into node-local temp, tile, evict.
    Patches PERSIST in ``patches_dir`` (on $SCRATCH) so tiling is a one-time cost and
    later runs reuse them. Concurrent. Returns a counts dict.
    """
    from tcga.slide_tiler import tile_slide

    cache_dir = Path(cache_dir)
    stage_dir = Path(stage_dir)
    patches_dir = Path(patches_dir)
    if not stream_to_local:
        cache_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    counts = {"slides": 0, "patches": 0, "reused": 0, "skipped": 0, "failed": 0}

    def work(row):
        sid = row[slide_id_col]
        fid = row[file_id_col]
        sdir = patches_dir / sid
        if sdir.exists() and any(sdir.glob("*.jpg")):
            return sid, "skipped", len(list(sdir.glob("*.jpg")))
        if not fid:
            return sid, "failed", 0
        try:
            cached = (cache_dir / fid).is_dir() and any((cache_dir / fid).iterdir())
            if cached or not stream_to_local:
                # pre-downloaded (or two-tier) -> tile from the persistent cache, keep it
                svs = _download_svs(fid, row.get(filename_col), cache_dir, token, session)
                n = tile_slide(svs, patches_dir, sid, patch_size=patch_size, level=level,
                               tissue_thresh=tissue_thresh, thumb_max_dim=thumb_max_dim,
                               jpeg_quality=jpeg_quality)
                return sid, ("reused" if cached else "downloaded"), n
            # stream into node-local temp, tile, evict the SVS
            svs = _download_svs(fid, row.get(filename_col), stage_dir, token, session)
            try:
                n = tile_slide(svs, patches_dir, sid, patch_size=patch_size, level=level,
                               tissue_thresh=tissue_thresh, thumb_max_dim=thumb_max_dim,
                               jpeg_quality=jpeg_quality)
            finally:
                _evict(svs.parent)
            return sid, "downloaded", n
        except Exception as e:
            logger.error("tile failed for %s (%s): %s", sid, fid, e)
            return sid, "failed", 0

    rows = [r for _, r in df.iterrows()]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in as_completed([ex.submit(work, r) for r in rows]):
            sid, status, n = fut.result()
            if status in ("downloaded", "reused"):
                counts["slides"] += 1
                counts["patches"] += n
                if status == "reused":
                    counts["reused"] += 1
            elif status == "skipped":
                counts["skipped"] += 1
                counts["patches"] += n
            else:
                counts["failed"] += 1

    logger.info("tile_process: %d slides tiled (%d patches total, %d reused-cache), "
                "%d already-tiled, %d failed", counts["slides"], counts["patches"],
                counts["reused"], counts["skipped"], counts["failed"])
    return counts
