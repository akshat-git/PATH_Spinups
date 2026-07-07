"""Stream TCGA slide thumbnails from GDC WITHOUT downloading whole SVS files.

The pathology models only consume small (~512 px) thumbnails, yet each SVS is
~250 MB-1.5 GB. Downloading the full slide just to render a thumbnail is the
scalability bottleneck (~500 GB / many hours for a full corpus).

GDC's ``/data/<uuid>`` endpoint supports HTTP Range (verified: ``Accept-Ranges:
bytes``, ``206 Partial Content``), and Aperio SVS files embed a small thumbnail
image. So we open the remote file over Range requests and let ``tifffile`` read
ONLY the bytes it needs (the IFD headers + the thumbnail strips) -- a few MB
instead of the whole slide. Nothing is written to disk except the final JPG.

If a slide's thumbnail can't be read this way, ``stream_thumbnails`` falls back
to a bounded full download -> openslide thumbnail -> delete for just that slide,
so disk never accumulates more than ``n_workers`` slides at once.
"""

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


# ---------------------------------------------------------------------------
# HTTP Range-backed, seekable, read-only file object (for tifffile)
# ---------------------------------------------------------------------------
class HTTPRangeFile(io.RawIOBase):
    """A seekable read-only file-like object served by HTTP Range requests.

    Fetches fixed-size blocks on demand and caches them, so tifffile's many
    small scattered reads (IFD walking) cost only a handful of range requests.
    """

    def __init__(self, url, session=None, headers=None, block_size=1 << 20, timeout=60):
        self._url = url
        self._session = session or requests.Session()
        self._headers = dict(headers or {})
        self._block = int(block_size)
        self._timeout = timeout
        self._pos = 0
        self._cache = {}          # block_index -> bytes
        self._size = self._fetch_size()

    def _fetch_size(self):
        r = self._session.get(self._url, headers={**self._headers, "Range": "bytes=0-0"},
                              timeout=self._timeout, stream=True)
        r.raise_for_status()
        cr = r.headers.get("Content-Range", "")
        r.close()
        if "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        raise IOError(f"server did not report a size for {self._url} (no Content-Range)")

    def _get_block(self, idx):
        cached = self._cache.get(idx)
        if cached is not None:
            return cached
        start = idx * self._block
        if start >= self._size:
            return b""
        end = min(start + self._block, self._size) - 1
        r = self._session.get(self._url, headers={**self._headers, "Range": f"bytes={start}-{end}"},
                              timeout=self._timeout)
        r.raise_for_status()
        data = r.content
        self._cache[idx] = data
        return data

    # --- io.RawIOBase interface ---
    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        return self._pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._size - self._pos
        size = max(0, min(size, self._size - self._pos))
        out = bytearray()
        while size > 0:
            idx, off = divmod(self._pos, self._block)
            block = self._get_block(idx)
            if not block:
                break
            chunk = block[off:off + size]
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            size -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


# ---------------------------------------------------------------------------
# Thumbnail extraction
# ---------------------------------------------------------------------------
def _pick_thumbnail_array(tif):
    """Return a small RGB ndarray: the Aperio 'Thumbnail' series if present,
    else the smallest pyramid level of the baseline series."""
    for s in getattr(tif, "series", []) or []:
        if (getattr(s, "name", "") or "").lower() == "thumbnail":
            return s.asarray()
    series = tif.series[0] if getattr(tif, "series", None) else None
    if series is None:
        return None
    levels = getattr(series, "levels", None)
    if levels:                       # smallest (last) pyramid level
        return levels[-1].asarray()
    return series.asarray()


def stream_thumbnail(file_id, out_path, size=(512, 512), token=None, session=None):
    """Read a slide's embedded thumbnail via HTTP Range and save a JPG.

    Returns True on success, False if the thumbnail could not be read (caller
    may then fall back to a full download).
    """
    import numpy as np  # noqa: F401  (imported for PIL fromarray dtypes)
    import tifffile
    from PIL import Image

    headers = {"X-Auth-Token": token} if token else {}
    url = f"{GDC_DATA_URL}/{file_id}"
    try:
        fh = HTTPRangeFile(url, session=session, headers=headers)
        with tifffile.TiffFile(fh) as tif:
            arr = _pick_thumbnail_array(tif)
        if arr is None:
            return False
        img = Image.fromarray(arr).convert("RGB")
        img.thumbnail(size)          # preserve aspect, fit within size (matches openslide)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "JPEG", quality=85)
        return True
    except Exception as e:  # network / format / partial-read issues -> let caller fall back
        logger.warning("range-stream thumbnail failed for %s: %s", file_id, e)
        return False


def _fallback_full_thumbnail(file_id, filename, out_path, size, tmp_dir, token=None, session=None):
    """Full download -> openslide thumbnail -> delete the SVS (bounded disk)."""
    import openslide

    sess = session or requests.Session()
    headers = {"X-Auth-Token": token} if token else {}
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_svs = tmp_dir / f"{file_id}_{filename or 'slide.svs'}"
    try:
        with sess.get(f"{GDC_DATA_URL}/{file_id}", headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp_svs, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        slide = openslide.OpenSlide(str(tmp_svs))
        thumb = slide.get_thumbnail(size)
        slide.close()
        if thumb.mode != "RGB":
            thumb = thumb.convert("RGB")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(out_path, "JPEG", quality=85)
        return True
    except Exception as e:
        logger.error("fallback full-download thumbnail failed for %s: %s", file_id, e)
        return False
    finally:
        try:
            if tmp_svs.exists():
                tmp_svs.unlink()
        except OSError:
            pass


@dataclass
class StreamResult:
    df: object            # DataFrame with a 'jpg_path' column added
    streamed: int = 0     # produced via range-read (no full download)
    fell_back: int = 0    # produced via full download fallback
    skipped: int = 0      # thumbnail already existed
    failed: int = 0       # could not produce a thumbnail


def stream_thumbnails(df, thumbnails_dir, size=(512, 512), n_workers=4,
                      fallback=True, tmp_dir=None, token=None,
                      file_id_col="file_id", filename_col="filename",
                      slide_id_col="slide_id"):
    """Produce ``<slide_id>.jpg`` thumbnails for every row of ``df`` concurrently.

    Per slide: try a range-read of the embedded thumbnail (no full download); on
    failure, if ``fallback`` is set, do a bounded full download -> openslide ->
    delete. Resumable: rows whose thumbnail already exists are skipped. Adds a
    ``jpg_path`` column and returns a StreamResult (mirrors SlideProcessor).
    """
    import pandas as pd  # noqa: F401

    thumbnails_dir = Path(thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tmp_dir) if tmp_dir else (thumbnails_dir.parent / "_stream_tmp")

    rows = list(df.iterrows())
    session = requests.Session()
    jpg_by_id = {}
    streamed = fell_back = skipped = failed = 0

    def work(row):
        slide_id = row[slide_id_col]
        file_id = row[file_id_col]
        out = thumbnails_dir / f"{slide_id}.jpg"
        if out.exists():
            return slide_id, out, "skipped"
        if file_id and stream_thumbnail(file_id, out, size, token, session):
            return slide_id, out, "streamed"
        if fallback and file_id and _fallback_full_thumbnail(
                file_id, row.get(filename_col), out, size, tmp_dir, token, session):
            return slide_id, out, "fell_back"
        return slide_id, None, "failed"

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        futs = [ex.submit(work, row) for _, row in rows]
        for fut in as_completed(futs):
            slide_id, out, status = fut.result()
            jpg_by_id[slide_id] = out
            if status == "streamed":
                streamed += 1
            elif status == "fell_back":
                fell_back += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

    logger.info("stream_thumbnails: %d streamed, %d fell back to full download, "
                "%d already existed, %d failed", streamed, fell_back, skipped, failed)

    df = df.copy()
    df["jpg_path"] = df[slide_id_col].map(lambda sid: jpg_by_id.get(sid))
    return StreamResult(df=df, streamed=streamed, fell_back=fell_back,
                        skipped=skipped, failed=failed)
