"""Pack per-slide loose patch JPGs into ONE .tar per slide.

Why: tiling persists ~16k tiny JPGs/slide (~2.26M files total). Every per-run STEP 4
`find`+`tar` over that many files is a Lustre metadata storm whose duration is wildly
variable (it made the GPU job time out in STEP 4). Packing each slide's patches into a
single `patches_tar/<slide_id>.tar` turns 2.26M files into ~142, so staging becomes a
handful of big sequential reads and extraction streams tar members (see pfm_common/data).

Resumable + atomic: a slide whose `<slide_id>.tar` + `<slide_id>.done` sentinel exist is
skipped; each tar is written to `.part` and renamed on completion, so an interrupted pack
never leaves a half tar that looks finished. Pure stdlib (tarfile) -- runs in tcga_build.
"""
import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)


def pack_all(patches_dir, tars_dir):
    """Pack every ``patches_dir/<slide_id>/*.jpg`` into ``tars_dir/<slide_id>.tar``
    (members stored flat, keeping the ``<slide_id>__x_y.jpg`` names so the benchmark's
    slide_id join still works). Returns a counts dict."""
    patches_dir = Path(patches_dir)
    tars_dir = Path(tars_dir)
    tars_dir.mkdir(parents=True, exist_ok=True)

    slides = sorted((d for d in patches_dir.iterdir() if d.is_dir())) if patches_dir.exists() else []
    packed = skipped = total_patches = 0
    for sd in slides:
        out = tars_dir / f"{sd.name}.tar"
        done = tars_dir / f"{sd.name}.done"
        if out.exists() and done.exists():
            skipped += 1
            continue
        jpgs = sorted(sd.glob("*.jpg"))
        if not jpgs:
            continue
        tmp = out.with_suffix(".tar.part")
        with tarfile.open(tmp, "w") as tf:      # no compression: JPGs are already compressed
            for j in jpgs:
                tf.add(str(j), arcname=j.name)
        tmp.rename(out)
        done.write_text(str(len(jpgs)))
        packed += 1
        total_patches += len(jpgs)
        logger.info("packed %s -> %s (%d patches)", sd.name, out.name, len(jpgs))

    logger.info("pack_patches: %d slides packed, %d already packed, %d patches total",
                packed, skipped, total_patches)
    return {"packed": packed, "skipped": skipped, "patches": total_patches, "slides": len(slides)}
