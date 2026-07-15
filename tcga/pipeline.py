"""
TCGA Dataset Builder — end-to-end pipeline orchestrator.

Chains the existing TCGA modules into a single YAML-driven pipeline:
query GDC → build slide table → generate manifests → download →
process slides → build gene matrix → assemble final dataset.

Usage:
    from tcga import TCGADatasetBuilder

    builder = TCGADatasetBuilder(cfg)
    builder.run()
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig

from tcga.config import TCGAConfig
from tcga.downloader import DownloadStatus, TCGADownloader
from tcga.etl import TCGASlideETL
from tcga.gene_matrix import GeneMatrix
from tcga.manifest import ManifestGenerator
from tcga.slide_processor import SlideProcessor

logger = logging.getLogger(__name__)

VALID_STEPS = ["etl", "manifest", "download", "download_svs_cache", "stream_thumbnails",
               "stage_process", "tile_slides", "pack_patches", "decode_patches",
               "process_slides", "gene_matrix", "assemble"]


class TCGADatasetBuilder:
    """Orchestrates the full TCGA dataset build pipeline.

    Each step is a method that reads / writes intermediate artifacts under
    ``data_dir`` so the pipeline is resumable.  Pass ``force=True`` to
    re-run steps even when their artifacts already exist.

    Args:
        cfg: A resolved OmegaConf ``DictConfig`` (or plain dict) matching
            the schema of ``configs/tcga_dataset.yaml``.
        force: If True, ignore existing artifacts and re-run every step.
    """

    def __init__(self, cfg: DictConfig, force: bool = False):
        self.cfg = cfg
        self.force = force

        # Build TCGAConfig from the YAML fields
        self.tcga_config = TCGAConfig(
            project_ids=list(cfg.projects),
            data_dir=Path(cfg.data_dir),
            include_demographics=cfg.etl.include_demographics,
            include_diagnosis=cfg.etl.include_diagnosis,
            include_maf=cfg.etl.include_maf,
            access=cfg.access,
        )
        self.tcga_config.ensure_directories()

        # Convenience aliases
        self._tables_dir = self.tcga_config.tables_dir
        self._manifests_dir = self.tcga_config.manifests_dir

    # ── public entry point ────────────────────────────────────────────

    def run(self, steps: Optional[List[str]] = None) -> Path:
        """Run the pipeline.

        Args:
            steps: Override which steps to execute.  Defaults to
                ``cfg.steps``.

        Returns:
            Path to the final dataset CSV.
        """
        steps = steps or list(self.cfg.steps)
        self._validate_steps(steps)

        logger.info("Pipeline steps: %s", steps)
        logger.info("Projects: %s", list(self.cfg.projects))
        logger.info("Data dir: %s", self.tcga_config.data_dir)

        dispatch = {
            "etl": self.run_etl,
            "manifest": self.run_manifest,
            "download": self.run_download,
            "download_svs_cache": self.run_download_svs_cache,
            "stream_thumbnails": self.run_stream_thumbnails,
            "stage_process": self.run_stage_process,
            "tile_slides": self.run_tile_slides,
            "pack_patches": self.run_pack_patches,
            "decode_patches": self.run_decode_patches,
            "process_slides": self.run_process_slides,
            "gene_matrix": self.run_gene_matrix,
            "assemble": self.run_assemble,
        }

        for step in steps:
            logger.info("=" * 60)
            logger.info("STEP: %s", step)
            logger.info("=" * 60)
            dispatch[step]()

        dataset_path = self._tables_dir / "dataset.csv"
        self._print_summary(dataset_path)
        return dataset_path

    # ── step 1: ETL ───────────────────────────────────────────────────

    def run_etl(self) -> pd.DataFrame:
        """Query GDC API and build flat slide table."""
        output = self._tables_dir / "slide_table.parquet"

        if output.exists() and not self.force:
            logger.info("Slide table already exists at %s — skipping ETL", output)
            return pd.read_parquet(output)

        etl = TCGASlideETL()
        df = etl.build_slide_table(
            project_ids=self.tcga_config.project_ids,
            include_demographics=self.tcga_config.include_demographics,
            include_diagnosis=self.tcga_config.include_diagnosis,
            include_maf=self.tcga_config.include_maf,
            access=self.tcga_config.access,
        )
        df = etl.add_local_paths(df, self.tcga_config)
        logger.info("Slide table: %d rows, %d columns", len(df), len(df.columns))

        self._stringify_paths(df).to_parquet(output, index=False)
        self._stringify_paths(df).to_csv(output.with_suffix(".csv"), index=False)
        logger.info("Saved slide table → %s", output)
        return df

    # ── step 2: manifests ─────────────────────────────────────────────

    def run_manifest(self) -> Dict[str, Optional[Path]]:
        """Generate download manifests from the slide table."""
        slide_manifest = self._manifests_dir / "slides_manifest.txt"
        maf_manifest = self._manifests_dir / "maf_manifest.txt"

        if slide_manifest.exists() and not self.force:
            logger.info("Manifests already exist — skipping manifest generation")
            return {"slides": slide_manifest, "maf": maf_manifest if maf_manifest.exists() else None}

        df = self._load_slide_table()
        gen = ManifestGenerator()

        # Slide manifest
        gen.create_slide_manifest(df, slide_manifest)
        logger.info("Slide manifest → %s", slide_manifest)

        # MAF manifest (only if MAF data present)
        maf_path: Optional[Path] = None
        if self.tcga_config.include_maf:
            maf_path = gen.create_maf_manifest(df, maf_manifest)
            if maf_path:
                logger.info("MAF manifest   → %s", maf_path)
            else:
                logger.info("No MAF files found — skipping MAF manifest")

        # Subset manifests for testing -- stratified across projects so a small
        # sample spans LUAD/LUSC/LGG/GBM (not just whichever project sorts first),
        # and the MAF subset is built from exactly those slides so gene labels line up.
        max_files = self.cfg.download.get("max_files")
        target_gb = self.cfg.download.get("target_gb")
        subset_df = None
        if max_files:
            subset_df = gen.select_stratified(df, max_files, group_col="project_id")
        elif target_gb:
            subset_df = gen.select_by_byte_budget(df, target_gb, group_col="project_id")
        if subset_df is not None:
            if "project_id" in subset_df.columns:
                counts = subset_df["project_id"].value_counts().to_dict()
                logger.info("Stratified subset: %d slides across projects %s",
                            len(subset_df), counts)

            subset_slide = self._manifests_dir / "slides_manifest_subset.txt"
            gen.create_slide_manifest(subset_df, subset_slide)
            logger.info("Subset slide manifest (%d files) → %s", len(subset_df), subset_slide)

            if maf_path:
                subset_maf = self._manifests_dir / "maf_manifest_subset.txt"
                if gen.create_maf_manifest(subset_df, subset_maf):
                    logger.info("Subset MAF manifest (for the %d subset slides) → %s",
                                len(subset_df), subset_maf)
                else:
                    logger.info("No MAF files among subset slides — skipping subset MAF manifest")

        return {"slides": slide_manifest, "maf": maf_path}

    # ── step 3: download ──────────────────────────────────────────────

    def run_download(self) -> None:
        """Download slides and MAF files via gdc-client."""
        dl_cfg = self.cfg.download
        if not dl_cfg.get("enabled", True):
            logger.info("Downloads disabled in config — skipping")
            return

        downloader = TCGADownloader()
        max_files = dl_cfg.get("max_files")
        token_path = dl_cfg.get("token_path")
        if token_path:
            token_path = Path(token_path)
        n_processes = dl_cfg.get("n_processes", 4)

        # --- Slides ---
        if dl_cfg.get("slides", True):
            manifest = self._resolve_manifest("slides_manifest", max_files)
            status = downloader.check_download_status(self.tcga_config.slides_dir, manifest)

            if status.status == DownloadStatus.COMPLETED and not self.force:
                logger.info("Slides already downloaded (%d/%d) — skipping",
                            status.files_downloaded, status.files_total)
            else:
                logger.info("Downloading slides from %s …", manifest)
                result = downloader.download_from_manifest(
                    manifest, self.tcga_config.slides_dir,
                    token_path=token_path, n_processes=n_processes,
                )
                logger.info("Slide download %s: %d/%d files",
                            result.status.value, result.files_downloaded, result.files_total)
                if result.error_message:
                    logger.error("Download error: %s", result.error_message)

        # --- MAF ---
        if dl_cfg.get("maf", True) and self.tcga_config.include_maf:
            maf_manifest = self._resolve_manifest("maf_manifest", max_files)
            if maf_manifest and maf_manifest.exists():
                status = downloader.check_download_status(self.tcga_config.maf_dir, maf_manifest)

                if status.status == DownloadStatus.COMPLETED and not self.force:
                    logger.info("MAF files already downloaded (%d/%d) — skipping",
                                status.files_downloaded, status.files_total)
                else:
                    logger.info("Downloading MAF files from %s …", maf_manifest)
                    result = downloader.download_from_manifest(
                        maf_manifest, self.tcga_config.maf_dir,
                        token_path=token_path, n_processes=n_processes,
                    )
                    logger.info("MAF download %s: %d/%d files",
                                result.status.value, result.files_downloaded, result.files_total)
                    if result.error_message:
                        logger.error("Download error: %s", result.error_message)
            else:
                logger.info("No MAF manifest found — skipping MAF download")

    # ── step 3b: stream thumbnails (async, no full SVS download) ──────

    def run_stream_thumbnails(self) -> pd.DataFrame:
        """Produce thumbnails by range-streaming each slide's embedded thumbnail
        from GDC -- no full SVS download -- falling back to a bounded full
        download only for slides whose thumbnail can't be range-read. Replaces
        the download(slides) + process_slides pair for the thumbnail workflow;
        transfers ~MBs/slide instead of ~GBs and keeps no SVS on disk.
        """
        from tcga.slide_streamer import stream_thumbnails

        full_df = self._load_slide_table()

        # Honour max_files with the SAME stratified subset the manifests use, so
        # streamed slides and the subset MAF (gene labels) line up.
        max_files = self.cfg.download.get("max_files")
        subset_df = full_df
        if max_files:
            subset_df = ManifestGenerator().select_stratified(full_df, max_files, group_col="project_id")
            if "project_id" in subset_df.columns:
                logger.info("Streaming thumbnails for %d stratified slides across %s",
                            len(subset_df), subset_df["project_id"].value_counts().to_dict())

        slides_cfg = self.cfg.slides
        size = tuple(slides_cfg.get("thumbnail_size", [512, 512]))
        n_workers = slides_cfg.get("n_workers", 4)
        fallback = slides_cfg.get("stream_fallback", True)

        token_path = self.cfg.download.get("token_path")
        token = Path(token_path).read_text().strip() if token_path else None

        # Bounded temp dir for the (rare) full-download fallback: node-local SSD
        # if available, else under data_dir. Only up to n_workers SVS exist at once.
        l_scratch = os.environ.get("L_SCRATCH")
        tmp_dir = (Path(l_scratch) / "stream_svs_tmp") if l_scratch \
            else (self.tcga_config.data_dir / "_stream_tmp")

        result = stream_thumbnails(
            subset_df, self.tcga_config.thumbnails_dir, size=size, n_workers=n_workers,
            fallback=fallback, tmp_dir=tmp_dir, token=token,
        )
        logger.info("Thumbnails: %d range-streamed, %d full-download fallback, %d existing, %d failed",
                    result.streamed, result.fell_back, result.skipped, result.failed)

        # Map jpg_path back onto the FULL table (non-subset rows stay None), then
        # persist -- keeps the full table intact like process_slides does.
        jpg_map = dict(zip(result.df["slide_id"], result.df["jpg_path"]))
        full_df = full_df.copy()
        full_df["jpg_path"] = full_df["slide_id"].map(lambda sid: jpg_map.get(sid))

        out = self._tables_dir / "slide_table.parquet"
        self._stringify_paths(full_df).to_parquet(out, index=False)
        self._stringify_paths(full_df).to_csv(out.with_suffix(".csv"), index=False)
        logger.info("Updated slide table with jpg_path → %s", out)
        return full_df

    # ── shared: pick the stratified staged subset (count or byte budget) ─────

    def _select_staged_subset(self):
        """Select the stratified staged subset from the full slide table, per
        download.max_files (count) or download.target_gb (byte budget). Shared by
        stage_process and download_svs_cache so the pre-downloaded SVS cache and the
        thumbnailer agree on exactly which slides are in scope. Returns (full, subset).
        """
        full_df = self._load_slide_table()
        gen = ManifestGenerator()

        max_files = self.cfg.download.get("max_files")
        target_gb = self.cfg.download.get("target_gb")
        if max_files:
            subset_df = gen.select_stratified(full_df, max_files, group_col="project_id")
        elif target_gb:
            subset_df = gen.select_by_byte_budget(full_df, target_gb, group_col="project_id")
        else:
            subset_df = full_df

        gb = (float(subset_df["file_size"].fillna(0).sum()) / (1024 ** 3)
              if "file_size" in subset_df.columns else float("nan"))
        spread = (subset_df["project_id"].value_counts().to_dict()
                  if "project_id" in subset_df.columns else {})
        logger.info("Staged subset: %d slides (~%.1f GB) across %s", len(subset_df), gb, spread)
        return full_df, subset_df

    # ── step 3b: pre-download the full SVS subset to the persistent cache ────

    def run_download_svs_cache(self) -> None:
        """Pre-download the FULL SVS of the staged subset into the persistent
        $SCRATCH cache (slides_dir), so a later staged run thumbnails from disk
        instead of streaming from GDC. This is the standalone "download all of TCGA"
        entry point (the download_svs_cache step); the staged pipeline's hybrid then reuses
        whatever is cached and streams only the rest.
        """
        from tcga.slide_stager import predownload_svs

        _, subset_df = self._select_staged_subset()
        slides_cfg = self.cfg.slides
        workers = slides_cfg.get("stage_download_workers", slides_cfg.get("n_workers", 6))
        token_path = self.cfg.download.get("token_path")
        token = Path(token_path).read_text().strip() if token_path else None

        counts = predownload_svs(subset_df, self.tcga_config.slides_dir,
                                 workers=workers, token=token)
        logger.info("SVS cache ready in %s: %d downloaded, %d already-cached, %d failed",
                    self.tcga_config.slides_dir, counts["downloaded"], counts["reused"],
                    counts["failed"])

    # ── step 3c: staged full-SVS loading (scratch cache + node-local stage) ──

    def run_stage_process(self) -> pd.DataFrame:
        """Full-SVS staged loading. Cache a stratified ~``download.target_gb`` (or
        ``max_files``) subset of FULL slides on $SCRATCH, and thumbnail each by
        copying it to a node-local ($L_SCRATCH) staging area, running openslide
        there, then evicting -- concurrently (downloads overlap thumbnailing) and
        resumably (skip thumbnailed slides; don't re-download cached SVS).
        """
        from tcga.slide_stager import acquire_stage_process

        full_df, subset_df = self._select_staged_subset()

        slides_cfg = self.cfg.slides
        size = tuple(slides_cfg.get("thumbnail_size", [512, 512]))
        workers = slides_cfg.get("stage_download_workers", slides_cfg.get("n_workers", 6))

        token_path = self.cfg.download.get("token_path")
        token = Path(token_path).read_text().strip() if token_path else None

        # node-local SSD staging area (pieces copied/streamed here, thumbnailed, evicted)
        local = os.environ.get("L_SCRATCH") or str(self.tcga_config.data_dir / "_stage")
        stage_dir = Path(local) / "tcga_stage"

        # stream_to_local: download each SVS straight into node-local temp and evict
        # it after thumbnailing (no persistent ~50 GB scratch cache). Only the tiny
        # thumbnails persist, so re-runs stay fast without relying on downloaded SVS.
        stream_to_local = bool(self.cfg.download.get("stream_to_local", False))
        logger.info("stage_process mode: %s",
                    "stream-to-local (no persistent SVS cache)" if stream_to_local
                    else "two-tier scratch cache")

        result = acquire_stage_process(
            subset_df, self.tcga_config.slides_dir, stage_dir,
            self.tcga_config.thumbnails_dir, size=size, workers=workers, token=token,
            stream_to_local=stream_to_local,
        )
        logger.info("Staged thumbnails: %d processed (%d downloaded, %d reused-cache), "
                    "%d existing, %d failed", result.processed, result.downloaded,
                    result.reused, result.skipped, result.failed)

        # map jpg_path onto the full table (non-subset rows stay None), persist
        jpg_map = dict(zip(result.df["slide_id"], result.df["jpg_path"]))
        full_df = full_df.copy()
        full_df["jpg_path"] = full_df["slide_id"].map(lambda sid: jpg_map.get(sid))
        out = self._tables_dir / "slide_table.parquet"
        self._stringify_paths(full_df).to_parquet(out, index=False)
        self._stringify_paths(full_df).to_csv(out.with_suffix(".csv"), index=False)
        logger.info("Updated slide table with jpg_path → %s", out)
        return full_df

    # ── step 3d: patch-level tiling (makes extraction GPU-bound) ─────────────

    def run_tile_slides(self) -> pd.DataFrame:
        """Tile the staged subset into tissue patches under a node-local patches dir
        (the furnace). Streams/reuses each SVS via the same hybrid as stage_process,
        cuts many patches per slide, evicts the SVS. Extraction then runs GPU-bound
        over the patches; benchmark.py mean-pools patches back to slide level.
        """
        from tcga.slide_stager import acquire_tile_process

        full_df, subset_df = self._select_staged_subset()

        slides_cfg = self.cfg.slides
        patch_cfg = self.cfg.get("patches", {}) or {}
        workers = slides_cfg.get("stage_download_workers", slides_cfg.get("n_workers", 6))
        stream_to_local = bool(self.cfg.download.get("stream_to_local", True))

        token_path = self.cfg.download.get("token_path")
        token = Path(token_path).read_text().strip() if token_path else None

        local = os.environ.get("L_SCRATCH") or str(self.tcga_config.data_dir / "_stage")
        stage_dir = Path(local) / "tcga_stage"                                   # node-local: transient SVS
        # Patches PERSIST on scratch as one tar per slide (patches_tar/<sid>.tar), tiled once
        # and reused; a later run skips tiling entirely. Writing tars directly (not loose
        # files) avoids millions of tiny files -- the metadata storm that made staging slow.
        patches_dir = Path(patch_cfg.get("patches_dir") or (self.tcga_config.data_dir / "patches_tar"))

        counts = acquire_tile_process(
            subset_df, self.tcga_config.slides_dir, stage_dir, patches_dir,
            patch_size=patch_cfg.get("patch_size", 256),
            level=patch_cfg.get("level", 0),
            tissue_thresh=patch_cfg.get("tissue_thresh", 0.10),
            thumb_max_dim=patch_cfg.get("thumb_max_dim", 2048),
            jpeg_quality=patch_cfg.get("jpeg_quality", 85),
            workers=workers, token=token, stream_to_local=stream_to_local,
        )
        logger.info("Tiled %d slides -> %d patches; extraction reads PFM_PATCH_DIR=%s",
                    counts["slides"] + counts["skipped"], counts["patches"], patches_dir)

        full_df = full_df.copy()
        full_df["patches_dir"] = str(patches_dir)
        return full_df

    # ── step 3e: pack per-slide patches into one .tar each (fast staging) ────
    def run_pack_patches(self) -> None:
        """Pack loose per-slide patch JPGs into one tar/slide (patches_tar/<slide_id>.tar)
        so a run stages ~N_slides big files instead of millions of tiny Lustre files.
        Idempotent/resumable; the loose patches are left in place as the source of truth."""
        from tcga.pack_patches import pack_all
        patches_dir = self.tcga_config.data_dir / "patches"
        tars_dir = self.tcga_config.data_dir / "patches_tar"
        counts = pack_all(patches_dir, tars_dir)
        logger.info("Packed patches -> %s: %d slides packed, %d already, %d patches",
                    tars_dir, counts["packed"], counts["skipped"], counts["patches"])

    # ── step 3f: pre-decode patch tars into raw uint8 bins (CPU, byte-level parallel) ──
    def run_decode_patches(self) -> None:
        """Pre-decode each patches_tar/<slide>.tar into patches_raw/<slide>.bin (raw uint8),
        so the GPU run does ZERO JPEG decode. Byte-level parallel over a GIL-free thread pool
        (cv2/libjpeg-turbo); resumable + atomic per slide. Reads decode.workers from config
        (or PFM_DECODE_WORKERS / all cores)."""
        from tcga.decode_patches import decode_all
        tars_dir = self.tcga_config.data_dir / "patches_tar"
        raw_dir = self.tcga_config.data_dir / "patches_raw"
        dec_cfg = self.cfg.get("decode", {}) or {}
        workers = int(os.environ.get("PFM_DECODE_WORKERS")
                      or dec_cfg.get("workers", 0) or (os.cpu_count() or 8))
        counts = decode_all(str(tars_dir), str(raw_dir), workers=workers)
        logger.info("Decoded patches -> %s: %d decoded, %d already, %d patches, %d failed",
                    raw_dir, counts["decoded"], counts["skipped"], counts["patches"], counts["failed"])

    # ── step 4: process slides ────────────────────────────────────────

    def run_process_slides(self) -> pd.DataFrame:
        """Create JPG thumbnails from SVS whole-slide images."""
        df = self._load_slide_table()
        slides_cfg = self.cfg.slides
        size = tuple(slides_cfg.get("thumbnail_size", [512, 512]))
        n_workers = slides_cfg.get("n_workers", 4)

        processor = SlideProcessor(n_workers=n_workers)
        result = processor.process_slides(
            df=df,
            output_dir=self.tcga_config.thumbnails_dir,
            size=size,
        )

        logger.info("Slide processing: %d processed, %d skipped, %d failed, %d missing",
                     result.processed, result.skipped, result.failed, result.missing)

        # Persist the updated table with jpg_path
        output = self._tables_dir / "slide_table.parquet"
        self._stringify_paths(result.df).to_parquet(output, index=False)
        self._stringify_paths(result.df).to_csv(output.with_suffix(".csv"), index=False)
        logger.info("Updated slide table with jpg_path → %s", output)

        return result.df

    # ── step 5: gene matrix ───────────────────────────────────────────

    def run_gene_matrix(self) -> GeneMatrix:
        """Build gene mutation matrix from downloaded MAF files."""
        gm_cfg = self.cfg.gene_matrix
        if not gm_cfg.get("enabled", True):
            logger.info("Gene matrix disabled in config — skipping")
            return GeneMatrix()

        output = self._tables_dir / "gene_matrix.parquet"
        if output.exists() and not self.force:
            logger.info("Gene matrix already exists at %s — loading", output)
            return GeneMatrix.load(output)

        gm = GeneMatrix()
        gm.build_from_maf_dir(self.tcga_config.maf_dir)
        gm.save(output)
        logger.info("Gene matrix %s → %s", gm.shape, output)
        return gm

    # ── step 6: assemble ──────────────────────────────────────────────

    def run_assemble(self) -> pd.DataFrame:
        """Merge slide table + gene matrix → final dataset."""
        df = self._load_slide_table()

        # Validate local paths (adds slide_exists / maf_exists)
        etl = TCGASlideETL()
        df = etl.validate_local_paths(df)

        # Merge gene matrix if it exists
        gm_path = self._tables_dir / "gene_matrix.parquet"
        if gm_path.exists():
            gm = GeneMatrix.load(gm_path)
            genes = self.cfg.gene_matrix.get("genes")
            genes = list(genes) if genes else None
            df = gm.merge(df, genes=genes)
            logger.info("Merged gene matrix (%d genes) into slide table", len(gm.genes) if genes is None else len(genes))

        # Save final dataset
        csv_out = self._tables_dir / "dataset.csv"
        parquet_out = self._tables_dir / "dataset.parquet"
        df_out = self._stringify_paths(df)
        df_out.to_csv(csv_out, index=False)
        df_out.to_parquet(parquet_out, index=False)
        logger.info("Final dataset: %d rows × %d columns", len(df), len(df.columns))
        logger.info("Saved → %s", csv_out)
        logger.info("Saved → %s", parquet_out)
        return df

    # ── helpers ───────────────────────────────────────────────────────

    def _load_slide_table(self) -> pd.DataFrame:
        """Load the slide table from parquet (must have been built by run_etl)."""
        path = self._tables_dir / "slide_table.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Slide table not found at {path}. Run the 'etl' step first."
            )
        return pd.read_parquet(path)

    def _resolve_manifest(self, name: str, max_files: Optional[int] = None) -> Path:
        """Prefer the subset manifest whenever one exists (run_manifest builds it
        for max_files OR target_gb), else fall back to the full manifest. Keying
        off the file's existence — not just max_files — keeps the MAF download
        aligned with a byte-budget (target_gb) slide subset."""
        subset = self._manifests_dir / f"{name}_subset.txt"
        if subset.exists():
            return subset
        return self._manifests_dir / f"{name}.txt"

    def _print_summary(self, dataset_path: Path) -> None:
        """Print a human-readable summary of the pipeline run."""
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        slide_table = self._tables_dir / "slide_table.parquet"
        if slide_table.exists():
            df = pd.read_parquet(slide_table)
            logger.info("Slide table : %d rows", len(df))

        gm_path = self._tables_dir / "gene_matrix.parquet"
        if gm_path.exists():
            gm = GeneMatrix.load(gm_path)
            logger.info("Gene matrix : %d samples × %d genes", *gm.shape)

        if dataset_path.exists():
            ds = pd.read_csv(dataset_path, nrows=0)
            n_rows = sum(1 for _ in open(dataset_path)) - 1
            logger.info("Dataset     : %d rows × %d columns", n_rows, len(ds.columns))
            logger.info("Output      : %s", dataset_path)

    @staticmethod
    def _validate_steps(steps: List[str]) -> None:
        for s in steps:
            if s not in VALID_STEPS:
                raise ValueError(
                    f"Unknown step '{s}'. Valid steps: {VALID_STEPS}"
                )

    @staticmethod
    def _stringify_paths(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Path objects to strings so parquet/csv serialisation works."""
        df = df.copy()
        path_cols = [c for c in df.columns if c.endswith("_path")]
        for col in path_cols:
            df[col] = df[col].apply(lambda v: str(v) if v is not None else None)
        return df
