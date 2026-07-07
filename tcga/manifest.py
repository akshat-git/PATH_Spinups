"""Manifest generation for GDC downloads."""

from pathlib import Path
from typing import Optional
import pandas as pd


class ManifestGenerator:
    """Generates manifest files for gdc-client downloads.

    Usage:
        manifest_gen = ManifestGenerator()

        # Create slide manifest
        slide_manifest = manifest_gen.create_slide_manifest(
            df, Path("manifests/slides_manifest.txt")
        )

        # Create MAF manifest (deduplicated)
        maf_manifest = manifest_gen.create_maf_manifest(
            df, Path("manifests/maf_manifest.txt")
        )
    """

    HEADER = "id\tfilename\tmd5\tsize\tstate"

    def create_slide_manifest(
        self,
        df: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Create manifest for slide image files.

        Args:
            df: DataFrame from TCGASlideETL.build_slide_table()
            output_path: Where to save manifest

        Returns:
            Path to created manifest
        """
        lines = [self.HEADER]
        for _, row in df.iterrows():
            lines.append(self._format_line(
                file_id=row["file_id"],
                filename=row["filename"],
                md5sum=row.get("md5sum", ""),
                file_size=row["file_size"],
                state=row.get("file_state", ""),
            ))

        self._write_manifest(lines, output_path)
        return output_path

    def create_maf_manifest(
        self,
        df: pd.DataFrame,
        output_path: Path,
    ) -> Optional[Path]:
        """Create manifest for MAF files (deduplicated).

        Many slides share the same MAF file (all slides from a sample).
        This creates a manifest with unique MAF files only.

        Args:
            df: DataFrame from TCGASlideETL.build_slide_table() with include_maf=True
            output_path: Where to save manifest

        Returns:
            Path to created manifest, or None if no MAF files
        """
        # Filter to rows with MAF and deduplicate
        if "has_maf" not in df.columns:
            return None

        maf_df = df[df["has_maf"] == True].copy()
        if maf_df.empty:
            return None

        maf_df = maf_df[["maf_file_id", "maf_filename", "maf_file_size", "maf_md5sum"]].drop_duplicates()

        lines = [self.HEADER]
        for _, row in maf_df.iterrows():
            lines.append(self._format_line(
                file_id=row["maf_file_id"],
                filename=row["maf_filename"],
                md5sum=row.get("maf_md5sum", ""),
                file_size=row["maf_file_size"],
                state="",
            ))

        self._write_manifest(lines, output_path)
        return output_path

    def _format_line(self, file_id, filename, md5sum, file_size, state) -> str:
        """Format a single manifest line."""
        return f"{file_id}\t{filename}\t{md5sum or ''}\t{file_size}\t{state or ''}"

    def _write_manifest(self, lines: list, output_path: Path):
        """Write manifest to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))

    def create_subset_manifest(
        self,
        manifest_path: Path,
        output_path: Path,
        max_files: int,
    ) -> Path:
        """Create a subset manifest with only the first N files.

        Useful for testing downloads without fetching everything.

        Args:
            manifest_path: Path to original manifest
            output_path: Where to save subset manifest
            max_files: Maximum number of files to include

        Returns:
            Path to created subset manifest
        """
        manifest_path = Path(manifest_path)
        output_path = Path(output_path)

        with open(manifest_path) as f:
            lines = f.readlines()

        # Keep header + first N files
        subset_lines = lines[:max_files + 1]  # +1 for header

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(subset_lines))

        return output_path

    def select_stratified(
        self,
        df: pd.DataFrame,
        max_files: int,
        group_col: str = "project_id",
    ) -> pd.DataFrame:
        """Pick up to ``max_files`` rows balanced across ``group_col`` (round-robin).

        A plain head(N) of the slide table returns whatever project sorts first
        (e.g. all LUAD), which makes subtype/gene tasks degenerate on a small
        sample. This instead takes one row from each project in turn until it has
        ``max_files``, so the sample spans LUAD/LUSC/LGG/GBM. Falls back to the
        head of ``df`` when ``group_col`` is absent.

        Args:
            df: slide table from TCGASlideETL.build_slide_table()
            max_files: number of rows to select (<=0 or None → return df unchanged)
            group_col: column to balance across (default: project_id)

        Returns:
            A new DataFrame with the selected rows.
        """
        if not max_files or max_files <= 0 or len(df) <= max_files:
            return df.copy()
        if group_col not in df.columns:
            return df.head(max_files).copy()

        groups = [g.reset_index(drop=True) for _, g in df.groupby(group_col, sort=True)]
        picked = []
        depth = 0
        while len(picked) < max_files:
            advanced = False
            for g in groups:
                if depth < len(g):
                    picked.append(g.iloc[depth])
                    advanced = True
                    if len(picked) >= max_files:
                        break
            if not advanced:  # every group exhausted
                break
            depth += 1

        return pd.DataFrame(picked).reset_index(drop=True)

    def select_by_byte_budget(self, df, target_gb, group_col="project_id", size_col="file_size"):
        """Pick a stratified subset whose total slide bytes reach ~``target_gb``.

        Round-robins across ``group_col`` (like select_stratified) but stops once
        the cumulative ``file_size`` hits the budget -- used to grab "~N GB / ~X%
        of the corpus" as a reusable on-scratch cache. Falls back to the whole df
        if the size/group columns are missing.

        Args:
            df: slide table with a per-slide ``file_size`` column
            target_gb: size budget in GiB (<=0 or None → return df unchanged)
            group_col: column to balance across (default: project_id)
            size_col: per-row byte-size column (default: file_size)

        Returns:
            A new DataFrame with the selected rows.
        """
        if not target_gb or target_gb <= 0:
            return df.copy()
        if size_col not in df.columns or group_col not in df.columns:
            return df.copy()

        budget = float(target_gb) * (1024 ** 3)
        groups = [g.reset_index(drop=True) for _, g in df.groupby(group_col, sort=True)]
        picked, total, depth = [], 0.0, 0
        while total < budget:
            advanced = False
            for g in groups:
                if depth < len(g):
                    row = g.iloc[depth]
                    picked.append(row)
                    total += float(row.get(size_col) or 0)
                    advanced = True
                    if total >= budget:
                        break
            if not advanced:  # every group exhausted before hitting the budget
                break
            depth += 1

        return pd.DataFrame(picked).reset_index(drop=True)
