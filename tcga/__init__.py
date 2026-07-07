# src/data/tcga/__init__.py
"""
TCGA / GDC Data Module

Provides tools for querying and managing TCGA data from the Genomic Data Commons.

Example:
    from tcga import GDCClient

    client = GDCClient()
    projects = client.list_projects(program="TCGA")
    cases = client.get_cases(project_id="TCGA-BRCA", max_results=10)

ETL Example:
    from tcga import TCGASlideETL, TCGAConfig

    config = TCGAConfig(project_ids=["TCGA-LUAD"])
    etl = TCGASlideETL()
    df = etl.build_slide_table(
        project_ids=config.project_ids,
        include_demographics=True,
        include_diagnosis=True,
        include_maf=True,
    )
    df = etl.add_local_paths(df, config)

Manifest & Download Example:
    from tcga import ManifestGenerator, TCGADownloader

    manifest_gen = ManifestGenerator()
    manifest_gen.create_slide_manifest(df, config.manifests_dir / "slides.txt")

    downloader = TCGADownloader()
    result = downloader.download_from_manifest(manifest_path, config.slides_dir)
"""

from tcga.gdc_client import (
    # Main client
    GDCClient,

    # Filter building
    GDCFilterBuilder,
    FilterOp,

    # Data classes
    GDCProject,
    GDCCase,
    GDCFile,
    GDCAnnotation,

    # Field reference classes (for documentation, prefer discover_fields())
    CaseFields,
    FileFields,
    ProjectFields,
    AnnotationFields,
)

from tcga.hierarchy import (
    HierarchyBuilder,
    HierarchyNode,
)

from tcga.etl import (
    TCGASlideETL,
)

from tcga.config import (
    TCGAConfig,
)

from tcga.manifest import (
    ManifestGenerator,
)

from tcga.downloader import (
    TCGADownloader,
    DownloadStatus,
    DownloadResult,
)

from tcga.gene_matrix import (
    GeneMatrix,
)

from tcga.slide_processor import (
    SlideProcessor,
    ProcessingResult,
)

from tcga.pipeline import (
    TCGADatasetBuilder,
)

__all__ = [
    # Client
    "GDCClient",
    "GDCFilterBuilder",
    "FilterOp",
    # Data classes
    "GDCProject",
    "GDCCase",
    "GDCFile",
    "GDCAnnotation",
    # Field references
    "CaseFields",
    "FileFields",
    "ProjectFields",
    "AnnotationFields",
    # Hierarchy
    "HierarchyBuilder",
    "HierarchyNode",
    # ETL
    "TCGASlideETL",
    # Config
    "TCGAConfig",
    # Manifest
    "ManifestGenerator",
    # Downloader
    "TCGADownloader",
    "DownloadStatus",
    "DownloadResult",
    # Gene Matrix
    "GeneMatrix",
    # Slide Processor
    "SlideProcessor",
    "ProcessingResult",
    # Pipeline
    "TCGADatasetBuilder",
]
