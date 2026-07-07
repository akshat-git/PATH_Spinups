"""Sanity-check that the lean TCGA data-build venv has everything the pipeline
needs. Run inside the container after `pip install` (see jobs/setup_tcga.sh).
Exits non-zero on the first missing import so the setup job fails loudly."""
import sys

print("Python:", sys.version.split()[0])

print("Testing TCGA pipeline imports...")
from tcga.pipeline import TCGADatasetBuilder  # noqa: F401
from tcga.etl import TCGASlideETL  # noqa: F401
from tcga.downloader import TCGADownloader  # noqa: F401
from tcga.slide_processor import SlideProcessor  # noqa: F401
from tcga.gene_matrix import GeneMatrix  # noqa: F401
from tcga.manifest import ManifestGenerator  # noqa: F401
from tcga.config import TCGAConfig  # noqa: F401
print("All TCGA pipeline imports OK")

import openslide  # noqa: E402
import pandas  # noqa: E402, F401
import pyarrow  # noqa: E402, F401
import omegaconf  # noqa: E402, F401
print("OpenSlide:", openslide.__version__)
print("pandas / pyarrow / omegaconf OK")
