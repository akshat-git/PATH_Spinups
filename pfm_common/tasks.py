"""Downstream task definitions over the TCGA dataset table.

The label source is the `dataset.csv` produced by the TCGA/GDC ETL pipeline in
`tcga/` (see `tcga/README.md` -- run the pipeline to build it). Each row is one
slide with columns `slide_id`, `jpg_path`, `project_id`, `has_maf`, and one 0/1
column per gene (TP53, KRAS, ...).

A task is just a row filter + a column to read the label from:

    luad_vs_lusc / lgg_vs_gbm : cancer-subtype classification (from project_id)
    kras / tp53 / egfr / idh  : gene-mutation prediction      (0/1, tumors only)

`labels_for_task` returns a {slide_id -> int label} map plus class names; the
benchmark joins that onto each model's frozen embeddings via slide_id.
"""
import csv

# Each entry: how to filter rows and where the label comes from.
#   filter_col / filter_values : keep only rows whose filter_col is in filter_values
#   require_maf                : keep only sequenced tumors (rows with has_maf == True)
#   label_source               : column to read the label from
#   label_map                  : map string values -> int class (subtype tasks)
#   class_names                : human-readable names, index = class id
TASK_REGISTRY = {
    "luad_vs_lusc": {
        "filter_col": "project_id",
        "filter_values": ["TCGA-LUAD", "TCGA-LUSC"],
        "label_source": "project_id",
        "label_map": {"TCGA-LUAD": 0, "TCGA-LUSC": 1},
        "class_names": ["LUAD", "LUSC"],
        "require_maf": False,
    },
    "lgg_vs_gbm": {
        "filter_col": "project_id",
        "filter_values": ["TCGA-LGG", "TCGA-GBM"],
        "label_source": "project_id",
        "label_map": {"TCGA-LGG": 0, "TCGA-GBM": 1},
        "class_names": ["LGG", "GBM"],
        "require_maf": False,
    },
    "kras": {"label_source": "KRAS", "require_maf": True, "class_names": ["KRAS-wt", "KRAS-mut"]},
    "tp53": {"label_source": "TP53", "require_maf": True, "class_names": ["TP53-wt", "TP53-mut"]},
    "egfr": {"label_source": "EGFR", "require_maf": True, "class_names": ["EGFR-wt", "EGFR-mut"]},
    # IDH status is IDH1 OR IDH2 mutated -- the MAF Hugo_Symbol is IDH1/IDH2, never a bare
    # "IDH", so the old single-column "IDH" label filled 0 for everyone. label_any_of ORs
    # the two gene columns (requires IDH1/IDH2 in the config's gene_matrix.genes).
    "idh":  {"label_any_of": ["IDH1", "IDH2"], "require_maf": True, "class_names": ["IDH-wt", "IDH-mut"]},
}

ALL_TASKS = list(TASK_REGISTRY)


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def _as01(v):
    """Coerce a gene-matrix cell to 0/1 (mutated=1). Blank/unparseable -> 0."""
    try:
        return 1 if int(float(v)) != 0 else 0
    except (TypeError, ValueError):
        return 0


def labels_for_task(dataset_csv, task):
    """Build the {slide_id -> int label} map for one task from dataset.csv.

    Rows are dropped when they lack a thumbnail (`jpg_path`), fail the task's
    row filter, or have a missing/unmappable label. Returns (labels, class_names).
    """
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Choose from: {ALL_TASKS}")
    cfg = TASK_REGISTRY[task]
    labels = {}
    with open(dataset_csv, newline="") as f:
        for r in csv.DictReader(f):
            if "slide_id" not in r:
                raise SystemExit(
                    f"{dataset_csv} has no 'slide_id' column -- is it the TCGA ETL "
                    f"dataset.csv? (columns: {list(r.keys())[:8]}...)"
                )
            # Do NOT gate on jpg_path. The thumbnail workflow sets jpg_path, but the
            # TILED workflow keys patches by slide_id and leaves jpg_path unset -- gating
            # on it dropped every tiled slide, so the benchmark saw "no labelled rows".
            # benchmark_one only uses labels for slides that actually have embeddings
            # (joined by slide_id), so labelling a slide with no embedding is harmless.
            if cfg.get("require_maf") and not _truthy(r.get("has_maf", "")):
                continue
            if "filter_values" in cfg and r.get(cfg["filter_col"]) not in cfg["filter_values"]:
                continue
            if "label_any_of" in cfg:
                cols = cfg["label_any_of"]
                vals = [r.get(c) for c in cols]
                if all(v in (None, "") for v in vals):
                    continue  # no gene-matrix data for these columns (unsequenced)
                labels[r["slide_id"]] = 1 if any(_as01(v) for v in vals) else 0
                continue
            raw = r.get(cfg["label_source"], "")
            if raw is None or raw == "":
                continue
            if "label_map" in cfg:
                if raw not in cfg["label_map"]:
                    continue
                labels[r["slide_id"]] = cfg["label_map"][raw]
            else:
                try:
                    labels[r["slide_id"]] = int(float(raw))
                except (TypeError, ValueError):
                    continue
    return labels, cfg["class_names"]
