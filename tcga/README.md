# TCGA / GDC ETL — the "data" half of the PFM benchmark

This package turns **GDC metadata + whole-slide images** into the two artifacts the
model half consumes: a folder of per-slide thumbnails and a label table.

```
   INPUT                                                 OUTPUT
   ─────                                                 ──────
   GDC (api.gdc.cancer.gov)          ┌──────────┐        runtime/tcga/thumbnails/<slide_id>.jpg
     projects LUAD/LUSC/LGG/GBM ───► │   ETL    │ ───►   runtime/tcga/tables/dataset.csv
     genes KRAS/TP53/EGFR/IDH        └──────────┘        (one row per slide: id, path, labels)
```

Driven by `python -m build_tcga_dataset --config <yaml>`, which runs a list
of `steps` through `TCGADatasetBuilder` (`pipeline.py`). Every step is **resumable**
(skips work whose output already exists) and writes only under `data_dir`
(bind-mounted `/tcga_data` → `$PFM_TCGA_ROOT` on `$SCRATCH`). The downloader is a
**pure GDC REST client** (`GET /data/<uuid>`, threaded, md5-verified, resumable) —
there is **no `gdc-client` binary** anywhere.

---

## Pipeline steps (what each produces)

```
 etl ─► manifest ─► [ stage_process | stream_thumbnails ] ─► download(maf) ─► gene_matrix ─► assemble
  │        │                │                                    │              │             │
  │        │                │                                    │              │             └─ tables/dataset.csv  ← FINAL
  │        │                │                                    │              └─ tables/gene_matrix.parquet
  │        │                │                                    └─ maf/<id>/<file>.maf.gz   (mutations)
  │        │                └─ thumbnails/<slide_id>.jpg   (+ jpg_path in the slide table)
  │        └─ manifests/{slides,maf}_manifest[_subset].txt
  └─ tables/slide_table.parquet   (one row per slide: file_id, slide_id, project_id, file_size, maf_*)
```

| Step | Module | In → Out |
|---|---|---|
| `etl` | `etl.py` + `gdc_client.py` + `hierarchy.py` | GDC REST query → `slide_table.parquet` |
| `manifest` | `manifest.py` | slide table → manifests **+ stratified subset** |
| `stage_process` | `slide_stager.py` | **staged full-SVS**: cache on `$SCRATCH` + node-local thumbnail |
| `stream_thumbnails` | `slide_streamer.py` | **range-stream**: embedded thumbnail only, no full download |
| `download` | `downloader.py` | manifest → files (here: MAF only; `download.slides: false`) |
| `process_slides` | `slide_processor.py` | (legacy) openslide SVS→JPG from pre-downloaded slides |
| `gene_matrix` | `gene_matrix.py` | MAF files → per-gene 0/1 matrix |
| `assemble` | `pipeline.py` | slide table + gene matrix → `dataset.csv` |

Default configs: `configs/tcga_staged.yaml` uses `stage_process` (the real
run); `configs/tcga_streaming.yaml` uses `stream_thumbnails` (lightweight).

---

## Stage 1 — Query GDC → flat slide table  (`etl.py`, `gdc_client.py`, `hierarchy.py`)

```
   GDCClient._paginate("files")  ──REST──►  api.gdc.cancer.gov
     filters: cases.project.project_id ∈ {LUAD,LUSC,LGG,GBM}
              data_type = "Slide Image",  access = "open",
              experimental_strategy = "Diagnostic Slide"
     hierarchy.py indexes case → sample → portion → slide; parent fields broadcast down
     + per case: MAF (Masked Somatic Mutation) file ids for the gene labels
   ────────────────────────────────────────────────────────────────
   → tables/slide_table.parquet   (one row per slide)
     file_id (GDC UUID) │ slide_id (join key) │ project_id │ filename
     file_size │ md5sum │ sample_id │ case_id │ maf_file_id │ has_maf │ …
```

## Stage 2 — Stratified subset  (`manifest.py`)

A full pull is ~2000 slides / ~500 GB. We take a **balanced slice** across projects
so subtype tasks aren't degenerate:

```
   select_by_byte_budget(target_gb=50)     select_stratified(max_files=N)
   round-robin LUAD→LUSC→LGG→GBM,          round-robin one slide per project
   summing file_size, stop at ~50 GB       until N slides
        │  (FINAL run)                          │  (proof/smoke)
        └───────────────┬──────────────────────┘
                        ▼
   manifests/slides_manifest_subset.txt   +   maf_manifest_subset.txt
   (MAF subset built from the SAME slides so gene labels line up;
    _resolve_manifest() prefers a *_subset.txt when present)
```

```
   WITHOUT stratification (head of table)     WITH stratification (round-robin)
   LUAD LUAD LUAD LUAD LUAD …  ✗ one class     LUAD LUSC LGG GBM LUAD LUSC …  ✓ all classes
   → luad_vs_lusc / lgg_vs_gbm degenerate      → every subtype/gene task trainable
```

## Stage 3 — Acquire slides → thumbnails  (two strategies, one output)

**A. Staged full-SVS** — `slide_stager.py` (`stage_process`, `tcga_staged.yaml`):

```
   GDC /data/<file_id> (full SVS ~250 MB–1.5 GB)     ── async: several slides in flight ──
        │ download (skip if cached)
        ▼
   $SCRATCH  runtime/tcga/slides/<file_id>/<name>.svs      ← persistent ~50 GB cache
        │ copy one slide
        ▼
   $L_SCRATCH/tcga_stage/<name>.svs  (node-local SSD)  ─ openslide.get_thumbnail(512²) ─┐
        │ evict local .svs after                                                         ▼
        └────────────────────────────────────────────────►  thumbnails/<slide_id>.jpg
   resumable: thumbnail exists → skip;  SVS cached → don't re-download
   (local disk never holds more than `stage_download_workers` slides at once)
```

**B. Range-streaming** — `slide_streamer.py` (`stream_thumbnails`, `tcga_streaming.yaml`):

```
   HTTPRangeFile(GDC /data/<file_id>) ── serves only requested byte ranges ──►
        tifffile opens the SVS remotely → reads only the embedded "Thumbnail" (~a few MB)
        (fallback: one bounded full download → thumbnail → delete, per slide)
        ▼
   thumbnails/<slide_id>.jpg     (nothing but the JPG hits disk; ~100× less transferred)
```

## Stage 4 — Labels → dataset.csv  (`gene_matrix.py`, `pipeline.py`)

```
   MAF files ─► gene_matrix: Hugo_Symbol per aliquot → sample → {gene: 0/1}
                                                                      │
   slide_table (+ jpg_path) ───────────────── merge on sample_id ────┘
                                                                      ▼
   tables/dataset.csv :  slide_id │ jpg_path │ project_id │ has_maf │ KRAS │ TP53 │ EGFR │ IDH │ …
```

`dataset.csv` (one row per slide, all metadata carried forward) is the label source;
the model half joins its embeddings by `slide_id = basename(jpg_path)` (no extension).

---

## Config knobs (`configs/tcga_dataset_*.yaml`)

| Key | Meaning |
|---|---|
| `projects` | GDC project ids to pull (LUAD/LUSC/LGG/GBM) |
| `access` | `open` (no token) or `controlled` (needs `download.token_path`) |
| `download.slides` | `false` in both configs — slides come via `stage_process`/`stream_thumbnails` |
| `download.maf` | download MAF files for gene labels |
| `download.target_gb` | staged subset size (byte budget) — FINAL run |
| `download.max_files` | subset by slide count — proof/smoke |
| `slides.thumbnail_size` | thumbnail dims (default 512×512) |
| `slides.stage_download_workers` | concurrent SVS download+stage workers |
| `slides.stream_fallback` | range-stream: full-download fallback if a thumbnail can't be range-read |
| `gene_matrix.genes` | genes to turn into 0/1 label columns |
| `steps` | which pipeline steps to run |

## CLI

```bash
python -m build_tcga_dataset --config configs/tcga_staged.yaml    # staged full-SVS
python -m build_tcga_dataset --config configs/tcga_streaming.yaml  # range-streaming
python -m build_tcga_dataset --steps etl,manifest --dry-run              # inspect, no execute
python -m build_tcga_dataset --config configs/tcga_staged.yaml download.target_gb=20
```

On the cluster: `jobs/final_setup.sh` (staged, GPU, end-to-end incl. training) or
`jobs/build_tcga_dataset.sh` (data only, CPU). See the top-level `README.md` for the
full model-side pipeline (extraction → benchmark → leaderboard).

---

## Details worth knowing

- **MAF is linked at the aliquot level, not the sample.** The `Tumor_Sample_UUID` in
  a MAF is an aliquot UUID; `GeneMatrix` resolves aliquot → sample via the GDC API so
  mutations join to the slide table on `sample_id`. Slides without MAF (e.g. normal
  tissue, un-sequenced tumors) get 0 for every gene (left join preserves all slides).
- **Resumable everywhere.** Re-running skips existing artifacts; interrupted downloads
  resume (`.part` files, md5 verified). Use `--force` to ignore caches.
- **Public API classes** (importable from `tcga`): `GDCClient`,
  `TCGASlideETL`, `ManifestGenerator`, `TCGADownloader`, `GeneMatrix`,
  `SlideProcessor`, `TCGAConfig`, `TCGADatasetBuilder`.
- **External refs:** GDC portal https://portal.gdc.cancer.gov/ · API docs
  https://docs.gdc.cancer.gov/API/Users_Guide/
