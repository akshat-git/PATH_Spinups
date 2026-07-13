# Pathology Foundation Models (PFM) benchmark on Sherlock

Benchmark **11 pathology foundation models** head-to-head on TCGA, on one footing
and identical metrics. Every model is used as a **frozen encoder**: turn each slide
into one embedding vector, train a tiny linear probe per task, and compare. The
whole stack is containerized (Apptainer), one isolated venv per model, and **all
I/O lives on `$SCRATCH`** — nothing touches `$HOME` or group storage. Every number
comes from real TCGA slides (GDC) and real published weights.

---

## 1 · Inputs → Outputs at a glance

If you only know the two ends, this is the whole thing:

```
 INPUT                                                                         OUTPUT
 ─────                                                                         ──────
 GDC / TCGA                                                                    a leaderboard:
   projects: LUAD, LUSC, LGG, GBM         ┌───────────────────┐               "which PFM is
   genes:    KRAS, TP53, EGFR, IDH   ───►  │   THIS PIPELINE   │  ───►          best on each
   (whole-slide images + mutations)       └───────────────────┘               TCGA task?"
                                                                               results.csv + heatmap_auroc.png
                                                                               (11 models × 6 tasks)
```

Expanded one level — two containers, one hand-off (a folder of thumbnails + a label table):

```
                 ┌──────────────────────────────────────────────────────────────────┐
   GDC REST      │  CPU CONTAINER  runtime/containers/tcga_build.sif                  │
   api.gdc       │                                                                    │
   .cancer.gov ──┼─►  query metadata ─► pick a stratified subset ─► fetch slides ─►   │
                 │    render a small thumbnail per slide ─► build a label table       │
                 └───────────────────────────────┬────────────────────────────────────┘
                                                  │   HAND-OFF (the only contract):
                                                  │     runtime/tcga/thumbnails/<slide_id>.jpg
                                                  │     runtime/tcga/tables/dataset.csv
                                                  ▼
                 ┌──────────────────────────────────────────────────────────────────┐
   GPU + torch   │  GPU CONTAINER  runtime/containers/pfm_base.sif                    │
                 │                                                                    │
                 │    for each model:  thumbnail ─► frozen encoder ─► embedding[N×D]  │
                 │    for each (model,task):  embeddings + labels ─► linear probe ─►  │
                 │                            accuracy / balanced-acc / F1 / AUROC    │
                 └───────────────────────────────┬────────────────────────────────────┘
                                                  ▼
                              runtime/embeddings/benchmark/results.{csv,json}
                              runtime/embeddings/benchmark/heatmap_auroc.png
```

Why two containers: the data build is CPU-only (GDC + openslide); model extraction
needs GPU + torch. Separate images so their Python worlds never collide.

---

## 2 · The whole pipeline, stage by stage

Each stage below lists **what goes in**, **what comes out** (the exact file), and
**which container** runs it. Follow the artifacts and you can trace one slide from a
GDC UUID all the way to a row in the leaderboard.

### Stage 1 — Query GDC → a flat slide table   ·   `tcga_build.sif` · `etl.py`

```
 in:  project ids (LUAD/LUSC/LGG/GBM), access=open
 ───────────────────────────────────────────────────────────────────────────
   GDCClient._paginate("files")  ──REST──►  api.gdc.cancer.gov
       filters: data_type="Slide Image", experimental_strategy="Diagnostic Slide"
       + per case: MAF (mutation) file ids for the gene labels
 ───────────────────────────────────────────────────────────────────────────
 out: tables/slide_table.parquet   (one row per slide)
      columns:  file_id  slide_id  project_id  filename  file_size  md5sum
                maf_file_id  maf_filename  has_maf  ...
```

`file_id` is the GDC UUID (used to download); `slide_id` is the join key used
everywhere downstream. `file_size` is what the byte-budget subset uses next.

### Stage 2 — Pick a stratified subset   ·   `manifest.py`

A full corpus is ~2000 slides / ~500 GB. We take a **balanced slice** so every
project is represented (else subtype tasks collapse to one class). Two selectors:

```
   full slide table (all projects)
            │
            ├─ select_by_byte_budget(target_gb=50)   ← FINAL run (staged full-SVS)
            │     round-robin LUAD→LUSC→LGG→GBM→LUAD… adding slides,
            │     summing file_size, stop at ~50 GB           ┐
            │                                                  ├─►  ~50 GB, ~balanced
            └─ select_stratified(max_files=N)        ← proof/smoke (by count)
                  round-robin one slide per project until N    ┘  e.g. 60 = ~15 each

   out: manifests/slides_manifest_subset.txt   (the chosen slides)
        manifests/maf_manifest_subset.txt       (their mutation files → labels align)
```

```
   WITHOUT stratification (head of table)        WITH stratification (round-robin)
   ┌──────────────────────────────┐             ┌──────────────────────────────┐
   │ LUAD LUAD LUAD LUAD LUAD LUAD │  ✗ one      │ LUAD LUSC LGG  GBM  LUAD LUSC │  ✓ all
   │ LUAD LUAD LUAD LUAD LUAD LUAD │    class →   │ LGG  GBM  LUAD LUSC LGG  GBM  │    classes →
   └──────────────────────────────┘    lung-only └──────────────────────────────┘    every task
     luad_vs_lusc, lgg_vs_gbm degenerate           luad_vs_lusc & lgg_vs_gbm trainable
```

### Stage 3 — Acquire + preprocess the slides   ·   `tcga_build.sif`

Every strategy shares one **acquisition fallback chain** (per slide, resumable):
**preprocessed artifact already on `$SCRATCH` → skip** · else **full SVS pre-downloaded
to the cache → use it (no download)** · else **stream the SVS into node-local `$TMPDIR`,
use it, evict** (nothing large persists). Only the small artifact (thumbnail or patches)
is kept. (The middle branch is used only if an SVS cache was pre-filled via the
`download_svs_cache` pipeline step; otherwise every slide streams on demand.)

**(C) FINAL run — patch tiling → GPU-bound** (`slide_tiler.py`, step `tile_slides`,
`configs/tcga_tiled.yaml`). Each slide is cut into **many tissue patches** (real WSI
workflow) instead of one thumbnail, so extraction is GPU-bound, not I/O-bound:

```
   acquire the SVS (fallback chain above) → node-local $TMPDIR/<name>.svs
        │  Otsu tissue mask on a downsampled thumbnail (adaptive per slide) →
        │  keep EVERY grid cell with ≥ tissue_thresh tissue (uncapped: ~16k/slide;
        │  ~22.6M total for the full ~1400 slides, ~2.26M for the 50 GB mini subset)
        │  write each patch DIRECTLY into one tar per slide (no millions of loose files)
        ▼
   PERSISTENT  runtime/<root>/patches_tar/<slide_id>.tar   ← tiled ONCE, reused every run
        │        (members: <slide_id>__x<X>_y<Y>.jpg; staging reads ~N_slides big files)
        └─ evict the node-local .svs
```

**(A) staged full-SVS → 1 thumbnail/slide** (`slide_stager.py`, step `stage_process`,
`configs/tcga_staged.yaml`, selected via `FINAL_CONFIG`). Same acquisition, but produces
one 512×512 thumbnail per slide instead of patches (fast, I/O-bound, coarse — a slide
overview, not tissue detail).

### Stage 4 — Assemble the label table   ·   `gene_matrix.py` + `pipeline.py`

```
   MAF mutation files (subset) ─► gene_matrix: aliquot → sample → per-gene 0/1
                                                                    │
   slide_table (+ jpg_path from Stage 3)  ───────── merge on sample ┘
                                                                    ▼
   out:  runtime/tcga/tables/dataset.csv          ← THE label source
         one row per slide:
           slide_id │ jpg_path │ project_id │ has_maf │ KRAS │ TP53 │ EGFR │ IDH
```

### Stage 5 — Extract embeddings (per model)   ·   `pfm_base.sif` · `runner.py`

This is the core "frozen encoder" step. Same loop for every model; the only
per-model code is a ~15-line `<model>_spin.py` giving `load()` and `embed()`.

```
   PFM_PATCH_DIR  = patches/  (tiling: many/slide)  OR  thumbnails/ (1/slide)
            │  data.find_patch_images() [recursive] → torch Dataset → DataLoader
            │  batch=PFM_BATCH_SIZE (64), num_workers=PFM_NUM_WORKERS (8, feeds the GPU)
            ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  <model>_spin.load()  → (frozen model, transform)                │
   │  batch of images ─► transform ─► [B,3,H,W] ─► model.eval()       │
   │                     (autocast fp16 on GPU, no grad)             │
   │  <model>_spin.embed(model, batch)  ─►  [B, D]                     │
   └─────────────────────────────────────────────────────────────────┘
            │  concat over all batches
            ▼
   runtime/<out>/embeddings/<model>/patch_embeddings.pt
       = { "model": <name>, "embeddings": Tensor[N, D], "paths": [<jpg path>, …] }
       N = # images (patches, or slides for thumbnails);  D = embedding width
```

> **GPU-bound vs I/O-bound.** With thumbnails, N ≈ 142 → the GPU finishes in seconds
> (extraction was never the bottleneck; the ~20 min download was). With tiling, N ≈ 100k+
> → the GPU is the worker. `num_workers=8` (not 2) keeps the dataloader ahead of the GPU;
> it was the single biggest lever (~15× faster/patch once the GPU stopped starving).

If a model is gated and the token lacks access, `runner` prints **`ACCESS NOT
GRANTED`** and exits `75` (that model is disregarded, not a crash). `titan` is a
*slide* encoder (needs precomputed CONCH `.h5` features, not thumbnails) → it loads
but produces no patch embeddings here ("load-only, expected").

### Stage 6 — Train a linear probe per (model × task)   ·   `benchmark.py`

The head-to-head. For every model that produced embeddings, and every task:

```
   patch_embeddings.pt [N×D]                 dataset.csv (labels)
            │                                      │
   mean-pool patches → 1 vector per slide         │
     (slide_id = basename before "__";            │
      thumbnails = no "__" → 1 row already)        │
            └──────── join on slide_id ────────────┘
                          │
                          ▼
              keep slides that have a label for this task
                          │  stratified train/val split (keeps class balance)
                          ▼
              fit_linear_probe:  z-score features → Linear(D→#classes)
                                 AdamW + CrossEntropy   (encoder stays FROZEN)
                          │
                          ▼
              metrics.compute_all → accuracy · balanced_accuracy · macro_F1 · AUROC
```

Run over the full grid, this is a matrix — one metric per cell:

```
                 luad_vs_lusc   lgg_vs_gbm   kras   tp53   egfr   idh
   conch            0.xx           0.xx      0.xx   0.xx   0.xx   —
   uni2             0.xx           0.xx      0.xx   0.xx   0.xx   —
   virchow          …              …         …      …      …      …
   …                                                              (AUROC per cell)
   phikon           …
```

Out: `runtime/embeddings/benchmark/results.csv` (scalars, one row per model×task)
and `results.json` (adds per-class precision/recall + confusion matrices).

### Stage 7 — Plot   ·   `plot_results.py`

```
   results.csv ─►  printed leaderboard (ranked by mean AUROC)
               ─►  summary_<metric>.csv        (model × task matrix)
               ─►  heatmap_<metric>.png        (default metric = AUROC)
```

---

## 3 · The one contract that ties it together: `slide_id`

Everything connects through a single key. There is no database — just this
convention, so it's worth seeing explicitly:

```
   GDC file  ──►  slides/<file_id>/TCGA-XX-….svs
                        │ openslide/tifffile thumbnail
                        ▼
                  thumbnails/<slide_id>.jpg ───────────────┐
                        │ runner saves the path with each   │  join key =
                        │ embedding                          │  basename(path)
                        ▼                                    │  without ".jpg"
                  patch_embeddings.pt  "paths":[…<slide_id>.jpg…]
                        │                                    │
                  dataset.csv  slide_id,jpg_path,labels… ────┘
                        ▲
                        └ ETL writes slide_id here too
```

If a model's `paths` and `dataset.csv`'s `slide_id`s don't share values, the
benchmark join is empty — that shared key is the whole integration.

---

## 4 · The 11 models

`D` is the embedding width each model emits (the columns of `patch_embeddings.pt`).

| Model | Type | HF repo | Gated | D (emb dim) |
|---|---|---|---|---|
| conch | patch | MahmoodLab/CONCH (pip zip archive) | yes | 512 |
| uni2 | patch | MahmoodLab/UNI2-h | yes | 1536 |
| virchow | patch | paige-ai/Virchow | yes | 2560 |
| virchow2 | patch | paige-ai/Virchow2 | yes | 2560 |
| phikon | patch | owkin/phikon-v2 | **no** | 1024 |
| exaone-path | patch | LGAI-EXAONE/EXAONE-Path-2.5 | yes | 768 |
| gigapath | patch (tile) | prov-gigapath/prov-gigapath | yes | 1536 |
| h-optimus | patch | bioptimus/H-optimus-0 | yes | 1536\* |
| mSTAR | patch | Wangyh/mSTAR (pip zip archive) | yes | 1024 |
| hipt | patch (ViT-256) | mahmoodlab/HIPT (ckpt from GitHub LFS) | no† | 384 |
| titan | **slide** | MahmoodLab/TITAN | yes | (slide encoder) |

\* h-optimus needs an HF access grant to `bioptimus/H-optimus-0`; without it the run
reports `NO-ACCESS` and moves on. † HIPT weights aren't on HF — the ViT-256
checkpoint is fetched from the GitHub repo (git-LFS); set `PFM_HIPT_CKPT` or let the
job fetch it. `titan` consumes precomputed CONCH `.h5` patch features, so in this
thumbnail pipeline it's load-only (by design).

## 5 · The 6 tasks (`pfm_common/tasks.py`)

| Task | Question | Label source | Positive/negative |
|---|---|---|---|
| `luad_vs_lusc` | lung subtype | `project_id` | LUAD vs LUSC |
| `lgg_vs_gbm` | glioma subtype | `project_id` | LGG vs GBM |
| `kras` | KRAS mutated? | gene matrix 0/1 | sequenced tumors (`has_maf`) |
| `tp53` | TP53 mutated? | gene matrix 0/1 | sequenced tumors |
| `egfr` | EGFR mutated? | gene matrix 0/1 | sequenced tumors |
| `idh` | IDH mutated? | gene matrix 0/1 | sequenced tumors |

> Known caveat: the config gene is `IDH` but the MAF symbol is `IDH1`/`IDH2`, so
> `idh` currently fills 0 for everyone and is skipped. Fix = use `IDH1`/`IDH2`.

---

## 6 · Repository layout

```
stanf_pfm/
├── pfm_setup.sh        # ★ driver: build container, per-model venvs, extract, benchmark (Slurm-aware)
├── README.md           # this file
├── CLAUDE.md           # current-state reference / working notes
├── models.txt     # the 11 model folder names
│
├── pfm_common/         # SHARED package imported by every model adapter
│   ├── config.py       #   env-driven paths & knobs; device()=cuda|cpu; single source of truth
│   ├── hf_auth.py      #   HuggingFace login from $HF_TOKEN (gated models only)
│   ├── data.py         #   find thumbnails/patches → torch Dataset/DataLoader + collate
│   ├── runner.py       #   ★ the shared extract loop (Stage 5): auth → batch → encode → save .pt
│   ├── tasks.py        #   the 6 task definitions (row filter + label column)
│   ├── train_probe.py  #   fit one linear probe over frozen embeddings
│   ├── metrics.py      #   accuracy, balanced acc, macro-F1, AUROC, confusion (numpy only)
│   ├── benchmark.py    #   ★ probe every (model × task) → results.{csv,json}  (Stage 6)
│   └── plot_results.py #   leaderboard + heatmap + summary CSV  (Stage 7)
│
├── models/            # the 11 model adapters (each: <model>/<model>_spin.py, ~15 lines)
│                       #   load()→(model,transform), embed()→[B,D]
│
├── build_tcga_dataset.py   # ETL CLI: `python -m build_tcga_dataset --config configs/…`
├── tcga/              # the TCGA/GDC ETL package — the "data" half (Stages 1–4)
│   ├── gdc_client.py       # generic GDC REST wrapper (queries; /data supports Range)
│   ├── etl.py              # flat slide table (Stage 1)
│   ├── manifest.py         # manifests + select_stratified + select_by_byte_budget (Stage 2)
│   ├── downloader.py       # threaded GDC REST download, md5-verified, resumable
│   ├── slide_stager.py     # ★ acquire SVS (cache/stream) → 1 thumbnail/slide  (Stage 3A)
│   ├── slide_tiler.py      # ★ Otsu tissue detection + patch tiling → many/slide (Stage 3C)
│   ├── slide_streamer.py   # ★ HTTP-Range thumbnail streaming, no full download (Stage 3B)
│   ├── slide_processor.py  # openslide SVS→JPG thumbnails
│   ├── gene_matrix.py      # MAF → per-gene 0/1 matrix (Stage 4)
│   ├── pipeline.py         # TCGADatasetBuilder: chains the steps
│   └── README.md           # ETL reference (diagrams, steps, config)
├── configs/
│   ├── tcga_tiled.yaml     # ★ DEFAULT: patch tiling → GPU-bound (tile_slides, uncapped)
│   └── tcga_staged.yaml    # staged full-SVS → 1 thumbnail/slide (FINAL_CONFIG override)
├── jobs/
│   ├── setup_tcga.sh          # one-time: build tcga_build.sif + venv (CPU/normal)
│   ├── final_setup.sh         # ★ FINAL run (GPU): self-bootstraps → tile/persist ALL patches → all models → train
│   ├── final_setup_mini.sh    # ★ same, but extract on a 1/MINI_FRACTION (default 10%) sample → short walltime
│   └── verify_tcga_env.py     # sanity-check the data-build venv imports (used by setup_tcga.sh)
│
└── runtime/            # ALL generated artifacts (on $SCRATCH, never $HOME/group). Git-ignored.
    ├── containers/     #   pfm_base.sif (torch) + tcga_build.sif (data)
    ├── venvs/          #   one venv per model + tcga_build
    ├── repos/HIPT/     #   cloned HIPT (ViT-256 code + checkpoint)
    ├── cache/          #   HF_HOME, pip, torch, apptainer caches (off $HOME)
    ├── <root>/         #   PFM_TCGA_ROOT: patches/<slide_id>/ (tiled, persistent) OR thumbnails/,
    │                   #     + slides/ (SVS cache only if pre-downloaded) + tables/dataset.csv
    └── <out>/          #   PFM_OUTPUT_DIR: <model>/patch_embeddings.pt + benchmark/{results.csv,heatmap}
```

> **Patches persist.** With tiling, `PFM_TCGA_ROOT/patches/` is written **once** and reused
> every run (the top of the acquisition fallback chain) — no re-tiling, no re-download.
> Full SVS never persist (each is streamed into node-local temp, tiled, and evicted).

---

## 7 · Run it

All heavy work goes through Slurm (never the login node). Both entrypoints are
**fully self-bootstrapping** — on a fresh checkout (after you add `runtime/.hf_token`)
they build every missing piece (data container/venv, dataset+patches, model container +
per-model venvs) before extracting. Nothing is assumed to pre-exist.

```bash
mkdir -p logs

# ── FULL run: 100% of ALL ~1400 slides (~500 GB TCGA), 8× H100, data-parallel ─
# Tiles each SVS -> per-slide tar; extraction shards each model across 8 GPUs and
# mean-pools to slide level. -G 8 -C GPU_MEM:80GB, --nodes=1, ~30 h walltime.
sbatch jobs/final_setup.sh
#   knobs: FINAL_TARGET_GB (cap size; default = null = ALL 500 GB), PFM_RUN_MODE
#          (shard|queue), FINAL_CONFIG, PFM_TCGA_ROOT, PFM_OUTPUT_DIR

# ── MINI run: 1% sample of a ~50 GB subset, 4 general GPUs → ~15 min, 30 min cap
sbatch jobs/final_setup_mini.sh
#   knobs: MINI_FRACTION (default 100 → 1/100 = 1%), FINAL_TARGET_GB (default 50)

# ── watch ────────────────────────────────────────────────────────────────────
squeue --me
tail -f logs/tcga_final_<jobid>.out        # STEP 1→5
tail -f logs/tcga_final_mini_<jobid>.out   # mini
```

Piece by piece (from an interactive GPU alloc, `salloc -p gpu -G 1`):

```bash
./pfm_setup.sh build                  # pull/build pfm_base.sif
./pfm_setup.sh setup [model…]         # create venv(s) + install deps
./pfm_setup.sh run   [model…]         # extract embeddings → runtime/embeddings/<model>/…
./pfm_setup.sh benchmark --dataset-csv $PFM_TCGA_ROOT/tables/dataset.csv
./pfm_setup.sh shell  <model>         # interactive shell inside a model's venv
```

---

## 8 · Environment knobs (all optional; sensible defaults)

| Var | Default | Meaning |
|---|---|---|
| `PFM_ROOT` | `<repo>/runtime` | root for all scratch I/O |
| `PFM_TCGA_ROOT` | `$PFM_ROOT/tcga` | dataset root (`patches/`, `thumbnails/`, `slides/`, `tables/dataset.csv`) |
| `PFM_PATCH_DIR` | — | explicit dir of encoder-input images (the persisted `patches/`, or thumbnails) |
| `PFM_OUTPUT_DIR` | `$PFM_ROOT/embeddings` | where embeddings + benchmark land |
| `PFM_BATCH_SIZE` / `PFM_NUM_WORKERS` | 8 / 2 | DataLoader batch size / workers (final_setup sets 64 / 8 for tiling) |
| `FINAL_CONFIG` | `configs/tcga_staged.yaml` | which acquisition config `final_setup.sh` uses (set `configs/tcga_tiled.yaml` for patch tiling) |
| `FINAL_TARGET_GB` | (config) | stratified SVS-subset size to acquire |
| `HF_TOKEN` | — | HuggingFace token (gated models); or put it in `runtime/.hf_token` |
| `PFM_HIPT_CKPT` | `$PFM_ROOT/repos/HIPT/.../vit256_small_dino.pth` | HIPT ViT-256 checkpoint |

ETL config (YAML): `download.target_gb` (subset size), `download.max_files` (subset by
count), `download.stream_to_local` (stream vs resident SVS cache), `patches.{patch_size,
level,tissue_thresh,max_patches}` (tiling), `slides.stage_download_workers`,
`gene_matrix.genes`, `steps`.

---

## 9 · Results

<!-- RESULTS: populated from the final run (job artifacts under
     runtime/embeddings/benchmark/). Until then this documents the shape. -->

_Populated after the final run completes._ The run writes:

- `runtime/embeddings/benchmark/results.csv` — one row per (model, task): accuracy,
  balanced accuracy, macro-F1, AUROC.
- `runtime/embeddings/benchmark/heatmap_auroc.png` — the model × task matrix.
- per-model `runtime/embeddings/<model>/patch_embeddings.pt` — shape `[N, D]`.

Expected coverage: **9 patch encoders extract + train** (conch, uni2, virchow,
virchow2, phikon, exaone-path, gigapath, mSTAR, hipt), **titan** is load-only, and
**h-optimus** is `NO-ACCESS` until its HF grant. Leaderboard numbers, the AUROC
heatmap, and the exact `N` land here once `final_setup.sh` finishes.
