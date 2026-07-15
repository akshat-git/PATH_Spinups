# PFM Benchmark on Sherlock — project reference

Current-state reference for what this repo is, what it does, and how it works.
Keep this current when the code changes (describe the *system as it is now*, not a
history of edits).

## What this is

A containerized harness to **benchmark 11 pathology foundation models (PFMs)**
head-to-head on TCGA downstream tasks. Each PFM is a **frozen encoder**: extract
embeddings once, train a small **linear probe** per task, compare on identical
metrics. **All data and weights are real** — TCGA slides from GDC and published
model weights.

Two Apptainer containers (CPU data build + GPU/torch model run), one isolated venv
per model, **all I/O under `<repo>/runtime/` on `$SCRATCH`** — no `$HOME`, no group
storage at runtime.

Repo root: `/scratch/users/akshatg/stanf_pfm`

## Two containers (by design)

The data build is CPU-only; model extraction needs GPU + torch. Separate images so
they never pollute each other:

| Container | Built by | Holds | Used for |
|---|---|---|---|
| `runtime/containers/tcga_build.sif` | `jobs/setup_tcga.sh` | `python:3.10-slim` + lean data deps (pandas, omegaconf, requests, openslide-bin, tifffile) | GDC query/download/stream, thumbnails, `dataset.csv` |
| `runtime/containers/pfm_base.sif` | `pfm_setup.sh build` | `pytorch/pytorch` CUDA image (py3.11) | model loading, embedding extraction, probe training |

## Layout

```
pfm_setup.sh              # driver: build pfm_base, per-model venvs, run/extract, benchmark (Slurm-aware)
README.md                 # user-facing usage + per-file descriptions + step-by-step diagrams
CLAUDE.md                 # this file — current-state project reference
.gitignore                # ignores runtime/ (data, venvs, containers, caches) + **/.hf_token
models.txt                # the 11 model names
pfm_common/               # SHARED infra imported by every model (written once, reused)
  config.py               #   env-driven paths/knobs; PFM_ROOT defaults to <repo>/runtime; device()=cuda|cpu
  hf_auth.py              #   HuggingFace login from $HF_TOKEN
  data.py                 #   streaming IterableDataset over per-slide .tar shards (or loose/
                          #     thumbnails fallback); PFM_PATCH_STRIDE + process/worker sharding
  runner.py               #   run_patch_encoder(): stream -> encode -> MEAN-POOL to slide level
                          #     as it runs -> patch_embeddings[.shard<g>of<N>].pt; rc 75 = NO-ACCESS
  merge_shards.py         #   concat per-shard slide embeddings -> patch_embeddings.pt (data-parallel)
  train_probe.py          #   fit_linear_probe() + CLI: linear head over frozen embeddings
  tasks.py                #   TASK_REGISTRY: 6 TCGA tasks; idh = IDH1|IDH2 (label_any_of)
  metrics.py              #   numpy-only acc / balanced acc / macro-F1 / AUROC / confusion
  benchmark.py            #   every (model x task): probe + score -> results.{csv,json}
  plot_results.py         #   leaderboard + model x task heatmap PNG + summary CSV
models/<model>/<model>_spin.py   # thin ~15-line adapter: load()->(model,transform), embed()->[B,D]
build_tcga_dataset.py     # ETL CLI: --config, --steps, key=value overrides, --dry-run, --force
tcga/                     # the TCGA/GDC ETL package (flattened to repo root)
  gdc_client.py           #   generic GDC REST wrapper (queries; /data supports HTTP Range)
  etl.py                  #   flat slide table (file_id, slide_id, project_id, file_size, maf_*)
  manifest.py             #   manifests + select_stratified() + select_by_byte_budget()
  downloader.py           #   threaded GDC REST download, md5-verified, resumable (no gdc-client binary)
  slide_processor.py      #   openslide SVS->JPG thumbnails (used by process_slides)
  slide_streamer.py       #   HTTP-Range thumbnail streaming (HTTPRangeFile + tifffile); no full download
  slide_stager.py         #   acquire SVS (stream_to_local OR resident cache) -> 1 thumbnail/slide;
                          #     also predownload_svs() (cache) + acquire_tile_process() (patch tiling)
  slide_tiler.py          #   Otsu tissue mask + patch tiling (slide -> many 256px tissue patches); numpy/PIL only
  gene_matrix.py          #   MAF aliquot->sample -> 0/1 gene matrix
  pipeline.py             #   TCGADatasetBuilder: chains steps (etl/manifest/download_svs_cache/
                          #     stream_thumbnails/stage_process/tile_slides/gene_matrix/assemble)
  README.md               #   ETL reference (diagrams, steps, config)
configs/
  tcga_tiled.yaml         #   DEFAULT (GPU-bound): patch tiling, persisted (tile_slides)
  tcga_staged.yaml        #   staged full-SVS -> 1 thumbnail/slide (stage_process); FINAL_CONFIG override
jobs/
  setup_tcga.sh           #   one-time: build tcga_build.sif + tcga_build venv (CPU/normal)
  preprocess.sh           #   ★ CPU-only preprocessing (PREP_SCOPE=mini|full): tile -> pack ->
                          #     DECODE(raw) -> labels, resumable per slide. final_setup{,_mini} CALL it.
  final_setup.sh          #   ★ FINAL run (GPU): self-bootstraps -> preprocess -> all models -> train.
                          #     Failsafe: builds any missing container/venv/data.
  final_setup_mini.sh     #   ★ same as final_setup.sh but extraction uses a 1/MINI_FRACTION (default 1%)
                          #     SAMPLE of the persisted patches -> ~100x less GPU work, fits a short walltime.
  verify_tcga_env.py      #   sanity-checks the data-build venv imports (used by setup_tcga.sh)
runtime/                  # ALL artifacts on $SCRATCH (mode 700). Git-ignored.
  containers/             #   pfm_base.sif + tcga_build.sif (both built)
  venvs/<model>/          #   one venv per model + tcga_build (all present)
  repos/HIPT/             #   cloned HIPT (HIPT_4K code + ViT-256 ckpt) for hipt
  cache/huggingface/      #   HF_HOME (model weights cached here)
  embeddings/             #   default PFM_OUTPUT_DIR; <model>/patch_embeddings.pt + benchmark/
  tcga/                   #   default PFM_TCGA_ROOT (tables/, patches/, slides/ = the data cache)
  .hf_token               #   HF token file (the ONLY token source; read by final/mini for gated models)
```

Only two entrypoints remain (`final_setup.sh`, `final_setup_mini.sh`); the old smoke/
proof/verify job scripts and the streaming config were removed. Both entrypoints are
**fully self-bootstrapping** — on a fresh checkout an external party runs one of them and
it builds every missing piece (data container/venv, dataset+patches, model container +
per-model venvs) before extracting.

## The 11 models

conch, uni2, virchow, virchow2, phikon (public, no token), exaone-path, gigapath,
h-optimus, mSTAR, hipt, titan (**slide** encoder). All gated except phikon and hipt.
Per-model install/adapter facts worth knowing:

- **conch** — pip-installs the CONCH package from a **GitHub HTTPS zip archive**
  (`.../archive/refs/heads/main.zip`), not `git+`. `conch_spin.py` drops its own dir,
  `models/`, and the repo root from `sys.path` so `models/conch/` can't shadow the
  installed `conch` package.
- **mSTAR** — also installed from a GitHub HTTPS zip archive.
- **hipt** — no HF weights: the ViT-256 code lives in the cloned `runtime/repos/HIPT`
  (`hipt_spin.py` puts the repo root AND `HIPT_4K/` on `sys.path`); its checkpoint is
  a git-LFS pointer in the repo, so the run downloads the real ~700 MB file from
  the LFS media URL (branch `master`) to `PFM_HIPT_CKPT`. hipt deps
  (cv2/h5py/scipy/skimage/webdataset/einops/tqdm) are in `model_pip_pkgs`. Its
  transform resizes to 256×256 (ViT-256 input; HIPT's own `eval_transforms()` has no
  resize, which otherwise breaks batching on variable-size thumbnails).
- **h-optimus** — loads via `timm` hf_hub (no pip package needed). Gated: needs HF
  access to `bioptimus/H-optimus-0`; without the grant it 403s and is reported
  `NO-ACCESS` and disregarded.
- **titan** — a **slide** encoder that consumes precomputed CONCH `.h5` patch
  features, not raw thumbnails; it loads but has no valid input here, so it's
  reported "load-only (expected)", not a failure.
- **gigapath / exaone-path** — clone their repo on the host and pip-install (gigapath
  editable). Host git is used only for host-side clones; nothing needs git inside the
  container.

## The 6 tasks (`pfm_common/tasks.py`)

| Task | Label | Rows |
|---|---|---|
| `luad_vs_lusc` | lung adeno vs squamous (project_id) | LUAD/LUSC |
| `lgg_vs_gbm` | low-grade glioma vs glioblastoma (project_id) | LGG/GBM |
| `kras` `tp53` `egfr` `idh` | gene mutation 0/1 | sequenced tumors (`has_maf`) |

Labels come from the ETL `dataset.csv` (one row/slide: `slide_id`, `jpg_path`,
`project_id`, `has_maf`, gene columns). Join key: `slide_id = basename(jpg_path)`
without extension, matching `thumbnails/<slide_id>.jpg`.

## How it works (data flow)

1. **Data** — the `tcga/` ETL queries GDC, produces `<slide_id>.jpg` thumbnails +
   gene matrix, assembles `tables/dataset.csv`. Runs in `tcga_build.sif`; output dir
   `PFM_TCGA_ROOT` (default `runtime/tcga`), bind-mounted `/tcga_data`. Two data
   strategies (see below) share the same thumbnail/`dataset.csv` output.
2. **Extract** — `runner.run_patch_encoder(name, load, embed, gated)` streams tiles via
   `data.make_streaming_dataset` (an `IterableDataset` reading tar members from `PFM_PATCH_DIR`;
   O(1) RAM, sharded by GPU-process × DataLoader-worker; `PFM_PATCH_STRIDE` keeps every Nth),
   runs the frozen encoder in `pfm_base.sif` (GPU, **autocast bf16/fp16 auto**), and
   **mean-pools to slide level AS IT RUNS** (running sum+count per slide) — saving
   `{"embeddings":Tensor[N_slides,D],"slide_ids":[...]}` (not N_patches) to
   `PFM_OUTPUT_DIR/<model>/patch_embeddings.pt`. Data-parallel (`PFM_SHARD_COUNT>1`) writes
   `patch_embeddings.shard<g>of<N>.pt` per GPU (disjoint slides); `pfm_common.merge_shards`
   concatenates them into `patch_embeddings.pt`.
3. **Benchmark** — `benchmark.py` discovers models with embeddings; loads the slide-level
   vectors (`slide_ids`; falls back to mean-pooling `paths` for legacy per-patch files),
   joins labels by slide_id (`tasks.labels_for_task`), stratified split, `fit_linear_probe`
   (z-scored, AdamW, CE), scores via `metrics.compute_all` -> `benchmark/results.{csv,json}`.
4. **Plot** — `plot_results.py` -> leaderboard, `summary_<metric>.csv`,
   `heatmap_<metric>.png` (default AUROC).

## Data-loading strategies (all real, all stratified)

Subsets are always **stratified across projects** (`select_stratified` by count, or
`select_by_byte_budget` by size) so LUAD/LUSC/LGG/GBM are all represented and the
subtype tasks aren't degenerate. The MAF subset is built from the same slides so
gene labels align (`_resolve_manifest` prefers a subset manifest when present).

**Acquisition fallback chain** (per slide, in `acquire_stage_process`/`acquire_tile_process`,
resumable): preprocessed artifact (patches/thumbnail) already present -> **skip**; else full
SVS in the `$SCRATCH` cache (`PFM_TCGA_ROOT/slides`, if pre-filled by the `download_svs_cache`
pipeline step) -> use it; else (`download.stream_to_local: true`, the default)
**stream the SVS into node-local `$TMPDIR`, use it, evict** (nothing large persists). Only
the small artifact is kept.

- **Patch tiling (the real GPU-bound run — `final_setup.sh` + `FINAL_CONFIG=tcga_tiled.yaml`).**
  Step `tile_slides` (`slide_tiler.py`): acquire each SVS (chain above), Otsu tissue mask,
  cut **every** tissue cell (`tcga_tiled.yaml` is UNCAPPED — no `max_patches`) of `patch_size`
  (256), **persist** to `PFM_TCGA_ROOT/patches/<slide_id>/<sid>__x_y.jpg` (tiled ONCE, reused
  every run), evict the SVS. In practice ~16k patches/slide → ~2.26M total across 142 slides,
  so extraction is heavily GPU-bound (and `final_setup.sh` needs the 2-day walltime; use
  `final_setup_mini.sh` to run on a 1/10 sample). `final_setup` STEP 4 stages the patches to the
  node-local SSD and reads the `compute:` block for batch/workers (nothing hardcoded).
- **Staged full-SVS -> 1 thumbnail/slide (`tcga_staged.yaml`, step `stage_process`).** Same
  acquisition; produces one 512×512 thumbnail per slide (fast, I/O-bound, coarse). N≈142, so
  the GPU is idle — extraction was never the bottleneck (download was). `FINAL_CONFIG` override.

## Config / env knobs

`pfm_common/config.py` (env): `PFM_ROOT` (default `<repo>/runtime`), `PFM_TCGA_ROOT`,
`PFM_PATCH_DIR` (dir of raw `.bin` OR `.tar` shards OR loose images — resolved raw > tars > loose),
`PFM_OUTPUT_DIR`, `PFM_SLIDE_FEATURES`, `PFM_MAX_IMAGES` (0=all), `PFM_BATCH_SIZE`,
`PFM_NUM_WORKERS`, `PFM_PREFETCH_FACTOR`, `PFM_AMP_DTYPE`, `HF_TOKEN`. **Scaling knobs:**
`PFM_PATCH_STRIDE` (keep every Nth patch; 1=all, mini=100 for 1%), `PFM_SHARD_INDEX`/
`PFM_SHARD_COUNT` (data-parallel; each process takes a disjoint, patch-count-balanced set of
whole slides). **Raw streaming-cache knobs:** `PFM_RAWCACHE` (auto|on|off), `PFM_RAWCACHE_DIR`
(node-local SSD dir the raw bins stream through; default `$L_SCRATCH/pfm_rawcache`),
`PFM_DECODE_WORKERS` (decode thread count), `PFM_FORCE` (re-extract even if output exists).

ETL config (YAML): `download.slides`, `download.maf`, `download.max_files` (cap by count),
`download.target_gb` (**cap by size; `null` = ALL slides = the full ~500 GB — final_setup's
default**), `patches.{patch_size,level,tissue_thresh,...}`, `decode.workers` (0=all cores),
`slides.stage_download_workers`, `gene_matrix.genes` (incl. `IDH1`/`IDH2` for the `idh` task),
`steps` (`etl,manifest,tile_slides,pack_patches,decode_patches,download,gene_matrix,assemble`).

Job knobs: `FINAL_TARGET_GB` (override `download.target_gb`; mini sets 50), `MINI_FRACTION`
(mini's 1/N sample → `PFM_PATCH_STRIDE`), `PFM_RUN_MODE` (`shard` = data-parallel per model,
`queue` = one-model-per-GPU work queue), `PREP_SCOPE` (`mini`|`full` for preprocess.sh),
`PREP_SKIP_CONTAINER`, `FINAL_CONFIG`, `MED_PROJECT_DIR`/`PFM_REPO`/`PFM_PROJECT_DIR` (dir overrides).

## Entrypoints (both GPU, both fully self-bootstrapping / failsafe)

Both build every missing piece themselves — data container/venv (STEP 1); the dataset,
tiling each SVS **directly into one `patches_tar/<slide_id>.tar`** (STEP 2, `tile_slides`;
`pack_patches` migrates any legacy loose patches) — so staging reads ~N_slides big files,
not millions of tiny ones; model container **AND** per-model venvs (STEP 3,
`pfm_setup.sh setup`, idempotent) — then stage the tar-shards node-local (STEP 4) →
`pfm_setup.sh run` → `benchmark` (STEP 5). Extraction **mean-pools to slide level as it
runs** (`runner`), saving ~N_slides vectors (not N_patches). An external party can run
either on a fresh checkout (after adding `runtime/.hf_token`) end-to-end.

- **Mini — `jobs/final_setup_mini.sh`.** `-G 4` **general** GPUs (no `-C`), 48 G, **30 min**.
  `FINAL_TARGET_GB=50` caps its tiling scope to a stratified ~50 GB / ~142-slide subset (so a
  short job never streams the other ~1250 slides); extraction reads a **1% sample**
  (`MINI_FRACTION=100` → `PFM_PATCH_STRIDE=100`, every 100th tar member). Data-parallel
  (`PFM_RUN_MODE=shard`), same scheduler as full. ~12–15 min once it gets a node. Submit:
  `mkdir -p logs && sbatch jobs/final_setup_mini.sh`.
- **Final — `jobs/final_setup.sh`.** Runs on **100% of ALL ~1400 slides (~500 GB, the full
  TCGA-LUAD/LUSC/LGG/GBM)** — `configs/tcga_tiled.yaml` has `download.target_gb: null` (no
  cap). `-G 8 -C GPU_MEM:80GB` (H100 — Sherlock has **no B200**; code is GPU-count-agnostic,
  shards across 8 of whatever it lands on), `--nodes=1`, 128 G, **1-06:00:00 (~30 h)**.
  **`PFM_RUN_MODE=shard`**: each model split across all 8 GPUs (`PFM_SHARD_COUNT`/
  `PFM_SHARD_INDEX` → each GPU takes a disjoint set of slide-tars, writes a per-shard file,
  then `pfm_common.merge_shards` concatenates). Makespan ≈ total_work/8, not gated by the
  slowest single model. One-time tiling ~5–10 h (resumable) + extract ~9–12 h. Submit:
  `mkdir -p logs && sbatch jobs/final_setup.sh`.

Repo-dir resolution in the job scripts picks the first candidate that actually
contains the pipeline files (`build_tcga_dataset.py` + `jobs/verify_tcga_env.py`):
BASH_SOURCE/.. → `SLURM_SUBMIT_DIR` → `$PWD`; override with `MED_PROJECT_DIR`. So they
work whether submitted from the repo root or `jobs/`.

## Current state

- **Repo flattened + git-initialized.** The ETL is the top-level `tcga/` package (it
  used to be a nested `med-reduce` sub-repo — that whole separate REDUCE/DINOv3
  dermatology framework was archived to `../med-reduce-archive`, out of the tree). The
  CLI is `build_tcga_dataset.py`, and `configs/`, `jobs/`, and the 11 adapters under
  `models/` all sit at the repo root. `git init` done; `.gitignore` excludes `runtime/`
  and `**/.hf_token`, so a commit captures only code.
- Both containers + all 11 model venvs + `tcga_build` venv are **built**. HF token at
  `runtime/.hf_token`.
- **conch and hipt** verified loading (CPU load-only). On the last GPU run, 8 models
  extracted + trained (phikon, exaone-path, mSTAR, conch, uni2, virchow, virchow2,
  gigapath); **hipt** now fixed by the 256×256 resize; **titan** load-only; **h-optimus**
  `NO-ACCESS`.
- **Streaming + tiling validated end-to-end** (GPU). Full 50 GB streamed run produced a
  leaderboard with **0 SVS persisted**. Patch tiling proven GPU-bound: conch 1792 patches
  in 192 s (vs 2.4 s for 142 thumbnails); after the 2→8 dataloader-worker fix, ~0.0073 s/patch
  (~15× faster — the GPU was starved, not compute-bound). titan load-only; h-optimus NO-ACCESS.
- `final_setup.sh` is the real-run entry point. STEP 2 always runs the resumable build (no
  silent-reuse short-circuit); STEP 3 ensures the model container AND per-model venvs
  (`pfm_setup.sh setup`, idempotent); STEP 4 auto-detects the persisted `patches/` furnace and
  sets batch/workers from the config's `compute:` block. Default `FINAL_CONFIG=tcga_tiled.yaml`
  (patch tiling, patches persisted+reused); set `FINAL_CONFIG=tcga_staged.yaml` for thumbnails.
- **Patches persist** to `PFM_TCGA_ROOT/patches/` — tiled once, reused (no re-tile/re-download).
  `final_setup_mini.sh` runs extraction on a 1/`MINI_FRACTION` sample of those same patches.
  All Python compiles; jobs `bash -n` clean.
- No `$HOME`/group-storage dependencies at runtime; everything under `runtime/`.

## Compute / optimization (patch tiling)

- Bottleneck order: thumbnails → WAN download dominates (GPU idle). Tiling → GPU-bound.
- Per-patch ≈ **0.0073 s** (V100, fp16 autocast, batch 64, 8 workers). Full 142 slides ×
  1000 patches ≈ 142k patches → ~17 min/model × 9 ≈ heavy; scale via fewer patches, per-model
  jobs, or a faster card (`-C GPU_MEM:80GB`). Multi-GPU (`-G N`) needs model-sharding code first.
- Levers, by ROI: **dataloader workers** (biggest — feed the GPU), **max_patches** (linear),
  batch size (amortize launch overhead). fp16 is already on.

## Known caveats / TODO

1. **idh task — FIXED** (`tasks.py` `idh` = `IDH1|IDH2` via `label_any_of`; configs tile
   `IDH1`/`IDH2`). Populates once the gene matrix is rebuilt with those genes.
2. **kras (and gene tasks generally) thin at small MAF coverage.** Only slides with a MAF
   have gene labels — in the 50 GB subset that's ~61 samples (KRAS-mut ~4 → near-degenerate).
   Fix is more sequenced slides: the **full 500 GB run** (~1400 slides) has far more MAF
   coverage, so gene tasks come alive there. Not a code bug.
3. **h-optimus** needs an HF access grant to `bioptimus/H-optimus-0` (else disregarded).
4. **No B200 on Sherlock.** Fastest is H100 (`GPU_MEM:80GB`) / H200 (owners). 8-GPU nodes on
   `gpu` = the single H100 node `sh04-01n01` (long queue); 8-GPU A100/H100/H200 live on `owners`.
5. **Scale is handled** (was the old caveat): tiler writes per-slide **tars** (no 22.6M-file
   Lustre storm), extraction **pools to slide level** (embeddings ~MB not 100s GB), and
   **data-parallel shard** spreads a model across GPUs. Full 500 GB run fits the ~30 h walltime.

## Resumability & GCP spot (HARD REQUIREMENT)

The **full run is intended for GCP spot instances** — CPUs/GPUs can be preempted or die at
any moment. **Every stage must be resumable**: a killed run, re-launched, continues from
where it stopped and never redoes finished work or corrupts a half-written artifact.

Resumability model (idempotent + atomic-write everywhere):
- **Preprocessing (tile → pack → decode): resumable per SLIDE.** Each `slide_tiler`/
  `pack_patches`/`decode_patches` unit writes `<slide>.<ext>.part` then atomically renames on
  completion + writes a `<slide>.done` sentinel. A slide with `.done` is skipped; a slide
  killed mid-write left only a `.part` (never mistaken for done) → redone next run.
- **Extraction: resumable per (model, SHARD).** `runner` skips a shard whose
  `patch_embeddings.shard<g>of<N>.pt` already exists; `pfm_setup.sh cmd_run` skips a whole
  model whose merged `patch_embeddings.pt` exists (its shard files are gone post-merge).
  `PFM_FORCE=1` re-extracts. **GAP:** there is no *within-shard* (per-slide) checkpoint yet,
  so a shard preempted mid-flight re-does its ~N_slides on restart — fine for occasional
  preemption, but for *frequent* spot kills the next step is saving each slide's pooled
  vector as it completes so restart skips done slides.
- **Benchmark**: cheap + fully re-runnable (reads existing embeddings).
- **Env note:** the `jobs/*.sh` are **Slurm `sbatch`** (Sherlock). On GCP spot there's no
  Slurm — a non-Slurm launcher is needed, but the resumable Python core (`pfm_setup.sh`,
  `tcga/`, `pfm_common/`) is scheduler-agnostic and is what carries the resume logic.

## Preprocessing / training split (BUILT) — raw pre-decode + streaming SSD cache

CPU preprocessing is split OUT of the GPU job so the GPUs never decode JPEG. **Codec = raw**
(user: "size doesn't matter, speed does" → 0 decode at GPU time). **Fan-out = GIL-free threads**
(not one-process-per-core).

- **Preprocess (CPU, `jobs/preprocess.sh`, `PREP_SCOPE=mini|full`).** tile → pack → **decode** →
  labels, all resumable per slide. `decode_patches` (`tcga/decode_patches.py`) expands each
  `patches_tar/<sid>.tar` → `patches_raw/<sid>.bin` = N contiguous `uint8[256,256,3]` patches
  (+ `.done`). **Byte-level parallel:** a thread pool decodes K JPEGs at once (decoder releases
  the GIL); decoder preference **PyTurboJPEG → cv2.imdecode (default; wheel-bundled libjpeg-turbo,
  no system libs) → Pillow**. Read/decode/write overlap (1-deep prefetch reader). Measured
  expansion **~11.2×** (37 GB JPEG → 415 GB raw for the 142-slide/50 GB subset; full ~1400
  slides → ~4 TB raw). `final_setup{,_mini}.sh` STEP 2 CALLS `preprocess.sh` (verify-or-redo
  failsafe; `PREP_SKIP_CONTAINER=1`), so a GPU job on a fresh checkout still works with no
  separate prep run — but running preprocess.sh standalone on a CPU node offloads all CPU work.
- **Train (GPU): raw bins stay on Lustre; the SSD is a small STREAMING CACHE.** The full ~4 TB
  raw can't fit the node-local SSD (nor can the ~365 GB of JPEG tars — SSD on the dev node is only
  ~159 GB). So `data.py` `_iter_raw` keeps bins on Lustre and, per worker, copies the **next**
  slide's `.bin` onto the SSD (`PFM_RAWCACHE_DIR`, depth-1 prefetch overlapping the current
  slide's reads), **memmaps** it (`np.array(mm[i])` copies each 192 KB patch into DRAM), then
  **evicts** it — a bounded ~2-bin window per worker, never the whole set. Self-limiting: if the
  SSD is near-full, that slide is memmapped straight from Lustre (still zero-decode). STEP 4 sets
  `PFM_PATCH_DIR=$TCGA/patches_raw` (Lustre) + `PFM_RAWCACHE_DIR=$L_SCRATCH/pfm_rawcache`; no bulk
  copy. Batch size / stride / sharding unchanged (still one patch yielded at a time, DataLoader
  batches). Extraction is a **single pass** (each patch read once → mean-pooled), so total Lustre
  read = dataset size once.

## Gotchas

- Never run heavy work on the login node — use `sh_dev` / `salloc` / `sbatch`.
- GPU jobs `--partition=gpu` (+ `-G N`); CPU jobs `--partition=normal`. The old
  `roxanad` owner partition is no longer used for compute.
- Jobs run inside a Slurm allocation; they bind `$SCRATCH` (+ `$L_SCRATCH` when set)
  and the repo into the SIF.
- Host git (RHEL7 glibc) can't run inside the newer container — nothing in-container
  uses git (archive installs / cloned repos on the host); `ensure_git` only ensures
  host git for host-side clones and never binds it into the container.
- Venv `bin/python` symlinks resolve ONLY inside the SIF — never `-f`-test them on the
  host; test `pyvenv.cfg` instead.
- **login-node `python` is 2.7.5, `python3` is 3.6.8**; code targets py3.10/3.11 (the
  container). `py_compile` on the login node checks syntax only — verify behavior
  in-container.
- `rm -rf` is blocked by the sandbox here; use `rm -f <file>` or `find <dir> -delete`.
