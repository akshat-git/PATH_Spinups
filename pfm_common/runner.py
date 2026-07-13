"""The shared extraction loop. A model adapter calls run_patch_encoder() with two
callbacks; everything below (auth, data, batching, autocast, saving) is reused.
"""
import os
import time

from . import config, data, hf_auth

# Exit code the runner uses when a gated model can't be accessed (token missing or
# not approved). Distinct from a normal error so callers can disregard the model.
NO_ACCESS_EXIT = 75


def _resolve_amp(dev):
    """(enabled, dtype) for autocast, driven by config.AMP_DTYPE (the spec) -- not
    hardcoded. 'auto' picks bf16 on GPUs that support it (Ampere+), else fp16 (e.g.
    V100); 'float32'/'off' disables autocast. CPU always uses bf16 autocast."""
    import torch

    sel = str(getattr(config, "AMP_DTYPE", "auto")).lower()
    if dev != "cuda":
        return True, torch.bfloat16
    if sel in ("float32", "fp32", "off", "none", "disable"):
        return False, torch.float32
    if sel in ("bf16", "bfloat16"):
        return True, torch.bfloat16
    if sel in ("fp16", "float16", "half"):
        return True, torch.float16
    # auto: prefer bf16 where the hardware supports it natively, else fp16
    try:
        supports_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        supports_bf16 = False
    return True, (torch.bfloat16 if supports_bf16 else torch.float16)


def _is_access_error(exc):
    """True when an exception looks like a gated / unauthorized HF repo access denial."""
    txt = f"{type(exc).__name__}: {exc}"
    markers = (
        "GatedRepoError", "401", "403", "gated repo", "restricted",
        "authorized list", "Access to model", "awaiting", "must be authenticated",
        "No HF token found", "Cannot access gated repo",
    )
    return any(m in txt for m in markers)


def run_patch_encoder(name, load_fn, embed_fn, gated=False):
    """Run a patch-level foundation model over TCGA patches and save embeddings.

    name      : short model name, used for the output subdirectory.
    load_fn   : () -> (model, transform)  where transform maps PIL.Image -> Tensor[C,H,W].
    embed_fn  : (model, batch_tensor) -> Tensor[B, D]  (batch already on device).
    gated     : if True, require an HF token before attempting the download.
    """
    import torch
    from torch.utils.data import DataLoader

    dev = config.device()
    print(f"[{name}] device={dev} (uses GPU if available, else CPU).", flush=True)
    # Load weights, but treat a gated / unauthorized HF repo as "access not granted"
    # rather than a hard failure: print a clear message and exit 75 so the caller can
    # disregard this model instead of counting it as a bug.
    try:
        hf_auth.login_hf(required=gated)
        print(f"[{name}] loading model ...", flush=True)
        model, transform = load_fn()
    except SystemExit:
        raise
    except Exception as e:
        if _is_access_error(e):
            print(
                f"[{name}] ACCESS NOT GRANTED: gated model and the HF token is missing or "
                f"not approved for it -- disregarding {name}. Request access on its Hugging "
                f"Face page, then re-run.\n[{name}] ({type(e).__name__}: {e})",
                flush=True,
            )
            raise SystemExit(75)
        raise
    model = model.eval().to(dev)
    print(f"[{name}] model ready.", flush=True)

    # Stream tiles: never list/preload the dataset. resolve_patch_root() confirms there is
    # data (pulling just the first shard/path) and picks tar-shard mode (preferred) or the
    # loose-file fallback. The count is a one-time O(1)-memory pass so we log the tile count
    # (after stride) up front. PFM_PATCH_STRIDE keeps every Nth tile (1=all; mini=10).
    mode, root, recursive = data.resolve_patch_root()
    if root is None:
        print(
            f"[{name}] Model + weights loaded OK, but no TCGA patch images were found.\n"
            f"[{name}] Searched:\n{data.describe_sources()}"
            f"[{name}] Stage tiles and set PFM_PATCH_DIR=/path/to/patches, then re-run.",
            flush=True,
        )
        return None
    stride = max(1, config.PATCH_STRIDE)
    n_fed = data.count_patch_images(mode, root, recursive, stride=stride)
    src = "tar shards" if mode == "tars" else "loose files"
    print(f"[{name}] streaming {n_fed} tiles from {root} ({src}"
          f"{f', stride 1/{stride}' if stride > 1 else ', no cap'}).", flush=True)

    # Streaming IterableDataset: RAM = bounded prefetch window, independent of N. Every
    # tile is produced by exactly one worker (tar/stride sharded), so all fed tiles are
    # extracted exactly once -- no duplicates, no drops (no drop_last either).
    ds = data.make_streaming_dataset(transform, stride=stride)
    dl_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        collate_fn=data.collate,
        pin_memory=(dev == "cuda"),
    )
    if config.NUM_WORKERS > 0 and config.PREFETCH_FACTOR > 0:
        dl_kwargs["prefetch_factor"] = config.PREFETCH_FACTOR   # stage batches ahead (H2D overlap)
    loader = DataLoader(ds, **dl_kwargs)

    # Mean-pool to SLIDE level DURING extraction: keep a running sum + count per slide
    # instead of retaining every patch vector. The benchmark trains one sample per slide
    # (mean of its patches) anyway, so this is mathematically identical -- but the saved
    # file is ~N_slides vectors (~MB) not N_patches (100s of GB at scale), so it never
    # OOMs the benchmark's torch.load. slide_id = basename before the "__x_y" patch suffix.
    from collections import OrderedDict

    def _sid(p):
        return os.path.splitext(os.path.basename(p))[0].split("__", 1)[0]

    sums = OrderedDict()      # sid -> running sum Tensor[D] (float32, CPU)
    counts = OrderedDict()    # sid -> int
    n_seen = 0
    amp_enabled, amp_dtype = _resolve_amp(dev)
    print(f"[{name}] batch={config.BATCH_SIZE} workers={config.NUM_WORKERS} "
          f"autocast={'off' if not amp_enabled else amp_dtype}", flush=True)
    t0 = time.time()
    for bi, (batch, paths) in enumerate(loader):
        batch = batch.to(dev, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=amp_enabled):
                out = embed_fn(model, batch)
        out = out.float().cpu()                      # [B, D]
        for i, p in enumerate(paths):
            sid = _sid(p)
            if sid in sums:
                sums[sid] += out[i]
                counts[sid] += 1
            else:
                sums[sid] = out[i].clone()
                counts[sid] = 1
        n_seen += len(paths)
        if bi % 10 == 0:
            print(f"[{name}] batch {bi+1} ({n_seen} patches, {len(sums)} slides)", flush=True)

    if not sums:
        print(f"[{name}] no patches produced -- nothing to save.", flush=True)
        return None
    slide_ids = list(sums.keys())
    embeddings = torch.stack([sums[s] / counts[s] for s in slide_ids], 0)   # [N_slides, D]
    out_dir = config.output_dir_for(name)
    # Data-parallel: each GPU process wrote a DISJOINT set of slides, so save a per-shard
    # file; pfm_common.merge_shards concatenates them into patch_embeddings.pt. Single-shard
    # (SHARD_COUNT==1) writes the final file directly.
    if config.SHARD_COUNT > 1:
        out_path = os.path.join(out_dir, f"patch_embeddings.shard{config.SHARD_INDEX}of{config.SHARD_COUNT}.pt")
        tag = f" [shard {config.SHARD_INDEX}/{config.SHARD_COUNT}]"
    else:
        out_path = os.path.join(out_dir, "patch_embeddings.pt")
        tag = ""
    torch.save({"model": name, "embeddings": embeddings, "slide_ids": slide_ids,
                "patch_counts": [counts[s] for s in slide_ids], "n_patches": n_seen,
                "shard_index": config.SHARD_INDEX, "shard_count": config.SHARD_COUNT}, out_path)
    print(
        f"[{name}]{tag} saved slide-level embeddings {tuple(embeddings.shape)} "
        f"({len(slide_ids)} slides, mean-pooled from {n_seen} patches) -> {out_path} "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    return out_path
