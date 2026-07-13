"""Merge per-shard slide-embedding files into one patch_embeddings.pt.

Data-parallel extraction (pfm_setup run mode "shard") splits a model across N GPU
processes; each writes patch_embeddings.shard<g>of<N>.pt covering a DISJOINT set of
slides (tars are partitioned by slide). Because the slide sets don't overlap, merging
is a straight concatenation -- no cross-shard averaging needed.

    python -m pfm_common.merge_shards <model> <shard_count>
"""
import os
import sys

from . import config


def merge(name, shard_count):
    import torch

    out_dir = config.output_dir_for(name)
    embs, slide_ids, patch_counts, n_patches = [], [], [], 0
    shard_paths = []
    for g in range(shard_count):
        p = os.path.join(out_dir, f"patch_embeddings.shard{g}of{shard_count}.pt")
        if not os.path.isfile(p):
            raise SystemExit(f"[merge:{name}] missing shard file {p} -- a shard likely failed; not merging.")
        b = torch.load(p, map_location="cpu", weights_only=False)
        embs.append(b["embeddings"])
        slide_ids.extend(b["slide_ids"])
        patch_counts.extend(b.get("patch_counts", []))
        n_patches += int(b.get("n_patches", 0))
        shard_paths.append(p)

    # Disjoint slides across shards -> a slide_id must not repeat; guard against a bug.
    if len(slide_ids) != len(set(slide_ids)):
        raise SystemExit(f"[merge:{name}] duplicate slide_ids across shards -- tar sharding is not disjoint!")

    embeddings = torch.cat(embs, 0)
    out_path = os.path.join(out_dir, "patch_embeddings.pt")
    torch.save({"model": name, "embeddings": embeddings, "slide_ids": slide_ids,
                "patch_counts": patch_counts, "n_patches": n_patches}, out_path)
    print(f"[merge:{name}] merged {shard_count} shards -> {tuple(embeddings.shape)} "
          f"({len(slide_ids)} slides, {n_patches} patches) -> {out_path}", flush=True)
    for p in shard_paths:                          # tidy up the per-shard files
        try:
            os.remove(p)
        except OSError:
            pass
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m pfm_common.merge_shards <model> <shard_count>")
    merge(sys.argv[1], int(sys.argv[2]))
