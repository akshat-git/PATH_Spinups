"""Benchmark every PFM against every downstream task.

This is the head-to-head comparison: take the frozen embeddings each pathology
foundation model produced (pfm_common.runner), attach TCGA task labels (from the
ETL dataset.csv via pfm_common.tasks), train one linear probe per (model, task),
and score them all with the same interpretable metrics so the numbers are
directly comparable across models.

    python -m pfm_common.benchmark \
        --dataset-csv  /path/to/tcga/tables/dataset.csv \
        --tasks        luad_vs_lusc tp53 kras          # default: all tasks
        --models       uni2 virchow phikon             # default: all found

Embeddings are discovered under $PFM_OUTPUT_DIR/<model>/patch_embeddings.pt
(produced by `./pfm_setup.sh run <model>`). The join key between an embedding
and a label is the slide_id = basename(image_path) without extension, matching
the `<slide_id>.jpg` thumbnails the TCGA ETL writes.

Outputs (under --out-dir, default $PFM_OUTPUT_DIR/benchmark):
    results.csv   one row per (model, task) with the core metrics
    results.json  same data plus per-class metrics and confusion matrices

Then visualise/compare:  python -m pfm_common.plot_results --results <results.csv>
"""
import argparse
import csv
import json
import os

from . import config, metrics, tasks


def _slide_id(path):
    """<slide_id>.jpg (thumbnail) or <slide_id>__x<X>_y<Y>.jpg (tiled patch) ->
    <slide_id>. The `__` splits the patch provenance suffix off the ETL join key."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split("__", 1)[0]


def discover_models(output_dir):
    """Model names that have an extracted patch_embeddings.pt under output_dir."""
    found = []
    if not os.path.isdir(output_dir):
        return found
    for name in sorted(os.listdir(output_dir)):
        if os.path.isfile(os.path.join(output_dir, name, "patch_embeddings.pt")):
            found.append(name)
    return found


def _stratified_split(y, val_frac, seed):
    """Per-class shuffle so train and val keep the same class balance."""
    import numpy as np

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
        va.extend(idx[:n_val].tolist())
        tr.extend(idx[n_val:].tolist())
    return tr, va


def benchmark_one(emb_path, labels, n_classes, val_frac, epochs, lr, seed, min_samples):
    """Probe a single model's embeddings on a single task. Returns a metrics dict or a reason it was skipped."""
    import torch

    from collections import OrderedDict

    blob = torch.load(emb_path, map_location="cpu", weights_only=False)
    X = blob["embeddings"].float()

    # One vector per slide. New format: extraction already mean-pooled to slide level, so
    # blob["slide_ids"][i] labels X[i] directly. Legacy format: blob["paths"] are per-patch,
    # so mean-pool each slide's patches here. Either way the probe trains one sample/slide
    # (no patch leakage across the train/val split) -- the standard PFM aggregation.
    slide_vecs, y = [], []
    if "slide_ids" in blob:
        for i, sid in enumerate(blob["slide_ids"]):
            lab = labels.get(sid)
            if lab is not None:
                slide_vecs.append(X[i])
                y.append(int(lab))
    else:
        by_slide = OrderedDict()
        for i, p in enumerate(blob["paths"]):
            by_slide.setdefault(_slide_id(p), []).append(i)
        for sid, idxs in by_slide.items():
            lab = labels.get(sid)
            if lab is not None:
                slide_vecs.append(X[idxs].mean(dim=0))
                y.append(int(lab))
    if len(y) < min_samples:
        return {"skipped": f"only {len(y)} labelled slides (< {min_samples})"}
    if len(set(y)) < 2:
        return {"skipped": f"only one class present ({set(y)})"}

    Xk = torch.stack(slide_vecs).float()
    tr, va = _stratified_split(y, val_frac, seed)
    if not va or not tr:
        return {"skipped": "split left train or val empty"}

    from .train_probe import fit_linear_probe

    y_true, y_pred, y_prob, _ = fit_linear_probe(
        Xk, y, tr, va, n_classes, epochs=epochs, lr=lr, seed=seed
    )
    m = metrics.compute_all(y_true, y_pred, y_prob, n_classes)
    m["n_total"] = len(y)
    # total patches pooled for this model (slide-level blob carries n_patches; legacy
    # per-patch blob has "paths"). `paths` no longer exists in the slide-level path.
    m["n_patches"] = int(blob.get("n_patches") or len(blob.get("paths", [])) or len(y))
    m["dim"] = int(Xk.shape[1])
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark all PFMs across all downstream tasks.")
    ap.add_argument("--dataset-csv", default=os.path.join(config.TCGA_ROOT, "tables", "dataset.csv"),
                    help="TCGA ETL dataset.csv (label source). Default: $PFM_TCGA_ROOT/tables/dataset.csv")
    ap.add_argument("--embeddings-dir", default=config.OUTPUT_DIR,
                    help="dir of <model>/patch_embeddings.pt. Default: $PFM_OUTPUT_DIR")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models (default: all found)")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of tasks (default: all)")
    ap.add_argument("--out-dir", default="", help="default: <embeddings-dir>/benchmark")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-samples", type=int, default=10,
                    help="skip a (model, task) cell with fewer labelled embeddings")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.dataset_csv):
        raise SystemExit(
            f"dataset.csv not found: {args.dataset_csv}\n"
            "Build it first with the TCGA ETL pipeline -- see "
            "tcga/README.md (step `assemble`), then pass --dataset-csv."
        )

    models = args.models or discover_models(args.embeddings_dir)
    if not models:
        raise SystemExit(
            f"No embeddings found under {args.embeddings_dir}.\n"
            "Extract them first:  ./pfm_setup.sh run <model>   (writes <model>/patch_embeddings.pt)"
        )
    task_list = args.tasks or tasks.ALL_TASKS
    for t in task_list:
        if t not in tasks.TASK_REGISTRY:
            raise SystemExit(f"Unknown task '{t}'. Choose from: {tasks.ALL_TASKS}")

    out_dir = args.out_dir or os.path.join(args.embeddings_dir, "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[bench] models: {models}")
    print(f"[bench] tasks : {task_list}")
    print(f"[bench] labels: {args.dataset_csv}\n")

    rows = []
    for task in task_list:
        labels, class_names = tasks.labels_for_task(args.dataset_csv, task)
        n_classes = len(class_names)
        if not labels:
            print(f"[bench] task {task}: no labelled rows in dataset.csv -- skipping task.")
            continue
        print(f"[bench] === task {task}  ({len(labels)} labelled slides, classes={class_names}) ===")
        for model in models:
            emb = os.path.join(args.embeddings_dir, model, "patch_embeddings.pt")
            if not os.path.isfile(emb):
                print(f"[bench]   {model:14s} no embeddings -- skip")
                continue
            res = benchmark_one(emb, labels, n_classes, args.val_frac, args.epochs,
                                args.lr, args.seed, args.min_samples)
            if "skipped" in res:
                print(f"[bench]   {model:14s} skipped: {res['skipped']}")
                continue
            print(f"[bench]   {model:14s} n={res['n_total']:5d}  acc={res['accuracy']:.3f}  "
                  f"bal_acc={res['balanced_accuracy']:.3f}  f1={res['macro_f1']:.3f}  "
                  f"auroc={res['auroc']:.3f}")
            rows.append({"model": model, "task": task, "n_classes": n_classes,
                         "class_names": class_names, **res})
        print()

    if not rows:
        raise SystemExit(
            "No (model, task) cell produced a result. Common causes: embeddings and "
            "dataset.csv don't share slide_ids, or every cell was below --min-samples."
        )

    # results.json -- full detail (per-class metrics, confusion matrices)
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # results.csv -- the flat comparison table (scalars only)
    scalar_cols = ["model", "task", "n_classes", "n", "n_total",
                   "accuracy", "balanced_accuracy", "macro_f1", "auroc"]
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scalar_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in scalar_cols})

    print(f"[bench] wrote {len(rows)} (model, task) results")
    print(f"[bench]   table -> {csv_path}")
    print(f"[bench]   detail-> {json_path}")
    print(f"[bench] compare/plot:  python -m pfm_common.plot_results --results {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()
