"""Downstream training over frozen PFM embeddings -- a linear probe.

Pathology foundation models are used as frozen encoders: you extract embeddings
once (runner.run_patch_encoder) and then *train* a small head on a labelled task.
This module is the shared, model-agnostic trainer for that head, so "train on
TCGA" is one command regardless of which PFM produced the features.

`fit_linear_probe` is the reusable core (also called by pfm_common.benchmark to
compare every model across every task). The CLI below probes ONE embeddings file
against ONE labels CSV and reports the core interpretable metrics:

    python -m pfm_common.train_probe \
        --embeddings $PFM_ROOT/embeddings/uni2/patch_embeddings.pt \
        --labels     /path/to/labels.csv          # columns: path,label[,split]

The labels CSV maps each image path (as stored in the embeddings file) to a
class label. Train/val split is random unless a `split` column is provided.
"""
import argparse
import csv
import os

from . import metrics


def _load_embeddings(path):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["embeddings"], blob["paths"], blob.get("model", "?")


def _load_labels(csv_path):
    rows, splits = {}, {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["path"]] = r["label"]
            if r.get("split"):
                splits[r["path"]] = r["split"]
    return rows, splits


def fit_linear_probe(
    X, y, train_idx, val_idx, n_classes,
    epochs=100, lr=1e-3, weight_decay=1e-4, standardize=True, seed=0,
):
    """Train a linear head on frozen embeddings; return predictions on the val split.

    X          : Tensor[N, D] float embeddings
    y          : sequence[int] of length N (class indices)
    train_idx  : indices into X/y used for training
    val_idx    : indices used for evaluation
    standardize: z-score features using train-split statistics (recommended for
                 a fair cross-model comparison -- embedding scales differ wildly).

    Returns (y_true, y_pred, y_prob, head) for the val split, all numpy/torch.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.float()
    y_t = torch.as_tensor(list(y), dtype=torch.long)

    Xtr, Xva = X[train_idx].clone(), X[val_idx].clone()
    if standardize:
        mu = Xtr.mean(0, keepdim=True)
        sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd
    ytr, yva = y_t[train_idx], y_t[val_idx]

    head = nn.Linear(X.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()
    Xtr, ytr, Xva = Xtr.to(dev), ytr.to(dev), Xva.to(dev)

    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        lossf(head(Xtr), ytr).backward()
        opt.step()

    head.eval()
    with torch.inference_mode():
        logits = head(Xva)
        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(1)
    return yva.numpy(), pred.cpu().numpy(), prob.cpu().numpy(), head


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description="Linear probe over frozen PFM embeddings.")
    ap.add_argument("--embeddings", required=True, help="patch_embeddings.pt from runner")
    ap.add_argument("--labels", required=True, help="CSV with columns path,label[,split]")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="where to save the trained head (.pt)")
    args = ap.parse_args(argv)

    X, paths, model_name = _load_embeddings(args.embeddings)
    label_map, split_map = _load_labels(args.labels)

    keep = [i for i, p in enumerate(paths) if p in label_map or os.path.basename(p) in label_map]
    if not keep:
        raise SystemExit(
            "No embedding paths matched the labels CSV. The CSV 'path' column must "
            "match the stored image paths (or their basenames)."
        )

    def lbl(p):
        return label_map.get(p, label_map.get(os.path.basename(p)))

    def spl(p):
        return split_map.get(p, split_map.get(os.path.basename(p), ""))

    classes = sorted({lbl(paths[i]) for i in keep})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[probe] model={model_name}  N={len(keep)}  classes={classes}")

    X = X[keep].float()
    y = [cls_to_idx[lbl(paths[i])] for i in keep]

    have_split = any(spl(paths[i]) for i in keep)
    if have_split:
        tr = [j for j, i in enumerate(keep) if spl(paths[i]) == "train"]
        va = [j for j, i in enumerate(keep) if spl(paths[i]) in ("val", "valid", "test")]
    else:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(keep), generator=g).tolist()
        n_val = max(1, int(len(perm) * args.val_frac))
        va, tr = perm[:n_val], perm[n_val:]

    y_true, y_pred, y_prob, head = fit_linear_probe(
        X, y, tr, va, len(classes), epochs=args.epochs, lr=args.lr, seed=args.seed
    )
    m = metrics.compute_all(y_true, y_pred, y_prob, len(classes))
    print(
        f"[probe] N_val={m['n']}  acc={m['accuracy']:.4f}  "
        f"bal_acc={m['balanced_accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
        f"auroc={m['auroc']:.4f}"
    )

    out = args.out or os.path.join(os.path.dirname(args.embeddings), "linear_probe.pt")
    torch.save({"state_dict": head.state_dict(), "classes": classes, "model": model_name}, out)
    print(f"[probe] saved head -> {out}")


if __name__ == "__main__":
    main()
