"""Interpretable classification metrics for the downstream benchmark.

Implemented in numpy only (no scikit-learn) so the benchmark runs unchanged in
every model venv -- the only deps are numpy + torch, which all venvs already
have. Each function takes integer labels; AUROC additionally takes class
probabilities.

    y_true : (N,) int array of ground-truth class indices
    y_pred : (N,) int array of predicted class indices
    y_prob : (N, C) float array of per-class probabilities (softmax outputs)
"""
import numpy as np


def accuracy(y_true, y_pred):
    """Plain top-1 accuracy."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def per_class_recall(y_true, y_pred, n_classes):
    """Recall for each class (TP / support). NaN for classes with no support."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rec = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = y_true == c
        if mask.any():
            rec[c] = float((y_pred[mask] == c).mean())
    return rec


def balanced_accuracy(y_true, y_pred, n_classes):
    """Mean per-class recall -- robust to class imbalance, unlike plain accuracy."""
    rec = per_class_recall(y_true, y_pred, n_classes)
    rec = rec[~np.isnan(rec)]
    return float(rec.mean()) if len(rec) else float("nan")


def per_class_precision(y_true, y_pred, n_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    prec = np.full(n_classes, np.nan)
    for c in range(n_classes):
        pred_c = y_pred == c
        if pred_c.any():
            prec[c] = float((y_true[pred_c] == c).mean())
        else:
            prec[c] = 0.0  # no predictions for c -> precision conventionally 0
    return prec


def macro_f1(y_true, y_pred, n_classes):
    """Unweighted mean of per-class F1 -- treats every class equally."""
    prec = per_class_precision(y_true, y_pred, n_classes)
    rec = per_class_recall(y_true, y_pred, n_classes)
    f1s = []
    for c in range(n_classes):
        p, r = prec[c], rec[c]
        if np.isnan(r):
            continue  # class absent from ground truth -> skip
        f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else float("nan")


def _binary_auroc(y_true_bin, scores):
    """AUROC via the Mann-Whitney U / rank statistic (handles ties)."""
    y_true_bin = np.asarray(y_true_bin)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y_true_bin == 1).sum())
    n_neg = int((y_true_bin == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # AUROC undefined when only one class is present
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based; average over ties
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true_bin == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc(y_true, y_prob, n_classes):
    """ROC-AUC. Binary: AUC of the positive-class score. Multiclass: macro one-vs-rest."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if n_classes == 2:
        return _binary_auroc((y_true == 1).astype(int), y_prob[:, 1])
    aucs = []
    for c in range(n_classes):
        if (y_true == c).any() and (y_true != c).any():
            aucs.append(_binary_auroc((y_true == c).astype(int), y_prob[:, c]))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def confusion_matrix(y_true, y_pred, n_classes):
    """(C, C) integer matrix; rows = true class, cols = predicted class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_all(y_true, y_pred, y_prob, n_classes):
    """Bundle the core interpretable metrics into one dict (JSON-friendly floats)."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, n_classes),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "auroc": auroc(y_true, y_prob, n_classes),
        "per_class_recall": per_class_recall(y_true, y_pred, n_classes).tolist(),
        "per_class_precision": per_class_precision(y_true, y_pred, n_classes).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes).tolist(),
        "n": int(len(y_true)),
    }
