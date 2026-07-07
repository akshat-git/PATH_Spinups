"""Compare PFMs from benchmark results: a ranked text table + a model x task heatmap.

    python -m pfm_common.plot_results --results $PFM_OUTPUT_DIR/benchmark/results.csv

Reads the flat results.csv from pfm_common.benchmark and emits:
  - a printed comparison matrix (rows = model, cols = task) for the chosen metric
  - a printed leaderboard (mean metric across tasks, best model first)
  - <out>/heatmap_<metric>.png   (model x task heatmap; skipped if no matplotlib)
  - <out>/summary_<metric>.csv   (the same matrix as CSV, always written)

The metric defaults to AUROC (threshold-free, the standard headline for these
binary pathology tasks); pass --metric to switch.
"""
import argparse
import csv
import os
from collections import defaultdict

METRICS = ["auroc", "balanced_accuracy", "accuracy", "macro_f1"]


def _load(results_csv):
    with open(results_csv, newline="") as f:
        return list(csv.DictReader(f))


def _matrix(rows, metric):
    """Return (models, tasks, {(model,task): value})."""
    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    cell = {}
    for r in rows:
        try:
            cell[(r["model"], r["task"])] = float(r[metric])
        except (ValueError, KeyError):
            cell[(r["model"], r["task"])] = float("nan")
    return models, tasks, cell


def _fmt(v):
    return "  -  " if v != v else f"{v:.3f}"  # v!=v -> NaN


def print_table(models, tasks, cell, metric):
    w = max(14, max((len(m) for m in models), default=14) + 1)
    header = " " * w + "".join(f"{t[:11]:>12s}" for t in tasks) + f"{'mean':>12s}"
    print(f"\n=== {metric} : model x task ===")
    print(header)
    print("-" * len(header))
    # rank models by mean metric across available tasks
    means = {}
    for m in models:
        vals = [cell[(m, t)] for t in tasks if cell[(m, t)] == cell[(m, t)]]
        means[m] = sum(vals) / len(vals) if vals else float("nan")
    for m in sorted(models, key=lambda m: (-(means[m] if means[m] == means[m] else -1))):
        line = f"{m:<{w}}" + "".join(f"{_fmt(cell[(m, t)]):>12s}" for t in tasks)
        line += f"{_fmt(means[m]):>12s}"
        print(line)
    print()
    print(f"=== leaderboard (mean {metric} across tasks) ===")
    for rank, m in enumerate(sorted(means, key=lambda m: -(means[m] if means[m] == means[m] else -1)), 1):
        print(f"  {rank:2d}. {m:<16s} {_fmt(means[m])}")
    return means


def write_summary_csv(models, tasks, cell, means, metric, out_dir):
    path = os.path.join(out_dir, f"summary_{metric}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + tasks + ["mean"])
        for m in models:
            w.writerow([m] + [_fmt(cell[(m, t)]).strip() for t in tasks] + [_fmt(means[m]).strip()])
    return path


def plot_heatmap(models, tasks, cell, metric, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); skipping heatmap (summary CSV still written).")
        return None

    import numpy as np
    M = np.array([[cell[(m, t)] for t in tasks] for m in models], dtype=float)
    fig, ax = plt.subplots(figsize=(1.4 * len(tasks) + 3, 0.5 * len(models) + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(tasks)):
            v = M[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.78 else "black", fontsize=8)
    ax.set_title(f"PFM benchmark: {metric}")
    fig.colorbar(im, ax=ax, label=metric, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, f"heatmap_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare PFMs from benchmark results.csv.")
    ap.add_argument("--results", required=True, help="results.csv from pfm_common.benchmark")
    ap.add_argument("--metric", default="auroc", choices=METRICS)
    ap.add_argument("--out-dir", default="", help="default: alongside results.csv")
    args = ap.parse_args(argv)

    rows = _load(args.results)
    if not rows:
        raise SystemExit(f"No rows in {args.results}")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    models, tasks, cell = _matrix(rows, args.metric)
    means = print_table(models, tasks, cell, args.metric)
    summary = write_summary_csv(models, tasks, cell, means, args.metric, out_dir)
    print(f"\n[plot] summary table -> {summary}")
    png = plot_heatmap(models, tasks, cell, args.metric, out_dir)
    if png:
        print(f"[plot] heatmap       -> {png}")


if __name__ == "__main__":
    main()
