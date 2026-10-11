"""Plot intended ink and simulated ink from trajectory_proxy CSV exports."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    mask = np.array([r["contact"] == "1" for r in rows])
    desired = np.array([[float(r[f"desired_{axis}_mm"]) for axis in "xy"] for r in rows])
    ink = np.array([[float(r[f"ink_{axis}_mm"]) for axis in "xy"] for r in rows])
    return mask, desired, ink


def segmented(ax, points, mask, **kwargs):
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        if start is not None and (not active or i == len(mask)-1):
            stop = i if active else i-1
            ax.plot(points[start:stop+1, 0], points[start:stop+1, 1], **kwargs)
            start = None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--lag", type=Path, required=True)
    p.add_argument("--drag", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cases = [("No correction", args.baseline),
             ("Lag correction", args.lag),
             ("Lag + estimated drag correction", args.drag)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (title, path) in zip(axes, cases):
        mask, desired, ink = load(path)
        segmented(ax, desired, mask, color="0.7", linewidth=2.5)
        segmented(ax, ink, mask, color="navy", linewidth=1.3)
        rmse = np.sqrt(np.mean(np.sum((ink[mask]-desired[mask])**2, axis=1)))
        ax.set_title(f"{title}\nmodel ink RMSE {rmse:.1f} mm", fontsize=10)
        ax.set_xlabel("Paper x (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-5, 175)
        ax.set_ylim(-5, 240)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Paper y (mm)")
    fig.suptitle("HELLO 42 · intended strokes (gray) and simulated ink (blue)\n"
                 "Fitted lag across seven recordings; drag is a one-run hypothesis", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
