"""Plot held-out Z robustness results from the reproducible stress test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "artifacts"


def main():
    summary = json.loads((ROOT / "stress_summary.json").read_text())["scenarios"]
    with (ROOT / "stress_trials.csv").open(newline="") as handle:
        trials = list(csv.DictReader(handle))
    levels = (0, 0.1, 0.5, 2, 5, 10)
    keys = ["baseline"] + [f"leader_iid_noise_{n:g}mm" for n in levels[1:]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    axes[0].plot(levels, [summary[k]["mean_rmse_mm"] for k in keys], "o-", label="Full run")
    axes[0].plot(levels, [summary[k]["mean_first_2s_rmse_mm"] for k in keys],
                 "o-", label="First 2 s")
    axes[0].set(xlabel="Independent leader XYZ noise σ (mm)",
                ylabel="Mean held-out Z RMSE (mm)",
                title="All seven runs · leader measurement noise")
    axes[0].legend()
    early = []
    for key in keys:
        early.append(np.mean([float(r["first_2s_rmse_mm"]) for r in trials
                              if r["scenario"] == key and "213515" in r["recording"]]))
    axes[1].plot(levels, early, "o-", color="tab:green")
    axes[1].set(xlabel="Independent leader XYZ noise σ (mm)",
                ylabel="First 2 s Z RMSE (mm)",
                title="213515 stationary start · held-out run")
    labels = ["All samples", "50% random", "25% random", "Every 4th", "Every 8th"]
    keep = ["baseline", "random_keep_0.5", "random_keep_0.25",
            "downsample_4x", "downsample_8x"]
    axes[2].bar(np.arange(len(keep)), [summary[k]["mean_rmse_mm"] for k in keep],
                color="tab:blue")
    axes[2].set(xticks=np.arange(len(keep)), xticklabels=labels,
                ylabel="Mean held-out full-run Z RMSE (mm)",
                title="All seven runs · missing samples")
    axes[2].tick_params(axis="x", labelrotation=35)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Z predictor stress test · each run held out of fitting; 30 random trials per condition")
    output = ROOT / "stress_test.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
