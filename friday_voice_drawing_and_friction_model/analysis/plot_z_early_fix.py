"""Compare the old and motion-gated Z estimates on the 213515 run."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "examples"
OLD = ROOT / "follow_leader_213515" / "predicted_follower.csv"
NEW = ROOT / "z_all_seven" / "held_out_213515_z.csv"
OUTPUT = ROOT / "z_all_seven" / "early_z_comparison.png"


def main():
    with OLD.open(newline="") as handle:
        old_rows = list(csv.DictReader(handle))
    with NEW.open(newline="") as handle:
        new_rows = list(csv.DictReader(handle))
    if len(old_rows) != len(new_rows):
        raise ValueError("Comparison runs have different lengths")
    t = np.array([float(r["time_s"]) for r in old_rows])
    if not np.allclose(t, [float(r["time_s"]) for r in new_rows], atol=1e-6):
        raise ValueError("Comparison times do not align")
    series = (
        ("Recorded follower", [float(r["recorded_follower_z_mm"]) for r in old_rows], "#d15b3f"),
        ("Original predictor", [float(r["predicted_follower_z_mm"]) for r in old_rows], "#2356a5"),
        ("Old time-ramped correction", [float(r["predicted_follower_z_corrected_mm"]) for r in old_rows], "#9b65a8"),
        ("New motion-gated correction", [float(r["predicted_z_mm"]) for r in new_rows], "#34845b"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax in axes:
        for name, values, color in series:
            ax.plot(t, values, label=name, color=color, linewidth=1.6)
        ax.set(xlabel="Time since recording start (s)",
               ylabel="Base-frame Z (mm)")
        ax.grid(alpha=.2)
    axes[0].set(xlim=(0, 5), ylim=(15, 55), title="First five seconds")
    axes[1].set(xlim=(0, t[-1]), title="Full 213515 run")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Follower Z: correction activates after leader motion\n"
                 "New curve fitted on six other runs; final all-seven model saved separately")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
