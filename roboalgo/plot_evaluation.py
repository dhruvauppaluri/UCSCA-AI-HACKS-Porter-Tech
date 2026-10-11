"""Plot target, predicted, and recorded base-frame Z for an evaluated run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory produced by roboalgo/evaluate.py")
    args = parser.parse_args()
    with (args.run_dir / "comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No comparison samples")
    time = [float(r["time_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for column, label, color in (
            ("target_z_mm", "Sent target Z", "#9b8a7b"),
            ("predicted_follower_z_mm", "Predicted follower Z", "#2363a0"),
            ("recorded_follower_z_mm", "Recorded follower Z", "#ce603c")):
        ax.plot(time, [float(r[column]) for r in rows], label=label,
                color=color, linewidth=1.4)
    ax.set(xlabel="Time since recording start (s)",
           ylabel="Estimated base-frame Z (mm)",
           title="Follower Z prediction versus recorded motion")
    ax.grid(alpha=.2)
    ax.legend()
    output = args.run_dir / "z_comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
