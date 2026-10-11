"""Plot the fixed dog drawing with and without fitted lag compensation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "examples"
RUNS = (
    ROOT / "cartoon_dog_mujoco_uncompensated",
    ROOT / "cartoon_dog_mujoco_strict",
)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, directory in zip(axes, RUNS):
        with (directory / "trace.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary = json.loads((directory / "summary.json").read_text())
        desired = np.array([(float(r["desired_x_mm"]), float(r["desired_y_mm"]))
                            for r in rows])
        down = np.array([int(r["pen_down"]) for r in rows], dtype=bool)
        ink = np.array([(float(r["ink_x_m"]) * 1000,
                         float(r["ink_y_m"]) * 1000)
                        for r in rows if int(r["pen_paper_contact"]) and int(r["ink"])])
        ax.plot(np.where(down, desired[:, 0], np.nan),
                np.where(down, desired[:, 1], np.nan),
                c="#a0a0a0", lw=1.8, label="Intended stroke")
        ax.scatter(ink[:, 0], ink[:, 1], s=5, c="#102b95",
                   label="MuJoCo collision ink")
        ax.set_title(f"Lag correction {summary['lag_compensation']}\n"
                     f"Timed XY RMSE: {summary['simulated_pen_down_xy_rmse_mm']:.2f} mm\n"
                     f"3 mm stroke coverage: {summary['collision_stroke_sample_coverage_within_3mm_percent']:.1f}%")
        ax.set_xlim(20, 155)
        ax.set_ylim(45, 200)
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
        ax.set_xlabel("Inset paper x (mm)")
    axes[0].set_ylabel("Inset paper y (mm)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Unseen cartoon dog, same arm calibration and controller\n"
                 "Strict MuJoCo contact only; no new fit from the dog", fontsize=13)
    fig.tight_layout()
    destination = ROOT / "cartoon_dog_model_check.png"
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
