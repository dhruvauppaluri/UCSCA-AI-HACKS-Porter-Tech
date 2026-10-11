"""Export planned pen-tip waypoints for a MuJoCo drawing controller.

Coordinates are meters in the inset paper frame: origin at its bottom-left
mark, +x toward bottom-right, +y toward top-left, and +z above the paper.
The CSV is a task-space input, not actuator controls or measured ink.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from trajectory_proxy import build_target


FIELDS = ("time_s", "paper_x_m", "paper_y_m", "paper_z_m", "pen_down")


def export(content: str, output: Path, *, sample_rate_hz: float = 30.0,
           draw_speed_mm_s: float = 25.0, travel_speed_mm_s: float = 45.0,
           lift_mm: float = 10.0, press_depth_mm: float = 1.0) -> tuple[int, float]:
    if sample_rate_hz <= 0 or not np.isfinite(sample_rate_hz):
        raise ValueError("sample_rate_hz must be finite and positive")
    time_s, xyz_mm, pen_down = build_target(
        content, sample_period_s=1.0 / sample_rate_hz,
        draw_speed_mm_s=draw_speed_mm_s, travel_speed_mm_s=travel_speed_mm_s,
        lift_mm=lift_mm, press_depth_mm=press_depth_mm,
    )
    if not np.all(np.isfinite(xyz_mm)) or not np.all(np.isfinite(time_s)):
        raise ValueError("trajectory contains non-finite values")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("waypoint times must increase strictly")
    # The inset page is 170 by 235 mm; z is relative to its nominal surface.
    if np.any(xyz_mm[:, 0] < 0) or np.any(xyz_mm[:, 0] > 170) or np.any(xyz_mm[:, 1] < 0) or np.any(xyz_mm[:, 1] > 235):
        raise ValueError("planned path extends beyond the inset paper rectangle")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        for t, (x, y, z), down in zip(time_s, xyz_mm / 1000.0, pen_down):
            writer.writerow((f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", int(down)))
    return len(time_s), float(time_s[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content", help="Text to draw, for example '67 67'")
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV")
    parser.add_argument("--sample-rate-hz", type=float, default=30.0)
    parser.add_argument("--draw-speed-mm-s", type=float, default=25.0)
    parser.add_argument("--travel-speed-mm-s", type=float, default=45.0)
    parser.add_argument("--lift-mm", type=float, default=10.0)
    parser.add_argument("--press-depth-mm", type=float, default=1.0)
    args = parser.parse_args()
    count, duration = export(args.content, args.output, sample_rate_hz=args.sample_rate_hz,
                             draw_speed_mm_s=args.draw_speed_mm_s,
                             travel_speed_mm_s=args.travel_speed_mm_s,
                             lift_mm=args.lift_mm, press_depth_mm=args.press_depth_mm)
    print(f"Saved {count} paper-frame waypoints over {duration:.3f} s to {args.output}")


if __name__ == "__main__":
    main()
