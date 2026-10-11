"""Exploratory pen-path predictor using the archived trajectory fit.

This is a Cartesian response proxy, not rigid-body dynamics or a MuJoCo model.
It accepts explicitly labelled pen contact; the recording has no reliable contact
sensor from which to infer that state for a new path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "voice"))
from drawing import Page  # noqa: E402


def build_target(content: str, *, sample_period_s: float = 1 / 30,
                 draw_speed_mm_s: float = 25.0, travel_speed_mm_s: float = 45.0,
                 paper_z_mm: float = 0.0, lift_mm: float = 10.0,
                 press_depth_mm: float = 1.0):
    """Return time, XYZ target, and explicit contact for planned text.

    The page is the measured 170 x 235 mm inset rectangle. Its lower-left corner
    is (0, 0) in paper coordinates. The drawing planner receives centimeters.
    """
    page = Page({"x_min": 0, "x_max": 17, "y_min": 0, "y_max": 23.5},
                char_height=3, margin=0.5)
    return build_target_from_strokes(
        page.place(content), sample_period_s=sample_period_s,
        draw_speed_mm_s=draw_speed_mm_s, travel_speed_mm_s=travel_speed_mm_s,
        paper_z_mm=paper_z_mm, lift_mm=lift_mm, press_depth_mm=press_depth_mm)


def build_target_from_strokes(strokes, *, sample_period_s: float = 1 / 30,
                              draw_speed_mm_s: float = 25.0,
                              travel_speed_mm_s: float = 45.0,
                              paper_z_mm: float = 0.0, lift_mm: float = 10.0,
                              press_depth_mm: float = 1.0):
    """Build a follower-only trajectory from already placed paper strokes (cm)."""
    if not strokes or any(not stroke for stroke in strokes):
        raise ValueError("At least one nonempty stroke is required")
    if sample_period_s <= 0 or draw_speed_mm_s <= 0 or travel_speed_mm_s <= 0:
        raise ValueError("Sample period and speeds must be positive")
    if lift_mm <= 0 or press_depth_mm < 0:
        raise ValueError("Lift must be positive and press depth nonnegative")
    times, points, contacts = [0.0], [], []
    current = None

    def segment(destination, speed, contact):
        nonlocal current
        destination = np.asarray(destination, dtype=float)
        if current is None:
            current = destination
            points.append(destination.copy())
            contacts.append(contact)
            return
        distance = float(np.linalg.norm(destination - current))
        steps = max(1, math.ceil(distance / (speed * sample_period_s)))
        origin = current.copy()
        for k in range(1, steps + 1):
            points.append(origin + (destination - origin) * k / steps)
            contacts.append(contact)
            times.append(times[-1] + sample_period_s)
        current = destination

    for stroke in strokes:
        start = np.array([stroke[0][0] * 10, stroke[0][1] * 10, paper_z_mm + lift_mm])
        if current is not None:
            segment([current[0], current[1], paper_z_mm + lift_mm], travel_speed_mm_s, False)
        segment(start, travel_speed_mm_s, False)
        segment([start[0], start[1], paper_z_mm - press_depth_mm],
                travel_speed_mm_s, False)
        for x_cm, y_cm in stroke[1:]:
            segment([x_cm * 10, y_cm * 10, paper_z_mm - press_depth_mm],
                    draw_speed_mm_s, True)
        segment([current[0], current[1], paper_z_mm + lift_mm],
                travel_speed_mm_s, False)
    return np.asarray(times), np.asarray(points), np.asarray(contacts, dtype=bool)


def simulate(time_s, target_xyz_mm, contact, *, delay_ms, tau_ms,
             drag_intercept_mm=0.0, drag_per_depth=0.0,
             paper_z_mm=0.0):
    """Predict lagged XYZ and an empirical ink path from known contact states.

    Lateral drag is an exploratory displacement in the direction opposite motion,
    not a friction force. No drag is applied off-paper.
    """
    t = np.asarray(time_s, dtype=float)
    target = np.asarray(target_xyz_mm, dtype=float)
    contact = np.asarray(contact, dtype=bool)
    if t.ndim != 1 or target.shape != (len(t), 3) or contact.shape != (len(t),):
        raise ValueError("Expected time (N), target (N, 3), contact (N)")
    if len(t) < 2 or np.any(np.diff(t) <= 0) or tau_ms <= 0 or delay_ms < 0:
        raise ValueError("Time must increase; tau must be positive; delay nonnegative")
    delayed = np.column_stack([
        np.interp(t - delay_ms / 1000, t, target[:, axis],
                  left=target[0, axis], right=target[-1, axis])
        for axis in range(3)
    ])
    follower = np.empty_like(target)
    follower[0] = target[0]
    for i in range(1, len(t)):
        alpha = 1 - math.exp(-(t[i] - t[i - 1]) / (tau_ms / 1000))
        follower[i] = follower[i - 1] + alpha * (delayed[i] - follower[i - 1])
    ink = follower.copy()
    velocity = np.gradient(follower[:, :2], t, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    depth = np.maximum(0, paper_z_mm - follower[:, 2])
    displacement = np.maximum(0, drag_intercept_mm + drag_per_depth * depth)
    moving = contact & (speed > 1e-6)
    ink[moving, :2] -= displacement[moving, None] * velocity[moving] / speed[moving, None]
    ink[contact, 2] = paper_z_mm
    return follower, ink


def compensate_command(time_s, desired_xyz_mm, *, delay_ms, tau_ms,
                       xy_correction_limit_mm=5.0, z_correction_limit_mm=2.0):
    """Bounded inverse of the fitted lag model, for simulated feedforward only.

    The delayed look-ahead plus velocity term is the continuous-time inverse of
    a first-order low-pass response. Bounds limit corner and lift overshoot.
    """
    t = np.asarray(time_s, dtype=float)
    desired = np.asarray(desired_xyz_mm, dtype=float)
    if desired.shape != (len(t), 3) or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Expected increasing time and XYZ targets")
    future = t + delay_ms / 1000
    position = np.column_stack([np.interp(future, t, desired[:, i]) for i in range(3)])
    velocity = np.gradient(desired, t, axis=0)
    future_velocity = np.column_stack([np.interp(future, t, velocity[:, i]) for i in range(3)])
    command = position + (tau_ms / 1000) * future_velocity
    correction = command - desired
    horizontal_norm = np.linalg.norm(correction[:, :2], axis=1)
    scale = np.minimum(1, xy_correction_limit_mm / np.maximum(horizontal_norm, 1e-12))
    correction[:, :2] *= scale[:, None]
    correction[:, 2] = np.clip(correction[:, 2], -z_correction_limit_mm, z_correction_limit_mm)
    return desired + correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content", help="Letters, digits, words, or smiley")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--press-depth-mm", type=float, default=1.0)
    parser.add_argument("--drag", action="store_true", help="Apply the unvalidated empirical drag fit")
    parser.add_argument("--compensate", action="store_true", help="Simulate bounded lag compensation")
    parser.add_argument("--compensate-drag", action="store_true", help="Also counter the empirical drag hypothesis")
    args = parser.parse_args()
    if args.compensate_drag and (not args.compensate or not args.drag):
        parser.error("--compensate-drag requires both --compensate and --drag")
    response = json.loads((ROOT / "analysis" / "all_cartesian_model.json").read_text())
    fit = json.loads((ROOT / "analysis" / "results.json").read_text())
    t, desired, contact = build_target(args.content, press_depth_mm=args.press_depth_mm)
    delay_ms = response["isotropic_parameters"]["delay_s"] * 1000
    tau_ms = response["isotropic_parameters"]["tau_s"] * 1000
    adjusted_desired = desired.copy()
    if args.compensate_drag:
        velocity = np.gradient(desired[:, :2], t, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        moving = contact & (speed > 1e-6)
        shift = max(0, fit["coulomb_d0_mm"] + fit["coulomb_mu_eff"] * args.press_depth_mm)
        adjusted_desired[moving, :2] += shift * velocity[moving] / speed[moving, None]
    target = (compensate_command(t, adjusted_desired, delay_ms=delay_ms, tau_ms=tau_ms)
              if args.compensate else adjusted_desired)
    follower, ink = simulate(
        t, target, contact, delay_ms=delay_ms, tau_ms=tau_ms,
        drag_intercept_mm=fit["coulomb_d0_mm"] if args.drag else 0.0,
        drag_per_depth=fit["coulomb_mu_eff"] if args.drag else 0.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        names = ["time_s", "contact"] + [f"{kind}_{axis}_mm" for kind in ("desired", "target", "follower", "ink") for axis in "xyz"]
        writer = csv.writer(handle)
        writer.writerow(names)
        for i in range(len(t)):
            writer.writerow([t[i], int(contact[i]), *desired[i], *target[i], *follower[i], *ink[i]])
    error = float(np.sqrt(np.mean(np.sum((follower[contact, :2] - desired[contact, :2])**2, axis=1))))
    ink_error = float(np.sqrt(np.mean(np.sum((ink[contact, :2] - desired[contact, :2])**2, axis=1))))
    print(f"Saved {len(t)} samples to {args.output}; "
          f"simulated pen-down XY tracking RMSE={error:.2f} mm, ink RMSE={ink_error:.2f} mm; "
          f"compensation={'on' if args.compensate else 'off'}; "
          f"empirical drag={'on' if args.drag else 'off'}")


if __name__ == "__main__":
    main()
