"""Prepare SO-101 joint candidates from the 21 paper trajectory, offline only.

The archived URDF and replay-aligned paper frame are estimates. Output is a
candidate for supervised testing, not evidence of physical pen clearance.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "roboalgo" / "examples" / "21"
sys.path.insert(0, str(ROOT / "hardware" / "drawing" / "outputs"))
from fk_so101 import tip_position  # noqa: E402


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LOW = np.array([-95., -95., -95., -95., -170.])
HIGH = np.array([95., 95., 95., 95., 170.])
HOVER_MM = 30.0


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fk(q):
    return tip_position(dict(zip(JOINTS, q))) * 1000


def solve(target, seed):
    q = np.clip(np.asarray(seed, dtype=float).copy(), LOW, HIGH)
    for _ in range(60):
        residual = target - fk(q)
        if np.linalg.norm(residual) < 0.5:
            break
        jac = np.column_stack([(fk(q + np.eye(5)[j] * 0.03) - fk(q - np.eye(5)[j] * 0.03)) / 0.06
                               for j in range(5)])
        damping = 0.3
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(3), residual)
        q = np.clip(q + np.clip(dq, -2., 2.), LOW, HIGH)
    return q, float(np.linalg.norm(target - fk(q)))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", *[f"{n}.pos" for n in JOINTS],
                                               "paper_x_mm", "paper_y_mm", "paper_z_mm",
                                               "pen_down", "fk_residual_mm"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    paper = read_csv(EXAMPLE / "paper_waypoints.csv")
    prior = read_csv(EXAMPLE / "candidate_arm_angles_FOR_REVIEW.csv")
    sim = json.loads((EXAMPLE / "source_sim" / "summary.json").read_text())
    if len(paper) != len(prior):
        raise ValueError("Paper and prior joint trajectories have different lengths")
    rotation = np.asarray(sim["paper_to_world_rotation"], dtype=float)
    origin = np.asarray(sim["paper_origin_world_m"], dtype=float) * 1000
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.02):
        raise ValueError("Invalid paper frame")
    outputs = {}
    for mode in ("hover", "contact_candidate"):
        result = []
        seed = np.array([float(prior[0][f"candidate_{n}_deg"]) for n in JOINTS])
        for point, candidate in zip(paper, prior):
            if abs(float(point["time_s"]) - float(candidate["time_s"])) > 1e-5:
                raise ValueError("Misaligned paper and joint rows")
            xyz = np.array([float(point[f"paper_{axis}_mm"]) for axis in "xyz"])
            if not 0 <= xyz[0] <= 170 or not 0 <= xyz[1] <= 235:
                raise ValueError("Paper point outside inset")
            if mode == "hover":
                xyz[2] = HOVER_MM
            target = origin + rotation @ xyz
            seed, error = solve(target, seed)
            row = {"time_s": point["time_s"], **{f"{n}.pos": float(v) for n, v in zip(JOINTS, seed)},
                   "paper_x_mm": float(xyz[0]), "paper_y_mm": float(xyz[1]),
                   "paper_z_mm": float(xyz[2]), "pen_down": 0 if mode == "hover" else int(point["pen_down"]),
                   "fk_residual_mm": error}
            result.append(row)
        errors = np.array([r["fk_residual_mm"] for r in result])
        jumps = np.max(np.abs(np.diff(np.array([[r[f"{n}.pos"] for n in JOINTS] for r in result]), axis=0)), axis=1)
        outputs[mode] = {"max_fk_residual_mm": float(np.max(errors)),
                         "rms_fk_residual_mm": float(np.sqrt(np.mean(errors ** 2))),
                         "max_step_deg": float(np.max(jumps)), "rows": len(result)}
        write_csv(EXAMPLE / f"physical_{mode}_21_FOR_REVIEW.csv", result)
    report = {"source": "paper_waypoints.csv", "paper_frame": "replay-aligned MuJoCo estimate",
              "hover_height_mm": HOVER_MM, "results": outputs,
              "prior_corner_fit_rms_mm": sim["corner_fit_rms_mm"],
              "prior_pen_tip_fit_rms_mm": sim["pivot_fit_rms_mm"],
              "physical_contact_validated": False,
              "initial_sag_mm": None,
              "note": "Numerical IK reduces the archived-FK discrepancy. It does not validate the paper pose, pen tip, sag, or actual clearance."}
    (EXAMPLE / "physical_prep_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
