import contextlib
import io
import json
from pathlib import Path

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import check_corners

fk = check_corners.fk
base = Path(__file__).parent
pivot = json.loads((base / "pivot.json").read_text())
corners = json.loads((base / "corners.json").read_text())


def fit(samples):
    rows = []
    rhs = []
    for item in samples.values():
        T = fk(item["joint_positions"])
        # p_i + R_i * tip_offset = same fixed point c.
        rows.append(np.c_[T[:3, :3], -np.eye(3)])
        rhs.append(-T[:3, 3])
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    solution, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    t, c = solution[:3], solution[3:]
    per_sample = {}
    for name, item in samples.items():
        T = fk(item["joint_positions"])
        per_sample[name] = np.linalg.norm(T[:3, 3] + T[:3, :3] @ t - c) * 1000
    return t, c, rank, singular, per_sample


t, c, rank, singular, per_sample = fit(pivot)
print("tip_offset_in_urdf_gripper_frame_mm", np.round(t * 1000, 2))
print("tip_offset_length_mm", round(np.linalg.norm(t) * 1000, 2))
print("fixed_dot_in_robot_base_mm", np.round(c * 1000, 2))
print("rank", rank, "condition", round(singular[0] / singular[-1], 2))
print("pivot_error_per_sample_mm", {k: round(v, 2) for k, v in per_sample.items()})
print("pivot_rms_mm", round(np.sqrt(np.mean(np.square(list(per_sample.values())))), 2))
for omit in pivot:
    sub = {k: v for k, v in pivot.items() if k != omit}
    tt, _, _, _, _ = fit(sub)
    print("tip_offset_without", omit, np.round(tt * 1000, 1))

q = {}
for name, item in corners.items():
    T = fk(item["joint_positions"])
    q[name] = T[:3, 3] + T[:3, :3] @ t
    print(name, "tip_in_robot_base_mm", np.round(q[name] * 1000, 1))

for a, b in (("bottom_left", "bottom_right"), ("top_left", "top_right"),
             ("bottom_left", "top_left"), ("bottom_right", "top_right")):
    print(a, b, "length_mm", round(np.linalg.norm(q[b] - q[a]) * 1000, 1))
closure = q["top_right"] - q["bottom_right"] - q["top_left"] + q["bottom_left"]
print("rectangle_closure_mm", round(np.linalg.norm(closure) * 1000, 1))

from itertools import combinations
for n in (5, 4):
    scored = []
    for subset in combinations(pivot, n):
        tt, cc, _, _, errors = fit({k: pivot[k] for k in subset})
        scored.append((np.sqrt(np.mean([errors[k] ** 2 for k in subset])), subset, tt, cc))
    rms, subset, tt, cc = min(scored, key=lambda item: item[0])
    held = {}
    for k in pivot:
        if k not in subset:
            T = fk(pivot[k]["joint_positions"])
            held[k] = round(np.linalg.norm(T[:3, 3] + T[:3, :3] @ tt - cc) * 1000, 1)
    print("best", n, "subset", subset, "rms_mm", round(rms, 1), "offset_mm", np.round(tt * 1000, 1), "held_out_errors_mm", held)
