"""Explore a generated writing run in MuJoCo's native interactive viewer.

P pause/play, arrows step, R restart, I ink, N light marks, T intended path,
B compare run, C reset camera. Drag to orbit and scroll to zoom. Run with mjpython.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from mujoco_recording_replay import ink_style
from simulate_text_mujoco import make_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-run-dir", type=Path)
    args = parser.parse_args()
    def load_run(directory):
        with (directory / "trace.csv").open(newline="") as handle:
            trace = list(csv.DictReader(handle))
        states = np.load(directory / "viewer_states.npz")
        summary = json.loads((directory / "summary.json").read_text())
        return trace, states, summary

    rows, states, summary = load_run(args.run_dir)
    comparison = load_run(args.compare_run_dir) if args.compare_run_dir else None
    title = "CARTOON DOG" if summary["drawing"] == "cartoon-dog" else summary["text"]
    model, _, origin, axes, *_ = make_scene()
    data = mujoco.MjData(model)
    events = queue.SimpleQueue()
    playing, show_ink, show_near, show_path = True, True, True, True
    index = 0
    paper_center = origin + axes @ np.array([.085, .1175, 0.])
    with mujoco.viewer.launch_passive(model, data, key_callback=events.put) as viewer:
        def reset_camera():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = paper_center
            viewer.cam.distance = .78
            viewer.cam.azimuth = 125
            viewer.cam.elevation = -60
        with viewer.lock():
            reset_camera()
        print(f"{title} MuJoCo viewer ready: P play/pause, arrows step, R restart, "
              "I ink, N possible light marks, T intended path, B compare, C view.", flush=True)
        while viewer.is_running():
            while not events.empty():
                key = events.get_nowait()
                if key == glfw.KEY_P:
                    playing = not playing
                elif key == glfw.KEY_R:
                    index, playing = 0, True
                elif key == glfw.KEY_I:
                    show_ink = not show_ink
                elif key == glfw.KEY_N:
                    show_near = not show_near
                elif key == glfw.KEY_T:
                    show_path = not show_path
                elif key == glfw.KEY_B and comparison is not None:
                    rows, states, summary, comparison = (*comparison, (rows, states, summary))
                    index = min(index, len(states["row_count"]) - 1)
                elif key == glfw.KEY_C:
                    with viewer.lock():
                        reset_camera()
                elif key == glfw.KEY_RIGHT:
                    playing = False
                    index = min(index + 1, len(states["row_count"]) - 1)
                elif key == glfw.KEY_LEFT:
                    playing = False
                    index = max(index - 1, 0)
            count = int(states["row_count"][index])
            with viewer.lock():
                data.qpos[:] = states["qpos"][index]
                data.qvel[:] = states["qvel"][index]
                data.time = float(rows[count - 1]["time_s"])
                mujoco.mj_forward(model, data)
                scene = viewer.user_scn
                scene.ngeom = 0
                if show_path:
                    for row in rows[::5]:
                        if not int(row["pen_down"]) or scene.ngeom >= scene.maxgeom:
                            continue
                        position = origin + axes @ np.array([
                            float(row["desired_x_mm"]) / 1000,
                            float(row["desired_y_mm"]) / 1000, .0012])
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                                            mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([.0005, 0., 0.]), position,
                                            np.eye(3).ravel(),
                                            np.array([.6, .6, .6, .65], dtype=np.float32))
                        scene.ngeom += 1
                if show_ink:
                    for row in rows[:count:5]:
                        if (not int(row["ink"]) or
                                (int(row["inferred_near_contact"]) and not show_near) or
                                scene.ngeom >= scene.maxgeom):
                            continue
                        position = origin + axes @ np.array([
                            float(row["ink_x_m"]), float(row["ink_y_m"]), .0015])
                        radius, color = ink_style({
                            "inferred_near_contact": int(row["inferred_near_contact"])})
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                                            mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([radius, 0., 0.]), position,
                                            np.eye(3).ravel(), color)
                        scene.ngeom += 1
            viewer.set_texts((None, None,
                              f"{title} {summary.get('lag_compensation', 'on')} correction | "
                              f"{data.time:.1f}/{float(rows[-1]['time_s']):.1f}s | "
                              f"{'play' if playing else 'pause'} | 4x time",
                              "P play/pause | arrows step | R restart | "
                              "I ink | N light | T target | B compare | C view"))
            viewer.sync()
            if playing:
                index = (index + 1) % len(states["row_count"])
            time.sleep(1 / 15)


if __name__ == "__main__":
    main()
