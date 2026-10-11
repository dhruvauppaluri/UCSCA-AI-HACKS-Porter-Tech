"""Open the recorded SO-101 writing run in MuJoCo's interactive viewer.

Run on macOS with mjpython. Drag to orbit, scroll to zoom; P pauses/resumes,
left/right arrows step through samples while paused, R restarts, I toggles ink,
and C resets the camera.
The paper and contact marks retain the calibration assumptions of the replay.
"""

from __future__ import annotations

import argparse
import json
import queue
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from mujoco_recording_replay import (
    ARM_XML, CALIBRATION, ROOT, SOURCE, build_drawing_model, fit_paper,
    fit_pen_tip, joint_arrays, load_recording, make_fk, replay_align_paper,
    run_simulation, tip_positions,
)


def prepare_run(mode):
    config = json.loads((CALIBRATION / "paper_config.json").read_text())
    results = json.loads((ROOT / "analysis/results.json").read_text())
    contact_window = (results["contact_start_s"], results["contact_end_s"])
    times, observed_deg, commanded_deg, grip_obs, grip_cmd = load_recording(SOURCE)
    base = mujoco.MjModel.from_xml_path(str(ARM_XML))
    fk = make_fk(base)
    bias, tip_site, _ = fit_pen_tip(fk)
    captured_origin, axes, _ = fit_paper(fk, bias, tip_site, config)
    observed_tip = tip_positions(fk, observed_deg, bias, tip_site)
    paper_origin, _ = replay_align_paper(captured_origin, axes, observed_tip,
                                         times, config, contact_window)
    model = build_drawing_model(paper_origin, axes, tip_site, config)
    observed, commands, _ = joint_arrays(model, observed_deg, commanded_deg,
                                          grip_obs, grip_cmd, bias)
    rows, _, snapshots = run_simulation(model, times, observed, commands,
                                         paper_origin, axes, config, mode,
                                         contact_window, capture_video=True,
                                         capture_stride=1)
    return model, times, rows, snapshots, paper_origin, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("driven", "replay"), default="driven")
    args = parser.parse_args()
    model, times, rows, snapshots, paper_origin, axes = prepare_run(args.mode)
    data = mujoco.MjData(model)
    events: queue.SimpleQueue[int] = queue.SimpleQueue()

    def on_key(key):
        events.put(key)

    playing = True
    show_ink = True
    index = 0
    paper_center = paper_origin + axes @ np.array([.085, .1175, 0.])
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        def reset_camera():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = paper_center
            viewer.cam.distance = .78
            viewer.cam.azimuth = 125
            viewer.cam.elevation = -60
        with viewer.lock():
            reset_camera()
        print("MuJoCo viewer ready. Drag: orbit; scroll: zoom; P: pause/play; arrows: step; R: restart; I: ink on/off; C: reset camera.", flush=True)
        while viewer.is_running():
            while not events.empty():
                key = events.get_nowait()
                if key == glfw.KEY_P:
                    playing = not playing
                elif key == glfw.KEY_R:
                    index = 0
                    playing = True
                elif key == glfw.KEY_I:
                    show_ink = not show_ink
                elif key == glfw.KEY_C:
                    with viewer.lock():
                        reset_camera()
                elif key == glfw.KEY_RIGHT:
                    playing = False
                    index = min(index + 1, len(snapshots) - 1)
                elif key == glfw.KEY_LEFT:
                    playing = False
                    index = max(index - 1, 0)
            qpos, qvel, row_count = snapshots[index]
            with viewer.lock():
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.time = float(times[index])
                mujoco.mj_forward(model, data)
                scene = viewer.user_scn
                scene.ngeom = 0
                if show_ink:
                    for row in rows[:row_count]:
                        if not row["ink"] or scene.ngeom >= scene.maxgeom:
                            continue
                        position = paper_origin + axes @ np.array(
                            [row["ink_x_m"], row["ink_y_m"], .0004])
                        mujoco.mjv_initGeom(
                            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([.0013, 0., 0.]), position,
                            np.eye(3).ravel(), np.array([0., .02, .15, 1.], dtype=np.float32))
                        scene.ngeom += 1
            viewer.set_texts((None, None, "Recorded writing simulation",
                              f"{times[index]:.1f} / {times[-1]:.1f} s | "
                              f"{'playing' if playing else 'paused'} | "
                              "P pause  arrows step  R restart  I ink  C camera"))
            viewer.sync()
            if playing:
                index = (index + 1) % len(snapshots)
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
