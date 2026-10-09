# SO-101 pick-and-place simulator

This project trains a MuJoCo SO-101 arm to move a small box, bottle, or cylindrical container into a tray. It includes a Gymnasium environment, bounded Cartesian controller, SAC curriculum, camera adapters, a color-coded prototype detector, evaluation metrics, and a local command API.

The arm model and STL files come from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) under Apache 2.0; see `so101_sim/model/LICENSE-APACHE-2.0`. The XIAO stream adapter follows Seeed's [XIAO ESP32-S3 Sense camera example](https://wiki.seeedstudio.com/xiao_esp32s3_camera_usage/). No physical robot interface is enabled in this package.

Project handoff notes and measured checkpoint results are in [`handoffs/2026-10-09-project-handoff.md`](handoffs/2026-10-09-project-handoff.md). Current checkpoints and evaluations are in `trained-v4/`; earlier experiments are preserved with context in `experiments/archive/`. The provided UCSC AI Hackathon guide is in `references/`.

## Project archive and continuation

The repository includes the searchable [conversation record](handoffs/conversation-record.md), a [handoff/archive map](handoffs/2026-10-09-project-handoff.md), the supplied [hackathon guide PDF](references/Hacker%20Guide%20-%20UCSC%20AI%20Hackathon.pdf) and searchable [guide text](references/Hacker-Guide-UCSC-AI-Hackathon.md), plus offline snapshots of the full [track challenges](references/Track-Challenges-UCSC-AI-Hackathon.md) and [prize breakdown](references/Prizes-UCSC-AI-Hackathon.md). The handoff states the immediate objective: complete as much feasible manipulation and motion training as possible today, then spend tomorrow calibrating/refining on the real SO-101 and integrating camera/voice features. The conversation record excludes tool traces and internal reasoning; machine/hardware data not yet collected is listed as pending rather than invented.

## Physical AI track status

The AMD challenge's Mission 1 requires a full Hugging Face LeRobot setup on AMD hardware and a real pick-and-place into a bin. This project currently rehearses the task in custom MuJoCo/SB3 simulation; it has not yet been ported to LeRobot or tested on the loaner hardware. Complete and record the real Mission 1 path first, then add XIAO vision and ElevenLabs voice if time permits. The challenge also requires winning teams to publish code, datasets, models, and demo videos publicly on GitHub. This repository stays private during development; see the dated handoff for the eventual public-release checklist and the Mission 2 option.

## Install

Use Python 3.10–3.12. From this folder:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

MuJoCo image rendering on macOS needs access to the desktop graphics session. State-based training and tests do not render images.

## Train

```bash
.venv/bin/so101-train --out runs/curriculum --device cpu
```

The default curriculum runs `reach`, `grasp`, `lift`, `place`, `mixed`, and `robust` in order. Each stage first clones scripted examples and runs corrective imitation rounds, then performs SAC updates. It saves `bootstrap_<stage>.zip`, `latest_<stage>.zip`, `best_<stage>.zip`, and `final_<stage>.zip`; `bootstrap.jsonl` and `training.jsonl` record the measured results. Defaults total 620,000 SAC simulator steps, so start well before the event and expect runtime to depend on the machine. A quick pipeline check is:

```bash
.venv/bin/so101-train --stages reach --steps-per-stage 1500 --eval-freq 750 --out runs/smoke --device cpu
```

For a fast imitation baseline without substantial SAC updates, set `--steps-per-stage 1`. Keep the `bootstrap_*.zip` files: SAC can temporarily reduce success when its critic is still learning, and the best policy should be selected by held-out success and motion metrics. The scripted examples are generated from simulator state; only evaluation of the saved neural policy measures learned performance.

To resume on another machine after copying a checkpoint:

```bash
.venv/bin/so101-train --checkpoint runs/curriculum/final_place.zip \
  --stages mixed robust --out runs/continued --device cpu
```

Set `--device` only to a PyTorch device verified on the event machine. MuJoCo dynamics still run on CPU; an AMD GPU can speed up the SAC neural-network updates only if the installed PyTorch build supports it. Keep the training data and checkpoints together when moving between macOS and Linux.

The first three stages use only the box. `mixed` and `robust` sample all three objects and increase randomization of positions, object size and mass, friction, servo gains, coordinate noise, and action delay. The policy observes joint state, end-effector and target poses, object velocity, object type, previous action, and grasp history. Its six actions are Cartesian XYZ, wrist flex, wrist roll, and gripper opening change.

## Evaluate and inspect

```bash
.venv/bin/so101-eval runs/curriculum/final_robust.zip --stage robust \
  --episodes-per-object 20 --out runs/curriculum/evaluation.json
```

Add `--video-dir runs/curriculum/videos` when graphics rendering is available. The evaluator reports placement and grasp success by object, task duration, collision count, maximum joint speed, and RMS acceleration and jerk. The first episode of each object is saved as an MP4.

Try a typed simulated request:

```bash
.venv/bin/so101-demo runs/curriculum/final_robust.zip --object bottle \
  --video runs/curriculum/bottle_demo.mp4
```

To inspect the simulator independently of the neural policy, record explicitly labeled scripted rollouts:

```bash
.venv/bin/so101-expert-demo --stage mixed --out-dir runs/scripted_demo
```

## Teleoperate and train from demonstrations

The keyboard teleoperator uses the same six bounded actions as the policy and controller. Each keypress makes one small movement step, so input cannot stream unbounded motion. On macOS, launch the viewer through MuJoCo's `mjpython` helper:

```bash
.venv/bin/mjpython -m so101_sim.teleop --stage mixed --episodes 6 \
  --out runs/teleop/demonstrations.npz
```

The viewer accepts `W/S` for X, `A/D` for Y, `R/F` for Z, `J/L` for wrist flex, `I/U` for wrist roll, and `O/P` for gripper open/close. Each keypress advances one bounded 4.5 mm, 0.02 rad, or 3% gripper step; no key holds a motor command open-ended. Press `N` or Enter to end the current episode; Escape saves and exits. The default rotates across the box, bottle, and cylinder. Use `--object box` to repeat one object, or `--append` to add episodes to an existing dataset. The saved NPZ contains the 36-value observations, normalized six-value operator actions, object names, episode IDs, and episode success labels.

Clone the human demonstrations into the policy before SAC continues:

```bash
.venv/bin/so101-train --checkpoint runs/curriculum/final_mixed.zip \
  --stages mixed robust --demo-dataset runs/teleop/demonstrations.npz \
  --out runs/teleop/continued --device cpu
```

The dataset is filtered by object for each curriculum stage; early box-only stages use box episodes, while mixed and robust stages use all three. Human demonstrations initialize the policy by behavior cloning. SAC then continues in MuJoCo, and the normal held-out evaluation still decides whether the resulting checkpoint improved. Collect varied start positions and successful placements; label partial or failed episodes accurately, since failed teleoperation is still useful for corrective training but should not be counted as a successful demo.

Start the local API for the ElevenLabs tool bridge:

```bash
.venv/bin/so101-serve runs/curriculum/final_robust.zip
```

Operations: `GET /list_objects`, `POST /pick_and_place` with `{"object":"bottle","destination":"tray"}`, `GET /status`, and `POST /stop`. `pick_and_place` acknowledges the command and executes it in the simulator's worker thread. The API binds to `127.0.0.1` by default. Configure ElevenLabs to call a protected bridge reachable from its service; do not expose a robot control endpoint publicly without authentication.

## Camera and vision

`CameraSource.get_frame()` returns RGB pixels, a capture timestamp, and a sequence number. `MuJoCoCamera` renders the overhead scene at 320×240 and adds JPEG compression, brightness variation, blur, dropped frames, and configurable frame delay. `XIAOCamera` reads the board's MJPEG URL through OpenCV. `ColorObjectDetector` applies the same image-to-table interface to either source and rejects detections outside the pickup region.

The current detector recognizes the three prototype objects by their distinct red, green, and yellow colors. It is not an open-vocabulary household-object detector. The physical camera needs a four-point tabletop calibration via `TableCalibration.from_points()`; the analytic `TableCalibration.simulated()` is only for MuJoCo's fixed overhead camera. Check the XIAO's actual sensor, lens, mounting angle, resolution, JPEG quality, and latency at the venue.

Mark four visible table points, measure their XY coordinates in meters relative to the arm base, read their pixel coordinates from a camera frame, and save a calibration file:

```bash
.venv/bin/so101-calibrate-camera \
  --pixels 80,60 240,60 240,190 80,190 \
  --table-xy 0.10,-0.15 0.34,-0.15 0.34,0.15 0.10,0.15 \
  --out calibration/table.json
```

The example values above are placeholders, not a valid calibration for the event setup. Load the result with `TableCalibration.load()` when connecting `XIAOCamera` to the controller.

## Gripper model and event handoff

The upstream MJCF has a six-joint position control model, but its gripper semantics are not mapped to LeRobot's `0=closed`, `100=open` convention. `GripperMapping` makes the provisional conversion explicit. Verify the direction and endpoints before controlling a real arm.

The upstream collision mesh did not retain the box when the arm lifted. This simulator uses an **assisted kinematic grasp** when the end effector is near the object and the gripper closes below 85% open; the jaw must close below 60% for a grasp to count. It releases above 75% open. Arm collision meshes are disabled in this proxy mode to avoid stiff mesh contact, while table and object physics remain active when the object is released. This makes full-task policy training possible, but its grasp success rate does **not** predict physical grip reliability. Run `SO101PickPlaceEnv(..., assist_grasp=False)` to evaluate strict mesh contact, and replace or calibrate the proxy from real gripper measurements before treating sim-to-real grasp results as evidence. The real robot should first execute slow reaches and monitored test grasps.

At the event, measure joint direction and zero, safe joint ranges, camera-to-table mapping, gripper opening, motion lag, and grasp outcomes. Feed these measurements back into the model and randomization ranges. The package intentionally does not issue serial commands to the SO-101 until that calibration is completed.
