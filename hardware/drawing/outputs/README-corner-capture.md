# SO-101 corner capture

`corner_capture.py` runs leader-to-follower teleoperation and saves one follower joint-position snapshot for each requested paper corner. It uses the existing `my_awesome_leader_arm` and `my_awesome_follower_arm` calibrations and stable USB IDs. It does not use the camera, compute pen-tip coordinates, or move to stored corners.

## Before starting

Stop every existing `lerobot-teleoperate` process after bringing the arm to a safe, clear pose. **Do not run two teleoperation processes against the same serial ports.** The capture script checks for open ports and refuses to start if it detects another holder.

On the robot computer, use the LeRobot environment:

```bash
/home/amd-genai2/miniforge3/envs/lerobot/bin/python \
  /home/amd-genai2/drawing_capture/corner_capture.py run
```

The camera is omitted so the capture loop reads only the follower joints. Leader-to-follower control remains active while this command runs. Press Ctrl+C to stop it.

## Capture corners

Tape down the paper and mark the bottom-left, bottom-right, top-left, and top-right of the **usable drawing rectangle**. Keep the pen mount fixed. Guide the pen tip lightly to each mark and hold the leader still. In another terminal, run one command at a time:

The sheet is **220 mm wide by 275 mm long**. The four captured marks define an inset drawing rectangle measuring **170 mm wide by 235 mm long**. Its local coordinates are `(0, 0)`, `(170, 0)`, `(0, 235)`, and `(170, 235)` mm. These values are saved in `paper_config.json`. The inset's exact offset from the sheet edges has not been measured and is not needed if the four marks remain fixed.

```bash
python /home/amd-genai2/drawing_capture/corner_capture.py mark bottom_left
python /home/amd-genai2/drawing_capture/corner_capture.py mark bottom_right
python /home/amd-genai2/drawing_capture/corner_capture.py mark top_left
python /home/amd-genai2/drawing_capture/corner_capture.py mark top_right
```

You can also tell Codex “mark bottom left” while holding the pose, and Codex can issue the corresponding command over SSH. Wait for the saved-reading confirmation before moving to the next corner. Repeating a label replaces its latest snapshot.

## Record a hand-drawn circle without restarting teleoperation

Start `corner_capture.py run` once. While it remains in control of both arms, use a second terminal to start a trajectory log, draw with the leader, and stop the log:

```bash
python /home/amd-genai2/drawing_capture/corner_capture.py record-start rough_circle_01
# Draw by moving the leader; follower positions and leader commands are sampled in the same loop.
python /home/amd-genai2/drawing_capture/corner_capture.py record-stop
```

The recording is a timestamped JSONL file under `~/drawing_capture/data/`, with follower joint observations and leader joint commands at the teleoperation loop rate. `record-stop` closes the file while the teleoperation loop keeps running. `status` reports whether recording is active and how many samples have been saved. Start recording before touching the paper and stop after lifting the pen.

Read the saved points or service status:

```bash
cat /home/amd-genai2/drawing_capture/data/corners.json
python /home/amd-genai2/drawing_capture/corner_capture.py status
```

The file stores the follower's calibrated body-joint angles in degrees and its gripper position in percent open. The reported 1.5-inch pen-tip offset is recorded as an **estimate**; its direction and exact value must be calibrated before converting these snapshots to paper coordinates or commanding a drawing.

## Calibrate the pen-tip offset next

The updated script accepts `pivot_1` through `pivot_6`. To measure the pen tip relative to the robot's gripper frame, place the tip on **one fixed dot** in several different arm/wrist orientations. Lift and reposition between samples; do not drag the pen across the page. Hold the tip on that same dot for each `mark pivot_N` command. The six snapshots are kept in `data/pivot.json` separately from the four corners. A new script process must be started to activate this added feature; replacing the file does not update a process already running.

## Current measurements

The live capture process recorded all four inset corners and six pivot poses. The operator confirmed that the pen tip touched the **same dot** for all six pivot captures. Fitting the official SO-101 forward-kinematics model gives a pen-tip offset of approximately `(-7.5, 3.3, 37.6) mm` in the URDF gripper frame, with length `38.5 mm`. However, those six predicted tip positions disagree by `10.0 mm RMS` (individual errors `3.5–17.1 mm`). Applying the fitted offset to the four corners leaves a `22.0 mm` rectangle closure error. The drawing frame is therefore **not yet accurate enough for lettering or autonomous pen-down motion**.

These errors remain after checking LeRobot's joint-angle conversion and trying constant joint-zero corrections. The present data do not identify whether the main cause is arm model geometry, mechanical flex/backlash, or movement of the pen in its holder. A useful next measurement is to hold the pen tip on the same dot while approaching it repeatedly with the **same arm and wrist orientation**, then compare the recorded joint angles. That tests repeatability separately from geometric model error. Keep the pen mount rigid and the paper fixed for this measurement.

## Paper camera

Both USB cameras can see the workspace. After adjustment, `cam0` (`/dev/video0`) gives a fuller view of the paper and is now the default for snapshots. Its MJPEG server runs on robot localhost port 8767 and is forwarded to laptop `http://127.0.0.1:8767/cam0`. The original `cam2` (`/dev/video2`) remains available on robot localhost port 8765 and laptop `http://127.0.0.1:8766/cam2`. `camera_snapshot.py` reads either MJPEG stream without changing the live teleoperation process:

```bash
/home/amd-genai2/miniforge3/envs/lerobot/bin/python \
  /home/amd-genai2/drawing_capture/camera_snapshot.py blank
```

Snapshots and JSON metadata go to `~/drawing_capture/data/camera/`. Add `--camera cam2` to use the original camera instead. The helper was installed and tested on the robot computer while teleoperation remained active. Cam0's current 640 × 480 view shows the whole sheet and a reference frame was saved as `reference_cam0_20261010.jpg` in the shared outputs. Before measuring in millimeters, identify the four inset drawing-rectangle marks in that image and fit the perspective transform. Do not move the selected camera or paper between calibration and drawing.

To calculate drawing error in millimeters: (1) capture a clear blank reference image, (2) identify four visible marks with known paper coordinates to fit a pixel-to-paper perspective transform, (3) take a new image after each test dot with the arm out of view, and (4) detect the new ink relative to the blank image. Store each `(actual_x - target_x, actual_y - target_y)` pair with the commanded target and pose. Repeated dots at the same target estimate random variation; the average error across a grid supplies a correction map. The camera has not yet been calibrated to paper coordinates, so snapshots alone do not provide millimeter errors.
