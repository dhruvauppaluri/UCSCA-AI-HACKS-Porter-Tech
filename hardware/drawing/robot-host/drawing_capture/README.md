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

Read the saved points or service status:

```bash
cat /home/amd-genai2/drawing_capture/data/corners.json
python /home/amd-genai2/drawing_capture/corner_capture.py status
```

The file stores the follower's calibrated body-joint angles in degrees and its gripper position in percent open. The reported 1.5-inch pen-tip offset is recorded as an **estimate**; its direction and exact value must be calibrated before converting these snapshots to paper coordinates or commanding a drawing.

## Calibrate the pen-tip offset next

The updated script accepts `pivot_1` through `pivot_6`. To measure the pen tip relative to the robot's gripper frame, place the tip on **one fixed dot** in several different arm/wrist orientations. Lift and reposition between samples; do not drag the pen across the page. Hold the tip on that same dot for each `mark pivot_N` command. The six snapshots are kept in `data/pivot.json` separately from the four corners. A new script process must be started to activate this added feature; replacing the file does not update a process already running.

The script was tested in simulation without hardware and checked for imports and existing calibrations on the robot. It has **not** been started against the live arm.
