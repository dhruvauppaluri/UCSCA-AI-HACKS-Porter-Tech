# Follower-only physical “21” test preparation

## Use the existing recorded-run playback

Use `physical_hover_21_PLAYBACK.csv` with the existing `play_dataforarm.py`.
It has the same seven-column layout as the original leader playback files:
`time_s`, five body-joint `.pos` columns in degrees, and `gripper.pos` in
percent open. All 265 body-joint targets and timestamps match the generated
hover plan. The gripper holds at **2.024922%**, taken from target row 536 of
the source recording `arm_motion_20261010_214924.csv`; check that this matches
the current pen grip before replaying. This is a recorded grip value, not a
new measurement of the robot.

The robot host's installed configuration was inspected on 2026-10-11 and
uses degrees for the body joints. The repository playback script and
`/home/amd-genai2/usefulcommands/play_dataforarm.py` had identical SHA-256:
`025761d6c8ba0fa97e7fd9c9d1ed8b56b6b6e0ab000b03a93085d59e9c79a7ea`.

From the downloaded bundle on the robot host, with the existing LeRobot
environment active, use:

```bash
python play_dataforarm.py physical_hover_21_PLAYBACK.csv \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00 \
  --robot-id my_awesome_follower_arm --speed 0.4
```

Playback takes **44 seconds**, plus the script's initial three-second wait.
The original script connects/configures the follower, schedules each row by
its timestamp, and calls `follower.send_action()` just as for a recorded run.
It does not interpolate an approach: position near the first row beforehand.
Its connection setup temporarily releases torque, and its default disconnect
releases torque again. Start and finish with the arm supported, as appropriate
for this script. It does not record follower feedback or check clearance.

This file tests a **30 mm nominal hover**, not contact. Compatibility with the
working playback does not establish the model-generated path's physical
accuracy. No new physical run has been performed.

## Optional monitored hover runner

`physical_hover_21_FOR_REVIEW.csv` and `physical_contact_candidate_21_FOR_REVIEW.csv`
are SO-101 body-joint targets in calibrated degrees. They were generated from
`paper_waypoints.csv` with the archived URDF forward kinematics and numerical
inverse kinematics. The gripper is intentionally left at its measured setting.

The new numerical solve brings **model-internal** tip residual to 0.22 mm RMS
for both plans. This does not fix the uncertain paper and tool calibration:
the original corner fit is 13.22 mm RMS and pivot/pen-tip fit is 9.80 mm RMS.
The simulation's paper frame also includes a replay alignment shift. In
particular, the real tip may be at a different height than the modeled tip.

### Supervised placement check

1. Keep the pen clear and ensure the paper and arm layout match the recorded
   setup. Stop teleoperation before opening the follower serial port here.
2. With teleoperation, position the follower within 2 degrees of the first
   row in `physical_hover_21_FOR_REVIEW.csv`, then stop teleoperation while
   retaining torque. Support the arm if its shoulder cannot hold position.
3. On the robot host, run

   ```bash
   python drawing_capture/run_21_hover.py \
     --plan roboalgo/examples/21/physical_hover_21_FOR_REVIEW.csv
   ```

   The script checks the follower calibration, enabled torque, starting pose,
   tracking gap, and modeled tip height. It runs at 40% of the planned speed,
   logs observed joints, leaves the gripper alone, and keeps torque enabled
   on exit. Stop with Ctrl+C if clearance or holding deteriorates.

The hover path is a **30 mm nominal** height. Clearance still needs direct
observation because the paper pose and pen-tip estimates have centimeter-scale
residuals. `physical_contact_candidate_21_FOR_REVIEW.csv` is separate and is
not accepted by this runner. Contact requires a fresh measured pen-touch pose
and a successful supervised hover result first.

`physical_prep_manifest.json` records the solver residuals and limits of this
preparation. Regenerate the CSVs with `python roboalgo/prepare_physical_21.py`
using Python with NumPy.
