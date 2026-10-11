# Follower-only physical “21” test preparation

`physical_hover_21_FOR_REVIEW.csv` and `physical_contact_candidate_21_FOR_REVIEW.csv`
are SO-101 body-joint targets in calibrated degrees. They were generated from
`paper_waypoints.csv` with the archived URDF forward kinematics and numerical
inverse kinematics. The gripper is intentionally left at its measured setting.

The new numerical solve brings **model-internal** tip residual to 0.22 mm RMS
for both plans. This does not fix the uncertain paper and tool calibration:
the original corner fit is 13.22 mm RMS and pivot/pen-tip fit is 9.80 mm RMS.
The simulation's paper frame also includes a replay alignment shift. In
particular, the real tip may be at a different height than the modeled tip.

## Supervised placement check

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
