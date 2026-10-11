# Estimated Cartesian trajectories

Each CSV preserves source row order and time and converts joint angles through the saved SO-101 URDF to pen-tip XYZ in millimeters, relative to the URDF `base_link` frame. This is not a paper-corner origin or a camera coordinate frame. No paper-plane flattening, trajectory fitting, smoothing, or scale correction is applied.

## Columns

- `source_row`: one-based data row in the original recording, excluding its header.
- `segment_index`: increments if the recorded elapsed time resets.
- `time_s`: original elapsed seconds.
- `follower_x_mm`, `follower_y_mm`, `follower_z_mm`: estimated physical follower pen tip, calculated from measured follower joint angles.
- `target_x_mm`, `target_y_mm`, `target_z_mm`: pen position predicted from the joint command.
- `leader_model_x_mm`, `leader_model_y_mm`, `leader_model_z_mm`: leader joint angles applied to the follower model and pen offset. These are intended follower-space positions, not the physical location of the leader in the room. Leader-only recordings cannot provide actual follower positions.

The five arm joints are interpreted in degrees using the existing recorded-angle-to-URDF mapping. The gripper opening is excluded from this fixed pen-tip model. The pen must remain rigidly attached relative to the modeled gripper frame for that assumption to hold.

## Method and limitations

Forward kinematics composes the URDF joint-origin transforms and joint-axis rotations, then transforms the pen offset (-7.49, 3.33, 37.58) mm from the gripper frame to the base frame. The original URDF uses meters; output is multiplied by 1000. Printed decimals indicate numerical resolution, not physical accuracy.

The prior pivot fit had about 10 mm RMS residual and the modeled paper corners about 22 mm closure error. Those diagnostics are not a guaranteed error bound or a percentage error for these recordings. Angle-zero conventions, model geometry, pen attachment, and paper alignment have not been independently validated for these trajectories. Output Z does not directly measure paper clearance, contact, or pressure. Use these exports for inspecting trajectories, not as validated autonomous motion commands.

Follower observations are read before sending the current target, so same-row differences can include sampling delay and normal tracking lag. The three-circle snapshot overlaps the full teleoperation recording and includes startup and other movements; it is not three individually segmented circles.

All nine source CSVs and both unique JSONL recordings were converted. Row counts and finite XYZ values were checked. `manifest.json` provides source paths, hashes, model hash, offset, and row counts. Original files remain unchanged. Reproduce with `python hardware/drawing/outputs/convert_cartesian.py` (NumPy required).
