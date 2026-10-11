# Physical SO-101 drawing session

This directory archives the October 10, 2026 (PDT) pen drawing work on the AMD-hosted SO-101 leader and follower. The camera files, joint captures, motion plans, scripts, and controller logs are preserved together. File timestamps inside the capture data are UTC and may read October 11.

## Measured setup

- Paper: 220 × 275 mm. The four captured marks define a 170 × 235 mm inset drawing rectangle, with origin at its bottom-left mark.
- The pen protrudes about 1.5 inches from the gripper. A fit to six poses touching one dot estimated the tip offset as `(-7.49, 3.33, 37.58) mm` in the URDF gripper frame.
- That fit left about 10 mm RMS disagreement among the six same-dot poses; the corner model left about 22 mm of rectangle closure error. The captured joint readings are preserved, but the forward-kinematics conversion to pen-tip coordinates is not yet precise enough for autonomous pen-down drawing.
- The scripts use follower ID `my_awesome_follower_arm` and leader ID `my_awesome_leader_arm`. The live calibration files for those IDs are in `remote-data/calibration/`. The older repo snapshot under `data/hardware/calibration/` uses follower ID `my_awesome_follower_arm1`; do not substitute it silently.

## What happened

The four corner readings and six same-dot pivot readings were captured during leader-to-follower teleoperation. Cam0 and cam2 reference images were also saved, but camera measurements were not used to guide the attempted motion.

An autonomous hover-square attempt stopped before drawing. The follower shoulder sagged after control handoff, and subsequent servo-goal motion did not reliably lift the pen. The user observed paper contact. A 24 mm circle was planned from a user-confirmed center contact pose, but execution stopped at the second lift waypoint because the shoulder failed to follow its upward goal and drifted down. **No circle was drawn.** The arm was manually supported and powered off afterward.

On the later power-on, the follower reported all motor torques disabled and power-on default goals. A new teleoperation launch set the goals to measured positions before enabling torque, but it exited during `follower.configure()` when a write to motor ID 2 (`Torque_Enable`) failed. See `remote-data/teleop_live.log`. The process did not reach the teleoperation loop or start a circle recording. The communication and shoulder-holding fault needs hands-on resolution before further drawing attempts.

## Files

- `outputs/`: operator scripts, the in-loop teleoperation recorder, camera helpers, and reference photos.
- `work/`: captured corner and pivot readings, the URDF, geometry analysis scripts, and planned hover/circle trajectories. The plans are records of attempts, not validated safe motions.
- `remote-data/`: exact robot-host captures and controller logs, including failed motion traces, camera metadata and images, and the session's live calibration JSON.

Once the shoulder can hold position and motor ID 2 communicates reliably, `outputs/corner_capture.py` can run one persistent teleoperation loop. Its `record-start` and `record-stop` commands open and close a timestamped JSONL trajectory without stopping teleoperation or opening the serial port from another process. See `outputs/README-corner-capture.md` for the command sequence. The recorder was installed on the robot host, but no circle trajectory was captured in this session.
