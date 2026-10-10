# Measured SO-101 calibration, October 10, 2026

These are copies of the newly saved LeRobot calibration records from the AMD host. They preserve the raw encoder homing offsets and calibrated minimum and maximum counts for each motor. The values were read directly from the motor calibration registers at **2026-10-10 16:03:51 PDT** (23:03:51 UTC); both live register sets exactly matched their respective saved JSON files at that time. The read used `FeetechMotorsBus.read_calibration()` and did not send position commands or change calibration.

| Arm | LeRobot ID | Host file last modified (PDT) | Age at live check |
| --- | --- | --- | --- |
| Follower | `my_awesome_follower_arm1` | 2026-10-10 15:51:54 | 11 min 57 sec |
| Leader | `my_awesome_leader_arm` | 2026-10-10 15:53:30 | 10 min 21 sec |

Source files on the AMD host:

- Follower: `/home/amd-genai2/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm1.json`
- Leader: `/home/amd-genai2/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_awesome_leader_arm.json`

Repository copies are [`so_follower/my_awesome_follower_arm1.json`](so_follower/my_awesome_follower_arm1.json) and [`so_leader/my_awesome_leader_arm.json`](so_leader/my_awesome_leader_arm.json). They are measured hardware reference data for simulation mapping and analysis. The MuJoCo model has **not** been retuned from these raw counts: mapping encoder counts to model joint coordinates also requires confirming each joint's physical direction and zero pose. The stored ranges alone do not establish that every physical end stop was captured or that the gripper overload seen in earlier tests is resolved.

Use the exact IDs above when accessing these calibrations with LeRobot. A future recalibration may change homing offsets and raw counts; record a new dated snapshot rather than silently replacing this one.
