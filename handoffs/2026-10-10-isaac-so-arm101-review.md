# Isaac SO-ARM101 cube-lift reference

Reviewed 2026-10-10. The supplied [Medium walkthrough](https://medium.com/cvrealtime/training-a-simulated-so-arm101-robot-arm-to-lift-a-cube-with-nvidia-isaac-lab-e482b6141d0d) uses [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101), inspected at commit `e4624dea075b00a36dbc66bebd531d191c92e8cd`. This reference is **useful for a new contact-focused Isaac Lab experiment**, not an immediately deployable LeRobot policy.

## What it provides

- SO-ARM100 and SO-ARM101 URDF/assets and registered reach and lift-cube Isaac Lab tasks. The SO-ARM101 lift task uses five joint-position arm actions and a binary gripper action with 0.0 closed / 0.5 open in that simulator. The code uses a five-second episode, 0.01-second physics step with decimation 2 (50 Hz control), 4,096 parallel environments by default, and 50 for play.
- A staged PPO reward for reaching the cube, lifting it above **2.5 cm**, and moving a raised cube toward a generated object goal, with action-rate and joint-velocity penalties. Object start position is randomized. The article reports a usable reach policy in under an hour on a 16 GB RTX 5070 Ti and suggests reducing parallel environments to 1,024 if VRAM is tight; it does not provide a held-out success table in the article.
- Training and play scripts that save and load RSL-RL PPO checkpoints. The repo's README labels **sim-to-real transfer as work in progress**. We did not find a published trained checkpoint in the inspected source tree.

## What differs from our goal

The PPO policy sees **joint positions, joint velocities, the cube's simulator-frame position, a generated target position, and previous action**. It does not consume camera pixels or LeRobot's recorded six-value observation/action schema. The binary simulator gripper command is not the loaner follower's calibrated 0–100 gripper target. Its result therefore cannot be loaded into our ACT inference code or directly driven through `lerobot-record`.

The task has a cube and a target for moving it while lifted; it does **not** define a bin/tray release and settled placement success criterion. Its reward and termination do not require two finger-pad contacts, so a positive lift reward is not by itself evidence of the two-pad grasp standard used in our contact benchmark. Camera matching, object variation, lighting randomization, real motor mapping, and recorded LeRobot demonstrations would need to be added for our sim-to-real data workflow.

The source model follows TheRobotStudio's SO-ARM101 geometry. The supplied approximate Amazon kit listing describes Hiwonder servos and controller. We have not verified that the physical loaner matches this URDF, gripper geometry, or actuator model.

## Recommended use

1. Once the operator handles the required Omniverse license prompt, run one SO-ARM101 lift play/smoke task in the isolated Isaac environment already prepared for this project. Start with **one environment**, then measure VRAM and raise parallelism conservatively on the laptop's 12 GB GPU. The article's 16 GB throughput is not a measured result for our 12 GB setup.
2. Use its reach/lift scene, robot import, PPO configuration, and reward terms as a **starting point for contact curriculum development**. Add explicit simultaneous finger-contact and hold criteria and report measured lift success on held-out seeds before extending to bin placement/release.
3. For our intended LeRobot policy, add camera observations and a recorder with the real kit's measured names, frame rate, motor-unit convention, and gripper direction. Recollect a new contact-valid dataset after physical calibration, then train/evaluate with the existing LeRobot pipeline. Preserve this PPO experiment as a separate benchmark unless its observation/action interface is deliberately aligned.

This project remains on the MuJoCo path until an Isaac task has actually launched, produced contact-valid episodes, and been compared with the same real-arm task. The existing LeIsaac environment reached the Omniverse EULA prompt but did not run a task; no Isaac training result is claimed here.
