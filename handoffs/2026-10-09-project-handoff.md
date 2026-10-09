# SO-101 project handoff — October 9, 2026

## Goal and event context

Build a Physical AI hackathon demo around an SO-101 arm: voice command through ElevenLabs, object selection and tabletop localization from a Seeed XIAO ESP32-S3 Sense camera, then pick and place into a tray. The UCSC AI Hackathon runs October 10–11, 2026. The attached event guide is preserved in `references/`.

## AMD Physical AI track fit

Challenge and prize sources: [Track Challenges](https://docs.google.com/document/d/19wl9rK9_BJdIbm7BN4tkNmrui55nEmjhIxGgQClBLSY/edit) and [Prizes](https://docs.google.com/document/d/1Nb-ehkfLU9l2BYOoXNlkzfudzZco8za_ukzgOnQIKzY/edit).

AMD offers two missions. Mission 1 is the clearest fit for our first demo: set up the full Hugging Face LeRobot environment end to end on AMD's stack, then pick up a small object and place it in a bin. Mission 2 is a freestyle real-world robotics application using AMD edge (AI PC) and cloud resources. Our voice + XIAO vision + SO-101 concept is a plausible Mission 2 extension, but only if it runs on the real arm and uses the AMD edge/cloud resources.

The current implementation is useful preparation, not yet a complete track submission: it is a custom MuJoCo/Gymnasium + Stable-Baselines3 environment, has not been connected to LeRobot or the loaner arm, and has not been run on the AMD hardware. Its `trained-v4/` checkpoints and demos are simulator artifacts. The current detector distinguishes three color-coded props; it is not a general object detector. No operator-collected training dataset exists yet.

### Event build order

1. At the venue, bring up the official LeRobot SO-101 path on the AMD Ryzen AI PC and verify slow arm motion, gripper direction, and emergency stop.
2. Complete the smallest real Mission 1 task first: pick one small object and put it in the bin. Record setup steps and a continuous demo video.
3. If time remains, layer in the XIAO camera localization and ElevenLabs voice command, with voice selecting the requested object and destination while the robot controller remains local and bounded.
4. Use AMD cloud/Radeon compute only after confirming the provided runtime and model stack; log the exact dataset, training command, checkpoint, and evaluation results.
5. Preserve the full code, training dataset, trained model, and demo video. The challenge says winning teams must publish all four publicly on GitHub. This repo is currently private by the user's request; make it public only when preparing the final track-compliant release.

The challenge document lists $1,000 in compute credits for first place and $500 for second place in the AMD track. The prize document also says all tracks can compete for the Overall Prize, gives an ElevenLabs Best Use sponsor prize, and sets the Devpost deadline at Sunday October 11, 1:30 PM (judging at 2:00 PM). Verify event announcements for any changes.

## Current project state

- `so101_sim/` contains the official SO-101 MuJoCo model and a Gymnasium pick-and-place environment for a box, bottle, and cylinder.
- `control.py` bounds Cartesian and wrist steps, maps them through damped Jacobian IK, respects joint limits, and interpolates joint targets smoothly.
- `train.py` has a reach → grasp → lift → place → mixed → robust SAC curriculum. It can initialize the actor from scripted examples, corrective imitation, and the human teleoperation dataset format.
- `teleop.py` adds keyboard demonstration collection before hardware is available. It records bounded keypress actions with simulator observations, object name, episode ID, and episode success into compressed NPZ files. On macOS, launch the MuJoCo viewer through `mjpython`.
- The teleoperation NPZ is a project-specific simulation dataset; it is not yet a LeRobot dataset and no human demonstration file has been collected.
- `camera.py` emulates the overhead camera and XIAO MJPEG stream; `vision.py` has a calibrated color detector for the three colored prototype objects. This is not a general object detector.
- `server.py` provides local `list_objects`, `pick_and_place`, `status`, and `stop` endpoints for a future ElevenLabs tool bridge. No ElevenLabs account or live robot connection is configured here.
- `trained-v4/` contains current curriculum checkpoints, training/bootstrap logs, and held-out evaluations. `videos/` has scripted demos and one learned bottle-command simulation.
- `experiments/archive/` preserves earlier checkpoints and smoke runs. Those model versions predate the current assisted-grasp setup or were quick pipeline experiments; use `trained-v4/` as the current reference.

## Measured checkpoint results

Ten held-out episodes per object were run in the randomized simulator. Results are in `trained-v4/eval_mixed.json`, `eval_mixed_on_robust.json`, and `eval_robust.json`.

| Checkpoint and evaluation stage | Box | Bottle | Cylinder | Total |
| --- | ---: | ---: | ---: | ---: |
| `final_mixed.zip`, mixed | 7/10 | 6/10 | 2/10 | 15/30 |
| `final_mixed.zip`, robust | 7/10 | 6/10 | 3/10 | 16/30 |
| `final_robust.zip`, robust | 4/10 | 8/10 | 2/10 | 14/30 |

These v4 runs used 40 scripted bootstrap episodes, behavior cloning, and three DAgger rounds per stage, but only one SAC environment step per stage. Treat them as imitation baselines, not converged RL policies. The held-out checks reported occasional MuJoCo instability and an extreme joint-speed outlier, so do not send these policies to physical hardware without calibration and conservative safety limits.

## Grasp and sim-to-real caveats

The official model's collision meshes did not reliably retain the objects during lift. The current default simulator uses an assisted kinematic grasp proxy and disables arm mesh collisions in that mode. The object still collides with the table and tray after release. This enables end-to-end motion-policy work but makes simulated grasp success optimistic. `assist_grasp=False` is available for strict contact experiments; calibrate gripper geometry and contact behavior from the loaner arm before making sim-to-real claims.

The gripper mapping is explicit but provisional. Verify joint directions, zeros, limits, gripper endpoints, camera intrinsics, table homography, latency, and emergency stop on the actual setup before allowing motion. There is intentionally no serial driver in the package.

## First steps for the next agent/operator

1. Install from this folder with Python 3.10–3.12: `python3 -m venv .venv && .venv/bin/python -m pip install -e '.[test]'`.
2. Try keyboard data collection on macOS: `.venv/bin/mjpython -m so101_sim.teleop --stage mixed --episodes 6 --out runs/teleop/demonstrations.npz`.
3. Continue from `trained-v4/final_mixed.zip` with that dataset using `--demo-dataset ... --stages mixed robust`; evaluate saved checkpoints on held-out starts and preserve the higher-success checkpoint.
4. At the venue, inspect the loaner and XIAO, calibrate first, and conduct slow reach/grip/release tests with a physical stop available.
5. Prioritize real LeRobot setup and one successful physical pick/place for Mission 1. Then update simulation parameters and wire ElevenLabs through an authenticated bridge if time allows. Keep the bridge local by default; do not expose unauthenticated actuator control.

## Provenance

The robot model and STL assets are from TheRobotStudio SO-ARM100 repository and retain its Apache-2.0 license in `so101_sim/model/`. The original hackathon guide provided for this project is in `references/`. Earlier generated run artifacts are preserved under `experiments/archive/`; local virtual environments, caches, and credentials are excluded by `.gitignore`.
