# SO-101 LeRobot simulation run — 2026-10-10

## Scope and hardware

This run used Ubuntu 24.04 on WSL2 with a 12 GB RTX 5070 Ti Laptop GPU (driver 616.92). The physical SO-101 was unavailable for calibration or testing. MuJoCo 3.15.0 provided the fallback simulator. LeRobot 0.4.2 and PyTorch 2.7.0+cu128 provided the image-and-joint ACT pipeline. Exact Python packages for the MuJoCo, LeRobot, and LeIsaac environments are in [`repro/`](../repro/README.md).

LeIsaac 0.4.0 at `24d3bcd3f1e4585740fc79921782c41617237812` and Isaac Lab 2.3.0 installed in a separate WSL environment. The initial lift-cube launch stopped at NVIDIA Omniverse's required EULA prompt, so the MuJoCo path was used. The separate environments are necessary because Isaac Lab requires `packaging<24` and LeRobot 0.4.2 requires `packaging>=24.2`. The LeIsaac smoke runner is [`scripts/leisaac_smoke.py`](../scripts/leisaac_smoke.py).

The user supplied [ggand0/pick-101](https://github.com/ggand0/pick-101) during this run. Its contact model at `c9d5400ffa0b742fe58a65750fed7982008450d9` is available as an optional scene with finger pads, elliptic friction, and no assisted grasp. See [`PICK101-ATTRIBUTION.md`](../so101_sim/model/PICK101-ATTRIBUTION.md) and the MIT notice. The training scene remains the original assisted-grasp scene; contact-scene scores are a separate transfer test because the robot's tool site and grasp physics differ.

## Implemented data and control path

The scripted teacher uses simulator pose information to make demonstrations. The saved policy input has only a 160×120 overhead RGB frame and six sensed joint positions. Each saved action is the teacher's intended absolute six-joint target, bounded by the simulator controller. Joint actions are in simulated radians, including a provisional gripper joint. They are **not** calibrated LeRobot hardware motor commands.

The recorder saves only successful demonstrations for imitation training. Its neighboring `*-attempts.json` records every attempt, including failures, seeds, randomization samples, and progress flags. The keyboard recorder uses the same schema and retains failed attempts separately; an operator must drive it to produce human episodes.
The completed dataset's attempt log predates the addition of explicit object/tray XY fields to `randomization_state`; its saved seed and object type reproduce those positions with this code version.

The completed dataset contains **60 successful episodes and 3,314 synchronized frames**: 20 boxes from 21 attempts, 20 bottles from 24 attempts, and 20 cylinders from 26 attempts. The 11 failed attempts are excluded from imitation training and retained in the attempt log for analysis. A fresh Python process loaded the dataset with `video_backend="pyav"` and decoded a 3×120×160 camera tensor, a six-value state, and a six-value action.

At the robust stage, each reset samples object and tray XY offsets (object ±4.5 cm, tray ±2.25 cm), object size ±12%, mass ±30%, object contact friction ±30%, servo gain ±15%, joint damping ±25%, joint sensor bias ±0.015 rad, two-axis camera tilt ±5°, and action latency of 0, 1, or 2 control steps. Each recorded camera stream uses a per-episode JPEG quality from 38–80 and per-frame brightness ±15%, perspective jitter ±2 pixels, and 10% chance of a 3×3 blur. Camera frame delay and dropout are disabled in recorded demonstrations to preserve image/action alignment. The mixed stage uses narrower ranges in `so101_sim/env.py`.

Training demonstration seeds start at 20,000 for boxes, 30,000 for bottles, and 40,000 for cylinders. The final report uses 60,000–60,204 for mixed and 70,000–70,204 for robust (five seeds per object). These ranges do not overlap training. Checkpoint selection uses a separate 80,000 range. Each object receives the same five held-out attempts in each policy comparison. Results use the assisted kinematic grasp proxy described in the README; the proxy makes physical grasp success unproven.

## Reproduce

From the repository root in the `so101-lerobot` environment:

```bash
python -m so101_sim.collect_lerobot \
  --out runs/2026-10-09/lerobot-robust-60-v3 \
  --repo-id local/so101-mujoco-robust-60-v3 \
  --stage robust --successes-per-object 20 --max-attempts-per-object 80 --seed 20000

bash scripts/train_act_no_vae.sh runs/2026-10-09/lerobot-robust-60-v3 \
  local/so101-mujoco-robust-60-v3 runs/2026-10-09/act-no-vae-robust-60

python -m so101_sim.eval_lerobot \
  --dataset artifacts/so101-2026-10-10/dataset \
  --repo-id local/so101-mujoco-robust-60-v3 \
  --checkpoint artifacts/so101-2026-10-10/act_no_vae_checkpoints/002000 \
  --out artifacts/so101-2026-10-10/evaluations/act-no-vae-heldout-30.json \
  --video-dir artifacts/so101-2026-10-10/videos/act-no-vae-heldout \
  --episodes-per-object 5 --stages mixed robust --seed 60000
```

The initial ACT model is a 17M-parameter ResNet18-backed policy with 256 model width, four attention heads, two encoder layers, a 32-step action chunk, and one action executed per inference call. Both training scripts use batch size 16, 4,000 update steps, seed 20261010, PyAV video decode, and checkpoints every 1,000 steps. LeRobot's default backbone weights and remaining ACT optimizer settings are retained. Each saved inference checkpoint contains the preprocessor and postprocessor.

Checkpoint selection used six separate robust episodes per checkpoint at seeds 80,000–80,201 (two per object). In the initial VAE ACT run, steps 1,000, 2,000, and 3,000 each placed 1/6; step 4,000 placed 0/6. Of the tied checkpoints, step 2,000 had the lowest mean RMS jerk (650.5 rad/s³). The alternate ACT configuration uses `use_vae=false` and executes eight actions per 32-action chunk. Its step 2,000 checkpoint placed 3/6 with 356.6 rad/s³ mean RMS jerk; step 4,000 placed 2/6 with 341.4 rad/s³. The selected policy is the alternate step 2,000 checkpoint, selected by held-out placement success first. The small selection sample is not the final success estimate.

## Held-out measurements

`pick` means a simulated grasp was registered; `lift` means the object rose above the task threshold; `place` means it settled in the tray. Speed and jerk are computed from simulated arm joint position differences at 20 Hz. All values below are from fixed seeds and are simulation-only.

| Controller | Set | Pick | Lift | Place | Failures | Mean duration | Mean max joint speed | Mean RMS jerk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scripted teacher | Mixed, 15 | 15/15 | 15/15 | 12/15 | 3 | 4.49 s | 1.91 rad/s | 255.7 rad/s³ |
| Scripted teacher | Robust, 15 | 15/15 | 15/15 | 13/15 | 2 | 4.12 s | 1.94 rad/s | 273.1 rad/s³ |
| New cloned state-only policy | Robust, 15 | 12/15 | 12/15 | 7/15 | 8 | 7.99 s | 2.01 rad/s | 294.2 rad/s³ |
| Previous `trained-v4/final_mixed.zip` | Robust, 15 | 6/15 | 6/15 | 5/15 | 10 | 9.72 s | 1.72 rad/s | 219.4 rad/s³ |
| Continued 50k-step SAC | Robust, 15 | 10/15 | 10/15 | 2/15 | 13 | 10.69 s | 2.39 rad/s | 566.8 rad/s³ |
| Scripted teacher, pick-101 contact scene | Robust, 15 | 2/15 | 1/15 | 0/15 | 15 | 7.37 s | 1.96 rad/s | 223.4 rad/s³ |
| Selected no-VAE ACT, overhead RGB + joints | Mixed, 15 | 7/15 | 7/15 | 7/15 | 8 | 7.86 s | 1.95 rad/s | 320.7 rad/s³ |
| Selected no-VAE ACT, overhead RGB + joints | Robust, 15 | 6/15 | 6/15 | 6/15 | 9 | 8.36 s | 2.23 rad/s | 340.0 rad/s³ |
| Selected no-VAE ACT, pick-101 contact scene | Robust, 15 | 0/15 | 0/15 | 0/15 | 15 | 12.00 s | 2.14 rad/s | 330.9 rad/s³ |

The continued SAC run used 25,000 mixed and 25,000 robust updates after 40 scripted bootstrap episodes and three DAgger rounds per stage. Its periodic three-episode checks showed 0% placement and increasing jerk. The selected state-only neural baseline is the behavior-cloned robust checkpoint, not the final SAC checkpoint. Neither state-only checkpoint is a ready hardware LeRobot policy: their inputs include privileged Cartesian task state and their actions are Cartesian increments.

The selected ACT placed **13/30** held-out objects overall. By object, mixed placement was box 2/5, bottle 1/5, cylinder 4/5; robust placement was box 1/5, bottle 1/5, cylinder 4/5. In the assisted-grasp training scene, every registered pick also lifted and placed, so pick/lift/place equality reflects the proxy mechanics and is not evidence of reliable physical gripping. The scripted teacher remains the strongest placement controller on these seeds, while the learned ACT checkpoint is the only saved policy here using camera and six-joint inputs/actions in LeRobot format.

The final artifact dataset and selected checkpoint were loaded from their saved paths in a fresh evaluation process. Twelve held-out MP4s in [`artifacts/so101-2026-10-10/videos/act-no-vae-heldout`](../artifacts/so101-2026-10-10/videos/act-no-vae-heldout) capture the first success and failure for every object in both stages. All twelve decoded successfully with ImageIO. The short [`scripted_box_demo.mp4`](../artifacts/so101-2026-10-10/scripted_box_demo.mp4) also decoded successfully. ImageIO's default MP4 writer pads the policy's 160×120 frames to 160×128 for codec compatibility; the training data remains 160×120.

The pick-101 contact row uses the same robust seed set but a different robot model, no assisted grasp, and a stricter pick definition requiring simultaneous object contact with both finger pads. The training-scene scores should not be compared as like-for-like physical grip results. A direct run of pick-101's published nominal top-down example in our MuJoCo 3.15.0 environment made two-finger contact and lifted the cube, then slipped during its hold and reported failure. The source lockfile specifies MuJoCo 3.4.0, which may affect the reference behavior; we have not reproduced its reported reinforcement-learning scores.
The selected ACT checkpoint achieved **0/15** two-pad grasps, lifts, or placements in this contact scene. Three contact failure videos, one per object, are in [`artifacts/so101-2026-10-10/videos/act-pick101-contact`](../artifacts/so101-2026-10-10/videos/act-pick101-contact); all decoded successfully. This result is a concrete indication that the assisted-scene ACT checkpoint is not a physical-grip solution.

## Physical-arm handoff

Before an actual pick, measure joint direction, zero, safe limits, gripper open/close endpoints, commanded-motion lag, camera-to-table transform, and grasp force. Run slow supervised reach and grip trials first. Replace the assisted-grasp assumptions with measured contact behavior, collect real corrective LeRobot episodes, then compare real placement outcomes. Voice and broad vision integration should follow a reliable supervised pick-and-place trial.

For real recording, LeRobot 0.4.2 has `lerobot-calibrate`, `lerobot-record`, and `lerobot-replay`. The follower's calibrated motor readings, the camera image, and the commanded action are saved together in the [LeRobot v3 dataset structure](https://huggingface.co/docs/lerobot/lerobot-dataset-v3). Give the follower a stable ID and use it consistently so its calibration is reused. Save locally with `--dataset.root=<directory>` and `--dataset.push_to_hub=false`; LeRobot's top-level `--resume=true` reopens completed episodes after an interruption. Before training, reopen the dataset with PyAV, inspect episode count and decoded video/state/action samples, and keep failed real trials out of the imitation split. A physical `lerobot-replay` command moves the arm, so use it only after supervised motion-limit checks.

**Keyboard integration still needs a real-arm bridge.** The installed LeRobot `keyboard_ee` teleoperator emits Cartesian deltas and a gripper command, while the SO-101 follower expects motor targets; the default recorder processor is identity. The current `so101-teleop-lerobot` script records the MuJoCo viewer only. A bounded, calibrated inverse-kinematics adapter is needed before keyboard control of the physical follower. The XIAO HTTP MJPEG camera also needs a validated LeRobot camera adapter or a tested OpenCV capture path. Until those are checked, do not treat the simulation checkpoint or keyboard recorder as a plug-in real-arm controller. The [LeRobot SO-101 setup guide](https://huggingface.co/docs/lerobot/so101) covers follower calibration, and the [imitation-learning guide](https://huggingface.co/docs/lerobot/il_robots) covers the real recording and training workflow.
