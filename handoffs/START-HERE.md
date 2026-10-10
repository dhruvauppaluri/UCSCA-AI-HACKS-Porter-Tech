# SO-101: start here on another machine

Updated 2026-10-10. This is the current project handoff. For exact package versions, seeds, commands, and result files, use [`repro/run-manifest.json`](../repro/run-manifest.json) and the [simulation run record](2026-10-10-lerobot-simulation.md). The [resource index](resource-links.md) collects the external documents, hardware listing, repositories, and setup guides discussed so far.

## Goal and current status

Build a LeRobot camera-and-joint policy for real SO-101 pick-and-place. Simulation should provide the data, training, inference, and evaluation machinery before the loaner arm is available; real teleoperation should then reveal and correct the remaining mismatch. The physical kit has **not** been connected, calibrated, or tested in this repo. No real robot dataset exists yet.

Working now: MuJoCo scripted collection into LeRobot v3, ACT training, checkpoint loading, and repeatable simulated inference. The saved dataset has **60 successful demonstrations, 3,314 frames**, and 11 failed attempts logged separately. The selected ACT checkpoint is `artifacts/so101-2026-10-10/act_no_vae_checkpoints/002000`. It placed **13/30** held-out objects in the assisted-grasp scene (7/15 mixed, 6/15 robust). In the separate two-finger contact scene adapted from `pick-101`, it made **0/15** pad grasps, lifts, or placements. The scripted teacher also placed **0/15** in that contact scene. These are simulation results; the assisted placement score does not predict physical grip success. See the [per-object report and videos](2026-10-10-lerobot-simulation.md#held-out-measurements).

LeIsaac and Isaac Lab were installed separately, but the first task launch stopped at NVIDIA Omniverse's required EULA prompt. The prompt was not accepted; MuJoCo was used. The prior Stable-Baselines3 policies remain baselines, but their privileged state observations and Cartesian actions cannot directly drive the real LeRobot follower. No physical keyboard-control bridge exists. The MuJoCo keyboard recorder is ready but no human keyboard demonstrations were recorded; the operator chose to defer teleoperation until the real arm arrives.

## What the saved data actually contains

| Field | Saved simulation dataset | Real SO-101 example in the hackathon guide |
| --- | --- | --- |
| Robot and source | `so101_sim`; scripted teacher with privileged simulator pose | `so101_follower`; human drives follower with a leader arm |
| State and action | Six float32 values: five arm joints plus gripper; simulated joint positions and intended absolute joint targets, including provisional gripper angle, in radians | Six calibrated motor positions and sent targets; values depend on follower configuration and calibration |
| Camera | One simulated `observation.images.overhead`, 160×120 RGB | Example names `top` and `side`, each 640×480; the actual loaner camera names and mounts are unmeasured |
| Rate | 20 Hz | Example camera configuration requests 30 fps; achieved recording rate on the loaner is unmeasured |
| Task and grasp | Box, bottle, cylinder into tray; training uses assisted kinematic grasp | Example command describes cube into bin; real grasp uses physical fingers |
| Episodes | 60 successful sim episodes, 11 failed attempts separate | Example command requests 60 real teleoperation episodes; its repository does **not** contain those episode files |

The exact saved feature schema is in [`dataset/meta/info.json`](../artifacts/so101-2026-10-10/dataset/meta/info.json). LeRobot's `SO101Follower` v0.4.2 uses normalized **−100 to 100** for the five arm motors by default (`use_degrees=false`) and **0 to 100** for the gripper; `use_degrees=true` changes the arm-motor convention. The six-value shape does not make the current radian actions safe or compatible with those motor targets. Inspect the actual follower and its recorded values before setting any conversion, especially the gripper direction and limits.

The Amazon listing supplied for the approximate loaner kit describes a **Hiwonder** SO-ARM101 with two cameras, Hiwonder bus servos, and a BusLinker V3.0 board. The standard LeRobot `so101_follower` implementation uses a **Feetech** bus and STS3215 motors. The exact loaner revision, motor IDs, controller, and whether its supplied software exposes the standard LeRobot follower interface remain **unverified**. Do not assume the Seeed/Feetech port and driver commands apply until the physical labels and controller are checked.

## Decision: recollect after matching the loaner

Keep today's dataset and checkpoint as a reproducible baseline. Do **not** relabel its radian vectors or resize its overhead images and claim it is hardware-matched. Recollect a new, versioned **contact-valid** simulation dataset once the actual kit has been measured. Match the real recorder's field names and order, motor units and gripper convention, camera names and processing, target frame rate, and task setup. Recollecting generates trajectories under the corrected contact and control model; conversion alone cannot remove the assisted-grasp behavior learned in today's examples.

The interface can be matched exactly; images, latency, and physical contact can only be approximated and checked against real trials. If the real recorder uses two cameras, a checkpoint trained with only `overhead` cannot be passed a two-camera observation unchanged. Either recollect/retrain with both views or deliberately configure the real run to use a single matched view. A renamed image key alone cannot correct a different viewpoint. Keep source datasets immutable, record the mapping/calibration revision, and use disjoint evaluation seeds and separate real evaluation episodes.

## First session with the physical kit

1. Inventory the actual follower, leader, servo models, controller board, voltage, USB ports, and cameras. Confirm whether LeRobot's Feetech `so101_follower` driver supports this specific kit; use the vendor's appropriate integration if it does not. Mount the arm and make the motion area clear before powered trials.
2. Connect and calibrate the follower and leader with the verified driver. Record one stationary observation and one small, supervised, bounded movement per joint. Save the **raw** six-value state/action field names and values, directions, zero positions, safe limits, gripper endpoints, command clipping, and measured response lag. Do not replay today's simulated targets on the arm.
3. Capture synchronized samples from every intended camera. Record its name, placement, resolution, actual FPS, frame latency, color orientation, and visible workspace. Decide the training camera set and processing before collecting many episodes. Make the object, bin/tray, and reset positions measurable.
4. Record a few slow leader-to-follower real demonstrations and inspect their LeRobot v3 `meta/info.json`, decoded videos, state/action arrays, timestamps, and gripper direction. Keep failed physical trials for analysis, outside the successful imitation split.
5. Set the simulator's camera, motor mapping, latency, gripper geometry/contact, and task layout from those measurements. Generate a **new** contact-valid simulation dataset. Train a new checkpoint using the already working scripts and evaluate it on held-out sim seeds while collecting 20–50 varied real demonstrations. Fine-tune or mix sim and real only after their observations and actions have the same meaning; weight real examples more heavily if mixed.
6. Run supervised real pick-and-place evaluations. Compare pick, lift, place, release, time, motion smoothness, and failure videos. Add corrective demonstrations for the dominant real failures, then retrain. Prioritize reliable contact and placement before voice or broader vision features.

The first session is therefore calibration and targeted correction, with the data/training/evaluation infrastructure ready. It is **not** yet a physical-policy refinement-only day: the 0/15 contact result and unverified hardware interface are substantive gaps.

## Recover and run on another machine

```bash
git clone https://github.com/dhruvauppaluri/roboarm.git
cd roboarm
git lfs install
git lfs pull
```

The private repo requires access; Git LFS contains the dataset videos, checkpoint weights, evaluation videos, and other large artifacts (about 395 MB uploaded in the 2026-10-10 run). Follow [`repro/README.md`](../repro/README.md) to create the Linux environments. Its `pip freeze` files are inventories from the original WSL machine, including editable paths that must be reinstalled from this checkout. In the LeRobot environment, use PyAV video decoding. A fresh process already loaded the saved v3 dataset and selected checkpoint and ran simulated inference; the exact command is in the [run manifest](../repro/run-manifest.json) and [run handoff](2026-10-10-lerobot-simulation.md#reproduce). Check `git lfs ls-files` and `artifacts/so101-2026-10-10/dataset/meta/info.json` before diagnosing missing media.

The AMD hackathon notebook is a **training recipe** for an MI300X/ROCm machine, not a published loaner-arm dataset or checkpoint. It uses LeRobot 0.4.1 in the example, whereas our saved run used 0.4.2. Cloud GPU compute may help ACT training; the current MuJoCo physics loop is CPU based, so a cloud GPU alone does not accelerate it. Keep robot control and initial recording local. If moving training to AMD, install a version-compatible ROCm PyTorch environment and verify dataset decode and checkpoint loading there rather than copying the CUDA environment verbatim.

## Pointers

- [All external links and source roles](resource-links.md)
- [Measured run, comparisons, videos, and hardware handoff](2026-10-10-lerobot-simulation.md)
- [Exact machine-readable run manifest](../repro/run-manifest.json)
- [Environment reconstruction](../repro/README.md)
- [Detailed `pick-101` source review](2026-10-10-pick101-review.md)
- [Isaac SO-ARM101 lift tutorial assessment](2026-10-10-isaac-so-arm101-review.md)
- [Earlier project and conversation archive](2026-10-09-project-handoff.md)
