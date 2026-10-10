# Pick-and-place execution path — 2026-10-10

## Decision

Prioritize the AMD Physical AI Mission 1: a real SO-101 pick of one small, repeatable object and release into a generous bin, using LeRobot on the event's AMD stack. [The event track statement](../references/Track-Challenges-UCSC-AI-Hackathon.md) names a Radeon AI PRO R9700 cloud GPU for training and requires the full LeRobot workflow on AMD hardware. Isaac Sim is a separate NVIDIA experiment. It should not displace real-arm calibration, contact-valid demonstrations, or evaluation.

The saved ACT checkpoint placed 13/30 objects in the assisted-grasp MuJoCo scene but made 0/15 two-pad grasps in the contact scene. It is not a real-gripper policy. The contact scene and scripted controller now support a genuine two-pad lift/placement experiment; see the measured record below. Preserve the original assisted-grasp dataset and checkpoint as separate baselines.

## What can run now

Use a small cylinder or game piece as the first simulated target. The scripted sequence is approach above the object, align the finger pads, lower to grip height, close, lift vertically, travel above the bin, lower, release, and retreat. The contact simulator only registers a pick when **both finger pads touch** the object. `scripts/tune_contact_expert.py` searches approach height and lateral offset on tuning seeds; `so101_sim.eval_expert` evaluates disjoint fixed seeds. Both report pick, lift, and place separately. The [nine-setting tuning grid](../artifacts/so101-2026-10-10/evaluations/contact-cylinder-tuning.json) found no reliable robust setting, so data collection kept the default 0.008 m height and zero lateral offset.

The LeRobot contact recorder accepts `--physics-variant pick101_contact --objects cylinder`. It screens attempts without rendering and replays only successful seeds while saving synchronized overhead RGB video, six joint positions, and six joint targets. It raises an error if a successful seed fails replay. The attempt log records failures and randomization. The dataset is still a *simulated* radians-based dataset, not a real SO-101 motor-unit dataset.

The [first contact cylinder dataset](../artifacts/so101-2026-10-10/contact-cylinder-30) contains **30 successful episodes, 1,710 frames**, selected from 98 attempts with collection seed base 90,000 (actual cylinder seeds 110,000 onward). The [attempt log](../artifacts/so101-2026-10-10/contact-cylinder-30-attempts.json) includes all 68 failed attempts. A later [100-episode contact dataset](../artifacts/so101-2026-10-10/contact-cylinder-100) contains **5,720 frames** from 314 attempts, with [214 failures logged separately](../artifacts/so101-2026-10-10/contact-cylinder-100-attempts.json). It uses the same seed series and includes the first 30 demonstrations. Contact-training randomization here varies cylinder position (±0.035 m) and camera JPEG quality, brightness, blur, and perspective jitter; object mass, friction, damping, and latency remain fixed in these place-stage demonstrations. The earlier assisted-grasp robust dataset exercises broader physics variation but cannot be treated as contact-valid data.

A separate [30-episode scripted benchmark](../artifacts/so101-2026-10-10/evaluations/contact-cylinder-teacher-heldout-30-120steps.json) at seed base 130,000 and a 120-step cap registered two-pad contact in **26/30**, lift in **16/30**, and settled tray placement in **6/30**. The first ACT experiment trained for 1,000 updates on the 30-episode set. Four checkpoints were selected on separate seeds; the 750-step checkpoint had the best contact result (2/4 contact, 1/4 lift, 0/4 placement). On the same 30 held-out starts as the teacher it achieved **2/30 contact, 0/30 lift, 0/30 placement** ([report](../artifacts/so101-2026-10-10/evaluations/contact-act-750-heldout-30.json), [failure video](../artifacts/so101-2026-10-10/videos/contact-act-750-heldout/place_cylinder_failure.mp4)).

The 100-episode ACT experiment trained for 4,000 updates. Four checkpoints were tested on separate selection seeds; the 3,000-step checkpoint tied for the most picks (1/4) and had lower joint jerk than the 1,000-step checkpoint. On the 30 held-out place starts, the selected model achieved **0/30 contact, 0/30 lift, 0/30 placement** ([report](../artifacts/so101-2026-10-10/evaluations/contact-act-100-place-heldout-30.json), [failure video](../artifacts/so101-2026-10-10/videos/contact-act-100-place/place_cylinder_failure.mp4)). On 15 randomized contact starts it achieved **0/15 contact, lift, and placement** ([report](../artifacts/so101-2026-10-10/evaluations/contact-act-100-robust-heldout-15.json), [failure video](../artifacts/so101-2026-10-10/videos/contact-act-100-robust/robust_cylinder_failure.mp4)); the [scripted controller on those same starts](../artifacts/so101-2026-10-10/evaluations/contact-cylinder-teacher-robust-heldout-15.json) achieved **7/15 contact, 4/15 lift, 2/15 placement**. More successful scripted demonstrations did not improve the learned contact policy on this evaluation. The contact dataset, model, and videos are useful for diagnosing the task, but the scripted controller remains the stronger measured contact baseline. This is one simulated object and a 20% teacher placement rate on easier starts, so it is a narrow experimental starting point rather than a reliable grasp solution.

```bash
python -m so101_sim.eval_expert --out runs/contact-cylinder-heldout.json \
  --objects cylinder --stages robust --episodes-per-object 15 --seed 70000 \
  --physics-variant pick101_contact
python -m so101_sim.collect_lerobot --out runs/recollect-contact-cylinder-30 \
  --repo-id local/contact-cylinder-30 --stage place --objects cylinder \
  --physics-variant pick101_contact --successes-per-object 30 \
  --max-attempts-per-object 150 --seed 90000
LEROBOT_STEPS=4000 LEROBOT_SAVE_FREQ=1000 bash scripts/train_act_no_vae.sh \
  artifacts/so101-2026-10-10/contact-cylinder-100 \
  local/contact-cylinder-100 runs/contact-act-100-4000
```

Do not claim a dependable contact policy until a newly trained checkpoint beats the contact teacher on held-out starts. Do not merge the contact dataset with the assisted-grasp dataset without labeling the distinct physics.

## AMD cloud: training environment

The event's [SO-101 example](https://github.com/andrewgschmidt/AMD_Hackathon/blob/main/robotics/robotics_2026/so101_example.md) and [ROCm quick start](https://github.com/andrewgschmidt/AMD_Hackathon/blob/main/robotics/robotics_2026/QuickStart.md) describe the AMD Linux route. An R9700 supports ROCm, but the event instance image and installed versions must be checked after access is granted. ROCm PyTorch still uses `torch.device("cuda")`; `torch.version.hip` distinguishes it from NVIDIA CUDA ([PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)).

After the event provides an instance and a verified ROCm/PyTorch environment:

```bash
git clone https://github.com/dhruvauppaluri/roboarm.git
cd roboarm
git lfs install && git lfs pull
# Follow the event's ROCm quick start for its specific image and R9700.
# Install LeRobot 0.4.2 plus PyAV without replacing the ROCm PyTorch build.
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
bash scripts/amd_cloud_smoke.sh \
  artifacts/so101-2026-10-10/dataset local/so101-mujoco-robust-60-v3 \
  artifacts/so101-2026-10-10/act_no_vae_checkpoints/002000 runs/amd-act-smoke
```

The smoke script decodes a video frame, verifies six-value state/action data, loads the checkpoint, requires a ROCm GPU, and trains ACT for 10 steps. The main training script now accepts `LEROBOT_STEPS`, `LEROBOT_BATCH_SIZE`, `LEROBOT_NUM_WORKERS`, `LEROBOT_SAVE_FREQ`, and `LEROBOT_DEVICE` overrides. Run longer training only after the short job succeeds and save its package versions, dataset ID, command, checkpoint, and real evaluation results. As of this record, the operator has **no AMD cloud access**, so this cloud run has not happened.

## NVIDIA Brev credit: optional Isaac Sim test

The event guide lists **$25 in NVIDIA Brev GPU credits**. The operator has neither a redemption code/link nor a Brev account yet. When the event supplies a code, sign in at [NVIDIA Brev](https://brev.nvidia.com/), redeem it under Billing → Credits, inspect the live hourly price and remaining credit, and select **one L40S in VM mode** if available. NVIDIA's [Isaac Sim 5.1 Brev guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_advanced_cloud_setup_brev.html) explicitly uses L40S. First run the 5.1 compatibility/container smoke and verify a Vulkan RTX renderer, then try the pinned LeIsaac SO-101 task. The 5.1 stack is what this repo pinned; a newer Isaac Sim image may not run the same LeIsaac revision.

Restrict any streaming ports to your IP as the NVIDIA guide instructs. Brev bills running GPU time, including idle time; stop the VM when testing ends and push useful files before deleting it ([Brev instance lifecycle](https://docs.nvidia.com/brev/concepts/gpu-instances)). A $25 credit is a brief test budget, not a basis for assuming a long reinforcement-learning run. Do not spend it on A100/H100 compute-only hardware for Isaac's RTX renderer.

## First physical-arm session

1. Identify the actual arm controller, servos, leader/follower ports, power, and camera devices. Confirm the kit is supported by LeRobot's SO-101 driver before issuing motor commands. The approximate kit listing may not match the reference model.
2. Calibrate with the verified driver. For each joint, record raw observation/action name, direction, neutral value, safe endpoints, and response lag. Test one slow bounded motion at a time; do not replay simulator radians on the arm.
3. Fix the camera names and placement before collecting demonstrations. The event [SO-101 example](https://github.com/andrewgschmidt/AMD_Hackathon/blob/main/robotics/robotics_2026/so101_example.md) uses `top` and `side` at 640×480/30 fps; our simulated data uses only `overhead` at 160×120/20 fps. Record the real schema and recalibrate or recollect simulation data to match it.
   Run `python scripts/compare_lerobot_schemas.py SIM_DATASET REAL_DATASET` to get an exact metadata diff. Matching metadata alone does not establish matching motor units or physical behavior.
4. Mark one pickup region and a large bin. Teleoperate 20–50 **successful** varied picks with deliberate close, vertical lift, travel, and release. Save failures separately. Inspect one episode's video, state, action, timestamps, and gripper direction before recording the rest.
5. Train/fine-tune on AMD ROCm, then run at least 10 supervised real trials from fixed reset positions. Report pick, lift, release/placement, timing, and failure videos. Add corrective demos targeting the dominant failure. Keep a human teleoperation demonstration available if the policy fails.

The [cross-machine start page](START-HERE.md) and [run manifest](../repro/run-manifest.json) preserve the earlier data and policy. No physical arm or cloud access was available during this work; those outcomes remain unmeasured.

### Event command template after driver verification

These commands follow the event's [SO-101 example](https://github.com/andrewgschmidt/AMD_Hackathon/blob/main/robotics/robotics_2026/so101_example.md). Replace ports, IDs, and camera indices with detected values; use them only if the kit is compatible with LeRobot's `so101_follower` and `so101_leader` drivers.

```bash
lerobot-find-port
lerobot-find-cameras opencv
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=event_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=event_leader
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM1 \
  --robot.id=event_follower --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 --teleop.id=event_leader
```

After a slow bounded motion check, add the verified `--robot.cameras` mapping and record a *single* inspection episode with `lerobot-record` before committing to a larger batch. The event example supplies the full two-camera record and `--policy.path` evaluation commands. Keep dataset roots local until the first episode's video, timestamps, state, and action arrays are validated.
