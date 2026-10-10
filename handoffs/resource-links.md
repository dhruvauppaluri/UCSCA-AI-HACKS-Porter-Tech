# SO-101 project links

Updated 2026-10-10. This index preserves the supplied Google Docs, kit listing, notebooks, repositories, and supporting documentation for another machine. Links are source references, **not** evidence that the physical loaner has the exact hardware or camera configuration shown in an example.

## Project and supplied event links

- [Private roboarm repository](https://github.com/dhruvauppaluri/roboarm) — canonical source, Git LFS artifacts, handoffs, and evaluation results.
- [Google Doc: track challenges](https://docs.google.com/document/d/19wl9rK9_BJdIbm7BN4tkNmrui55nEmjhIxGgQClBLSY/edit) and [Google Doc: prize breakdown](https://docs.google.com/document/d/1Nb-ehkfLU9l2BYOoXNlkzfudzZco8za_ukzgOnQIKzY/edit) — links previously supplied in the project conversation. Offline snapshots are in [`references/`](../references/); check the Docs for later updates. These documents were not used to infer the kit's data interface.
- [Event Discord invite in the supplied guide](https://discord.gg/s3Y6shpckU) — event coordination; availability may change.

## Approximate loaner kit and recording workflow

- [Supplied Amazon SO-ARM101 listing](https://www.amazon.com/SO-ARM101-Imitation-High-Torque-Recognition-Assembled/dp/B0GT9D7PGP?th=1) — approximate kit described as Hiwonder with dual cameras; verify the actual unit's motor bus, controller, voltage, and camera mounts.
- [Seeed SO-ARM100/101 LeRobot tutorial](https://wiki.seeedstudio.com/lerobot_so100m_new/) — assembly, Feetech setup, calibration, cameras, recording, training. Its example hardware may differ from the Amazon kit.
- [AMD hackathon repository](https://github.com/andrewgschmidt/AMD_Hackathon) — event setup material, inspected at `67b5e4c6948e38db310441dc3b1fe9a8dc56ace6`. [SO-101 example](https://github.com/andrewgschmidt/AMD_Hackathon/blob/main/robotics/robotics_2026/so101_example.md) shows leader/follower recording with `top` and `side` cameras at 640×480/30 fps; [ROCm quick start](https://github.com/andrewgschmidt/AMD_Hackathon/blob/main/robotics/robotics_2026/QuickStart.md), [MI300X training notebook](https://github.com/andrewgschmidt/AMD_Hackathon/blob/main/robotics/robotics_2026/training-models-on-rocm.ipynb), and [FAQ](https://github.com/andrewgschmidt/AMD_Hackathon/blob/main/robotics/robotics_2026/FAQ.md) cover cloud setup and troubleshooting. This repo contains recipes, not the example's actual 60 recorded episodes.
- [Official LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101), [imitation learning workflow](https://huggingface.co/docs/lerobot/il_robots), and [LeRobot dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) — real calibration, teleoperation, record/train/evaluate, and saved data format.
- [LeRobot v0.4.2 SO101Follower source](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/robots/so101_follower/so101_follower.py) and [configuration](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/robots/so101_follower/config_so101_follower.py) — authoritative field names, Feetech bus, default body normalization, optional degrees, and gripper range. Inspect the real kit before using this driver.

## Simulation and training sources

- [TheRobotStudio SO-101 simulation model](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) — original robot model/assets; [model notes](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md) discuss gripper mapping.
- [`ggand0/pick-101`](https://github.com/ggand0/pick-101) — contact-focused cube-lift RL reference. Its source reports a 2-million-frame result for lift, not tray placement; no published policy checkpoint was found in the inspected revision. See the [local deep review](2026-10-10-pick101-review.md) and model attribution.
- [SO-ARM101 Isaac Lab cube-lift walkthrough](https://medium.com/cvrealtime/training-a-simulated-so-arm101-robot-arm-to-lift-a-cube-with-nvidia-isaac-lab-e482b6141d0d) and its [upstream source](https://github.com/MuammerBay/isaac_so_arm101) — ready reach/lift PPO task with privileged simulator observations. See the [local source assessment](2026-10-10-isaac-so-arm101-review.md) for the exact task and LeRobot gaps.
- [LeIsaac installation](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/), [state-machine demonstration collection](https://lightwheelai.github.io/leisaac/docs/features/state_machine/), and [direct LeRobot recorder](https://lightwheelai.github.io/leisaac/docs/features/lerobot_recorder/) — optional simulator path; task launch in this project stopped at the Omniverse EULA prompt.
- [MuJoCo overview](https://mujoco.readthedocs.io/en/stable/overview.html) and [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) — physics and GPU-compute context; no AMD cloud run has been performed here.

## Later camera and voice integration

- [Seeed XIAO ESP32-S3 Sense camera example](https://wiki.seeedstudio.com/xiao_esp32s3_camera_usage/) — basis for the existing stream adapter; actual camera hardware and mount still need validation.
- [ElevenLabs webhook tools](https://elevenlabs.io/docs/eleven-agents/customization/tools/webhook-tools) and [realtime speech recognition](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/client-side-streaming) — optional voice integration after supervised physical pick-and-place works.
