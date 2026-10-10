# SO-101 simulation environment record

The three `*-pip-freeze.txt` files are exact package inventories from the WSL2 run on 2026-10-10. They are audit records; the editable `so101-sim` and `leisaac` entries contain paths from the original laptop. Use the commands below to create new environments on another Linux machine, then compare `pip freeze` with the inventories.

The machine-readable [`run-manifest.json`](run-manifest.json) records the dataset, checkpoint selection, commands, seeds, randomization ranges, and evaluation artifact paths. The [dated handoff](../handoffs/2026-10-10-lerobot-simulation.md) includes the measured results and real-arm limits.

Hardware used: Ubuntu 24.04 on WSL2, RTX 5070 Ti Laptop GPU (12,227 MiB), NVIDIA driver 616.92, CUDA 12.8 PyTorch wheels. LeIsaac source revision: `24d3bcd3f1e4585740fc79921782c41617237812` (LeIsaac 0.4.0).

## MuJoCo and state-only baseline

```bash
conda create -n so101-mujoco python=3.11 -y
conda activate so101-mujoco
python -m pip install -e '.[test]'
python -m pytest -q tests/test_env.py tests/test_camera.py
```

## LeRobot data, ACT training, and inference

```bash
conda create -n so101-lerobot python=3.11 -y
conda activate so101-lerobot
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'lerobot==0.4.2' 'mujoco>=3.3,<4' 'numpy==2.2.6'
python -m pip install --no-deps 'stable-baselines3==2.6.0'
python -m pip install 'gymnasium==1.1.1' 'matplotlib==3.10.7' 'fastapi>=0.115,<1' 'uvicorn>=0.34,<1'
python -m pip install --no-deps -e .
python -m pip check
```

Pass `--dataset.video_backend=pyav` to `lerobot-train` and `video_backend="pyav"` to `LeRobotDataset(...)`. TorchCodec is installed by LeRobot 0.4.2 but cannot load without shared FFmpeg libraries in this WSL image. PyAV successfully decodes the saved videos.
The selected ACT experiment was trained with [`scripts/train_act_no_vae.sh`](../scripts/train_act_no_vae.sh); the initial VAE experiment used [`scripts/train_act.sh`](../scripts/train_act.sh).

The training script accepts `LEROBOT_DEVICE`, `LEROBOT_STEPS`, `LEROBOT_BATCH_SIZE`, `LEROBOT_NUM_WORKERS`, and `LEROBOT_SAVE_FREQ` environment overrides. On a provisioned AMD ROCm machine, run [`scripts/amd_cloud_smoke.sh`](../scripts/amd_cloud_smoke.sh) after following the event's ROCm setup. It requires a HIP-backed PyTorch GPU, decodes the dataset, loads the checkpoint, and runs a 10-step ACT job. It was validated locally with a two-step NVIDIA CUDA run; AMD cloud execution awaits instance access. See the [competition execution record](../handoffs/2026-10-10-competition-execution.md).

## LeIsaac setup gate

```bash
conda create -n so101-leisaac python=3.11 -y
conda activate so101-leisaac
conda install -y -c nvidia cuda-toolkit=12.8.1
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac && git checkout 24d3bcd3f1e4585740fc79921782c41617237812
python -m pip install 'setuptools<81' 'wheel==0.43.0'
python -m pip install --no-build-isolation -e 'source/leisaac[isaaclab]' --extra-index-url https://pypi.nvidia.com
```

LeIsaac 0.4.0 / Isaac Lab 2.3.0 requires `packaging<24`; LeRobot 0.4.2 requires `packaging>=24.2`. Keep them in separate environments. The operator accepted the Omniverse EULA on 2026-10-10. The next task attempt stalled after Isaac Sim reported no Vulkan graphics device in WSL2. Asset URLs, the exact launch command, and diagnostic output are in the [Isaac launch record](../handoffs/2026-10-10-isaac-launch.md).
