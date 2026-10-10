# Isaac Sim / LeIsaac launch record — 2026-10-10

The operator typed `Yes` in a separate PowerShell window while the NVIDIA Omniverse EULA was waiting in the WSL Isaac Sim process. PowerShell rejected it as a command. The same `Yes` was then sent to the waiting process, which printed `The EULA was accepted.` No further EULA action is required for this installation.

The isolated LeIsaac environment is `/home/dhruv/so101-miniforge/envs/so101-leisaac` in Ubuntu 24.04 WSL2. LeIsaac is at `24d3bcd3f1e4585740fc79921782c41617237812`. NVIDIA's WSL `nvidia-smi` reports the RTX 5070 Ti Laptop GPU, driver 616.92, 12,227 MiB. CUDA visibility alone did not make Isaac Sim renderable.

The first post-EULA launch failed because the task scene asset was absent. These official release assets were downloaded into ignored `runs/leisaac-assets/`:

- [SO-101 follower USD](https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd) → `runs/leisaac-assets/robots/so101_follower.usd`
- [Table with cube scene ZIP](https://github.com/LightwheelAI/leisaac/releases/download/v0.1.2/table_with_cube.zip) → extract `scene.usd` and companion files into `runs/leisaac-assets/scenes/table_with_cube/`

The manager-based `LeIsaac-SO101-LiftCube-v0` task then parsed but failed during environment creation because `actions.arm_action` and `actions.gripper_action` are missing from that base configuration. The direct task was attempted with one environment and two steps:

```bash
cd /mnt/c/Users/dhruv/OneDrive/Documents/ChatGPT/roboarm
LEISAAC_ASSETS_ROOT="$PWD/runs/leisaac-assets" \
  /home/dhruv/so101-miniforge/envs/so101-leisaac/bin/python \
  scripts/leisaac_smoke.py --task LeIsaac-SO101-LiftCube-Direct-v0 \
  --steps 2 --headless --enable_cameras
```

That attempt printed `APP_LAUNCHED`, `PARSING_TASK`, and `TASK_PARSED`, but never `ENV_CREATED`. Isaac Sim reported `nvidia-smi not found in '/usr/bin/'`, an empty Vulkan GPU table, `Driver Version: 0`, `No device could be created`, and `GPU Foundation is not initialized`. It also could not load `libGLU.so.1` for Iray. After more than a minute without progress, the task process was interrupted. WSL has `/dev/dxg` and the NVIDIA CUDA/management path, but its enumerated Vulkan ICDs do not include an NVIDIA ICD. The cause of this graphics failure is not yet proven; do not treat the task as launched or trained.

Next step: establish a supported NVIDIA Vulkan graphics path for Isaac Sim (or use a supported native Linux setup), verify `libGLU.so.1`, and rerun the direct-task smoke. Keep the existing MuJoCo dataset/checkpoint as the measured simulation result until an Isaac task reaches reset and step successfully.
