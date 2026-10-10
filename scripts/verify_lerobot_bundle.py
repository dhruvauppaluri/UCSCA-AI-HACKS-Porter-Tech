"""Check video decode, six-joint fields, and ACT checkpoint on a new machine."""

import argparse
import json
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--require-rocm", action="store_true")
    args = parser.parse_args()
    if args.require_rocm and not (torch.cuda.is_available() and torch.version.hip):
        parser.error("ROCm PyTorch GPU is not available; check the cloud image and R9700 allocation")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LeRobotDataset(args.repo_id, root=args.dataset, video_backend="pyav")
    assert len(dataset) > 0, "dataset is empty"
    sample = dataset[0]
    assert tuple(sample["observation.state"].shape) == (6,)
    assert tuple(sample["action"].shape) == (6,)
    image = sample["observation.images.overhead"]
    assert image.numel() > 0, "decoded camera frame is empty"
    policy = ACTPolicy.from_pretrained(args.checkpoint).to(device).eval()
    param_count = sum(p.numel() for p in policy.parameters())
    print(json.dumps({"device": device, "torch": torch.__version__, "hip": torch.version.hip,
                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                      "frames": len(dataset), "image_shape": list(image.shape),
                      "state_shape": list(sample["observation.state"].shape),
                      "action_shape": list(sample["action"].shape),
                      "policy_parameters": param_count}, indent=2))


if __name__ == "__main__":
    main()
