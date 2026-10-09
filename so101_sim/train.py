"""SAC curriculum training with resumable stage checkpoints."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .env import SO101PickPlaceEnv
from .expert import ScriptedExpert
from .demonstrations import load_demonstrations, select_demonstrations


DEFAULT_STEPS = {
    "reach": 40_000, "grasp": 60_000, "lift": 80_000,
    "place": 120_000, "mixed": 160_000, "robust": 160_000,
}


def _fit_actor(model: SAC, observations: list[np.ndarray], actions: list[np.ndarray], epochs: int) -> float:
    obs_tensor = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=model.device)
    action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=model.device)
    optimizer = torch.optim.Adam(model.policy.actor.parameters(), lr=1e-4)
    model.policy.actor.train()
    loss = None
    for _ in range(epochs):
        indices = torch.randperm(len(obs_tensor), device=model.device)
        for batch in indices.split(256):
            predicted = model.policy.actor(obs_tensor[batch], deterministic=True)
            loss = torch.nn.functional.mse_loss(predicted, action_tensor[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return float(loss.detach().cpu()) if loss is not None else 0.0


def bootstrap_from_expert(model: SAC, stage: str, episodes: int, seed: int = 2000,
                          epochs: int = 80, dagger_rounds: int = 0,
                          human_demonstrations: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Collect scripted transitions, prefill replay, and clone safe task behavior."""
    if episodes <= 0 and human_demonstrations is None:
        return {"episodes": 0, "success_rate": None, "samples": 0}
    observations = []
    actions = []
    successes = []
    if episodes > 0:
        env = SO101PickPlaceEnv(stage=stage)
        try:
            for episode in range(episodes):
                options = {}
                if stage in ("mixed", "robust"):
                    options["object_name"] = ("box", "bottle", "cylinder")[episode % 3]
                obs, _ = env.reset(seed=seed + episode, options=options)
                expert = ScriptedExpert(env)
                info = {"success": False}
                for _ in range(env.max_steps):
                    action = expert.act()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    observations.append(obs.copy())
                    actions.append(action.copy())
                    model.replay_buffer.add(obs[None], next_obs[None], action[None],
                                            np.array([reward]), np.array([terminated or truncated]), [info])
                    obs = next_obs
                    if terminated or truncated:
                        break
                successes.append(bool(info["success"]))
        finally:
            env.close()
    scripted_samples = len(observations)
    if human_demonstrations is not None:
        human_obs, human_actions = human_demonstrations
        observations.extend(human_obs)
        actions.extend(human_actions)
    loss = _fit_actor(model, observations, actions, epochs) if observations else 0.0
    dagger_successes = []
    rng = np.random.default_rng(seed + 50_000)
    for round_number in range(dagger_rounds if episodes > 0 else 0):
        rollout_env = SO101PickPlaceEnv(stage=stage)
        try:
            for episode in range(max(3, episodes // 3)):
                options = {}
                if stage in ("mixed", "robust"):
                    options["object_name"] = ("box", "bottle", "cylinder")[episode % 3]
                obs, _ = rollout_env.reset(seed=seed + 10_000 + 100 * round_number + episode, options=options)
                expert = ScriptedExpert(rollout_env)
                beta = max(0.05, 0.5 / (round_number + 1))
                for _ in range(rollout_env.max_steps):
                    label = expert.act()
                    prediction, _ = model.predict(obs, deterministic=True)
                    action = label if rng.random() < beta else prediction
                    next_obs, reward, terminated, truncated, info = rollout_env.step(action)
                    observations.append(obs.copy())
                    actions.append(label.copy())
                    model.replay_buffer.add(obs[None], next_obs[None], action[None],
                                            np.array([reward]), np.array([terminated or truncated]), [info])
                    obs = next_obs
                    if terminated or truncated:
                        break
                dagger_successes.append(bool(info["success"]))
        finally:
            rollout_env.close()
        loss = _fit_actor(model, observations, actions, max(20, epochs // 2))
    return {"episodes": episodes, "success_rate": float(np.mean(successes)) if successes else None,
            "dagger_rounds": dagger_rounds,
            "dagger_rollout_success_rate": float(np.mean(dagger_successes)) if dagger_successes else None,
            "scripted_samples": scripted_samples,
            "human_demo_samples": len(human_demonstrations[0]) if human_demonstrations is not None else 0,
            "samples": len(observations), "bc_loss": loss}


class SuccessCallback(BaseCallback):
    def __init__(self, stage: str, output_dir: Path, eval_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.stage = stage
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.best_rank = (-1.0, -float("inf"))

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq:
            return True
        env = SO101PickPlaceEnv(stage=self.stage, observation_noise=True)
        successes = []
        steps = []
        jerk_scores = []
        try:
            for ep in range(3):
                options = {"object_name": ("box", "bottle", "cylinder")[ep]} if self.stage in ("mixed", "robust") else {}
                obs, info = env.reset(seed=100_000 + ep, options=options)
                q = [np.array(info["joint_qpos"][:5])]
                for step in range(env.max_steps):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, truncated, info = env.step(action)
                    q.append(np.array(info["joint_qpos"][:5]))
                    if done or truncated:
                        break
                successes.append(float(info["success"]))
                steps.append(info["step_count"])
                if len(q) >= 4:
                    jerk = np.diff(np.stack(q), n=3, axis=0) / env.ctrl.settings.action_period_s**3
                    jerk_scores.append(float(np.sqrt(np.mean(jerk**2))))
        finally:
            env.close()
        score = float(np.mean(successes))
        record = {"time": datetime.now(timezone.utc).isoformat(), "stage": self.stage,
                  "steps": self.num_timesteps, "success_rate": score,
                  "mean_episode_steps": float(np.mean(steps)),
                  "rms_joint_jerk_rad_s3": float(np.mean(jerk_scores)) if jerk_scores else None}
        with (self.output_dir / "training.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        rank = (score, -record["rms_joint_jerk_rad_s3"] if jerk_scores else -float("inf"))
        if rank > self.best_rank:
            self.best_rank = rank
            self.model.save(self.output_dir / f"best_{self.stage}.zip")
        self.model.save(self.output_dir / f"latest_{self.stage}.zip")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=tuple(DEFAULT_STEPS), default=list(DEFAULT_STEPS))
    parser.add_argument("--steps-per-stage", type=int, help="Override the default steps in each selected stage")
    parser.add_argument("--checkpoint", type=Path, help="Resume from an existing SAC checkpoint")
    parser.add_argument("--out", type=Path, default=Path("runs/default"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="cpu, mps, cuda, or auto; validate AMD runtime on site")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--bootstrap-expert-episodes", type=int, default=12,
                        help="Scripted demonstration episodes per stage; set 0 for pure SAC")
    parser.add_argument("--bc-epochs", type=int, default=80)
    parser.add_argument("--dagger-rounds", type=int, default=2,
                        help="Corrective imitation rounds after scripted demonstrations")
    parser.add_argument("--demo-dataset", type=Path,
                        help="Optional NPZ from so101-teleop; demonstrations are cloned before SAC")
    args = parser.parse_args()
    if args.device == "cpu" or args.device == "auto":
        torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    model = None
    demonstration_dataset = load_demonstrations(args.demo_dataset) if args.demo_dataset else None
    if args.checkpoint:
        model = SAC.load(args.checkpoint, device=args.device)
    for stage in args.stages:
        env = Monitor(SO101PickPlaceEnv(stage=stage))
        if model is None:
            model = SAC("MlpPolicy", env, seed=args.seed, device=args.device,
                        learning_rate=3e-4, buffer_size=100_000, batch_size=256,
                        learning_starts=1_000, train_freq=1, gradient_steps=1,
                        policy_kwargs={"net_arch": [256, 256]}, verbose=1)
        else:
            model.set_env(env)
        allowed_objects = {"box"} if stage in ("reach", "grasp", "lift", "place") else set(("box", "bottle", "cylinder"))
        human_demonstrations = (select_demonstrations(demonstration_dataset, allowed_objects)
                                if demonstration_dataset is not None else None)
        if human_demonstrations is not None and not len(human_demonstrations[0]):
            human_demonstrations = None
        bootstrap = bootstrap_from_expert(model, stage, args.bootstrap_expert_episodes,
                                          seed=args.seed + 2000, epochs=args.bc_epochs,
                                          dagger_rounds=args.dagger_rounds,
                                          human_demonstrations=human_demonstrations)
        with (args.out / "bootstrap.jsonl").open("a") as file:
            file.write(json.dumps({"stage": stage, **bootstrap}) + "\n")
        model.save(args.out / f"bootstrap_{stage}.zip")
        steps = args.steps_per_stage or DEFAULT_STEPS[stage]
        callback = SuccessCallback(stage, args.out, max(1, min(args.eval_freq, steps)))
        model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback)
        model.save(args.out / f"final_{stage}.zip")
        env.close()
    print(f"Checkpoints and metrics: {args.out.resolve()}")


if __name__ == "__main__":
    main()
