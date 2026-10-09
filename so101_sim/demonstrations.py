"""Portable teleoperation demonstration dataset helpers."""

from pathlib import Path

import numpy as np


OBSERVATION_SIZE = 36
ACTION_SIZE = 6


def save_demonstrations(path: Path, observations: list[np.ndarray], actions: list[np.ndarray],
                        object_names: list[str], episode_ids: list[int],
                        episode_success: list[bool], append: bool = False) -> int:
    """Save labeled transitions, optionally appending to a prior dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not append:
        raise FileExistsError(f"{path} already exists; pass --append or choose a new --out path")
    obs = np.asarray(observations, dtype=np.float32).reshape((-1, OBSERVATION_SIZE))
    acts = np.asarray(actions, dtype=np.float32).reshape((-1, ACTION_SIZE))
    names = np.asarray(object_names, dtype="U16")
    ids = np.asarray(episode_ids, dtype=np.int32)
    if not (len(obs) == len(acts) == len(names) == len(ids)):
        raise ValueError("each observation and action needs one object name and episode id")
    if not np.isfinite(obs).all() or not np.isfinite(acts).all():
        raise ValueError("demonstrations contain non-finite values")
    if np.any(np.abs(acts) > 1.0001):
        raise ValueError("demonstration actions must be normalized to [-1, 1]")
    successes = np.asarray(episode_success, dtype=np.bool_)
    if append and path.exists():
        old = load_demonstrations(path)
        offset = int(old["episode_ids"].max() + 1) if old["episode_ids"].size else 0
        obs = np.concatenate((old["observations"], obs))
        acts = np.concatenate((old["actions"], acts))
        names = np.concatenate((old["object_names"], names))
        ids = np.concatenate((old["episode_ids"], ids + offset))
        successes = np.concatenate((old["episode_success"], successes))
    np.savez_compressed(path, observations=obs, actions=acts, object_names=names,
                        episode_ids=ids, episode_success=successes,
                        schema_version=np.asarray([1], dtype=np.int32))
    return len(obs)


def load_demonstrations(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the small, non-pickle NPZ demonstration format."""
    with np.load(path, allow_pickle=False) as data:
        required = {"observations", "actions", "object_names", "episode_ids", "episode_success"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"demonstration dataset is missing fields: {sorted(missing)}")
        result = {key: data[key].copy() for key in required}
    if result["observations"].ndim != 2 or result["observations"].shape[1] != OBSERVATION_SIZE:
        raise ValueError(f"observations must have shape (N, {OBSERVATION_SIZE})")
    if result["actions"].shape != (len(result["observations"]), ACTION_SIZE):
        raise ValueError(f"actions must have shape (N, {ACTION_SIZE})")
    if len(result["object_names"]) != len(result["observations"]) or len(result["episode_ids"]) != len(result["observations"]):
        raise ValueError("dataset transition fields have different lengths")
    if not np.isfinite(result["observations"]).all() or not np.isfinite(result["actions"]).all():
        raise ValueError("demonstration dataset contains non-finite values")
    if np.any(np.abs(result["actions"]) > 1.0001):
        raise ValueError("demonstration actions must be normalized to [-1, 1]")
    return result


def select_demonstrations(dataset: dict[str, np.ndarray], allowed_objects: set[str] | None = None):
    mask = np.ones(len(dataset["observations"]), dtype=bool)
    if allowed_objects is not None:
        mask = np.isin(dataset["object_names"], tuple(allowed_objects))
    return dataset["observations"][mask], dataset["actions"][mask]
