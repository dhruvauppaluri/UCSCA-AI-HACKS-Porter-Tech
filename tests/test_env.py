import numpy as np
import pytest

from so101_sim.env import OBJECTS, SO101PickPlaceEnv
from so101_sim.gripper import GripperMapping


def test_all_objects_reset_and_stay_finite():
    env = SO101PickPlaceEnv(stage="mixed")
    try:
        for name in OBJECTS:
            obs, info = env.reset(seed=17, options={"object_name": name})
            assert obs.shape == env.observation_space.shape
            assert info["object_name"] == name
            for _ in range(12):
                obs, reward, _, _, info = env.step(env.action_space.sample())
                assert np.isfinite(obs).all()
                assert np.isfinite(reward)
                assert np.isfinite(info["object_xyz"]).all()
    finally:
        env.close()


def test_gripper_percent_mapping_endpoints():
    mapping = GripperMapping()
    assert mapping.to_percent(mapping.to_joint(0)) == 0
    assert mapping.to_percent(mapping.to_joint(100)) == 100
    assert mapping.to_joint(-10) == mapping.to_joint(0)
    assert mapping.to_joint(110) == mapping.to_joint(100)


def test_pick101_contact_variant_has_finger_pads_and_no_proxy():
    with pytest.raises(ValueError, match="assist_grasp=False"):
        SO101PickPlaceEnv(physics_variant="pick101_contact")
    env = SO101PickPlaceEnv(stage="robust", assist_grasp=False,
                            physics_variant="pick101_contact")
    try:
        env.reset(seed=123, options={"object_name": "box"})
        assert env.model.geom("static_finger_pad").id >= 0
        assert env.model.geom("moving_finger_pad").id >= 0
        assert not env.assist_grasp
        assert env.model.opt.noslip_iterations == 3
    finally:
        env.close()


def test_robust_randomization_replays_from_seed_and_joint_targets_are_bounded():
    env = SO101PickPlaceEnv(stage="robust", control_mode="joint")
    try:
        _, first = env.reset(seed=123, options={"object_name": "bottle"})
        initial_ctrl = env.data.ctrl[:6].copy()
        _, second = env.reset(seed=123, options={"object_name": "bottle"})
        assert first["randomization"] == second["randomization"]
        assert 0 <= first["randomization"]["action_delay_steps"] <= 2
        for _ in range(3):
            env.step(env.action_space.high)
        assert np.all(np.abs(env.data.ctrl[:5] - initial_ctrl[:5]) <=
                      3 * env.ctrl.settings.max_joint_step_rad + 1e-6)
        assert np.all(env.data.ctrl[:6] <= env.action_space.high + 1e-6)
    finally:
        env.close()


def test_diagnostic_pick_place_requires_release():
    env = SO101PickPlaceEnv(stage="place", max_steps=240)
    try:
        env.reset(seed=2)
        finished = False
        for phase, mode, count in [
            ("approach", "object", 30), ("close", "object", 10),
            ("lift", "lift", 30), ("move", "tray", 70),
            ("lower", "lower", 35), ("release", "lower", 25),
        ]:
            for _ in range(count):
                if mode == "object":
                    target = env._object_xyz() + [0, 0, 0.025]
                elif mode == "lift":
                    target = env._object_xyz() + [0, 0, 0.10]
                elif mode == "tray":
                    target = env._tray_xyz() + [0, 0, 0.10]
                else:
                    target = env._tray_xyz() + [0, 0, 0.035]
                action = np.zeros(6)
                action[:3] = np.clip((target - env._ee_xyz()) / 0.018, -1, 1)
                action[5] = 1 if phase == "release" else -1 if phase != "approach" else 0
                _, _, done, truncated, info = env.step(action)
                if phase != "release":
                    assert not info["success"]
                if done or truncated:
                    finished = True
                    break
            if finished:
                break
        assert info["success"]
        assert info["lifted_once"]
        assert not info["assisted_grasp_active"]
    finally:
        env.close()
