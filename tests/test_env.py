import numpy as np

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
