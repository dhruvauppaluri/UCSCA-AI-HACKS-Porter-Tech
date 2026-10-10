"""Gymnasium pick-and-place environment for the official SO-101 model."""

from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .control import ControlSettings, SafeCartesianController
from .gripper import GripperMapping


MODEL_PATH = Path(__file__).parent / "model" / "scene_pick_place.xml"
CONTACT_MODEL_PATH = Path(__file__).parent / "model" / "scene_contact_pick_place.xml"
OBJECTS = ("box", "bottle", "cylinder")
OBJECT_GEOMS = tuple(f"object_{name}" for name in OBJECTS)
BASE_SIZES = {
    "box": np.array([0.018, 0.018, 0.020]),
    "bottle": np.array([0.014, 0.035, 0.0]),
    "cylinder": np.array([0.020, 0.025, 0.0]),
}


@dataclass
class Randomization:
    position_m: float = 0.0
    size_fraction: float = 0.0
    mass_fraction: float = 0.0
    friction_fraction: float = 0.0
    servo_gain_fraction: float = 0.0
    coordinate_noise_m: float = 0.0
    action_delay_steps: int = 0
    joint_damping_fraction: float = 0.0
    joint_sensor_bias_rad: float = 0.0
    camera_tilt_deg: float = 0.0


STAGES = {
    "reach": Randomization(position_m=0.0),
    "grasp": Randomization(position_m=0.015),
    "lift": Randomization(position_m=0.025),
    "place": Randomization(position_m=0.035),
    "mixed": Randomization(position_m=0.04, size_fraction=0.1, mass_fraction=0.25, friction_fraction=0.25,
                            servo_gain_fraction=0.1, coordinate_noise_m=0.004, action_delay_steps=2,
                            joint_damping_fraction=0.15, joint_sensor_bias_rad=0.008, camera_tilt_deg=3.0),
    "robust": Randomization(position_m=0.045, size_fraction=0.12, mass_fraction=0.30, friction_fraction=0.30,
                             servo_gain_fraction=0.15, coordinate_noise_m=0.006, action_delay_steps=2,
                             joint_damping_fraction=0.25, joint_sensor_bias_rad=0.015, camera_tilt_deg=5.0),
}


class SO101PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, stage: str = "reach", object_name: str | None = None,
                 render_mode: str | None = None, max_steps: int = 240,
                 observation_noise: bool = True, assist_grasp: bool = True,
                 control_mode: str = "cartesian", physics_variant: str = "original"):
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {tuple(STAGES)}")
        if object_name is not None and object_name not in OBJECTS:
            raise ValueError(f"object_name must be one of {OBJECTS}")
        if control_mode not in ("cartesian", "joint"):
            raise ValueError("control_mode must be cartesian or joint")
        if physics_variant not in ("original", "pick101_contact"):
            raise ValueError("physics_variant must be original or pick101_contact")
        if physics_variant == "pick101_contact" and assist_grasp:
            raise ValueError("pick101_contact requires assist_grasp=False")
        self.stage = stage
        self.object_name = object_name
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.observation_noise = observation_noise
        self.assist_grasp = assist_grasp
        self.control_mode = control_mode
        self.physics_variant = physics_variant
        model_path = CONTACT_MODEL_PATH if physics_variant == "pick101_contact" else MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.ctrl = SafeCartesianController(self.model, ControlSettings())
        self.substeps = round(self.ctrl.settings.action_period_s / self.model.opt.timestep)
        self.gripper_map = GripperMapping()
        self.qids = self.ctrl.qpos_ids
        self.vids = self.ctrl.dof_ids
        self.ee_id = self.model.site("gripperframe").id
        self.obj_id = self.model.body("object").id
        self.tray_id = self.model.body("tray").id
        self.gripper_body_id = self.model.body("gripper").id
        self.gripper_body_ids = {self.model.body("gripper").id,
                                 self.model.body("moving_jaw_so101_v1").id}
        self.obj_free = self.model.joint("object_free").id
        self.obj_qadr = self.model.jnt_qposadr[self.obj_free]
        self.obj_vadr = self.model.jnt_dofadr[self.obj_free]
        self.geom_ids = {name: self.model.geom(f"object_{name}").id for name in OBJECTS}
        if self.assist_grasp:
            # The official arm meshes are preserved for rendering. In proxy
            # mode, avoid stiff finger mesh contacts while the policy learns
            # coordinated arm movement; strict mode keeps every collision.
            for geom_id in range(self.model.ngeom):
                if self.model.geom_group[geom_id] == 3 and self.model.geom_bodyid[geom_id] != 0:
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0
        elif self.physics_variant == "pick101_contact":
            # Retain the two contact pads but avoid unstable full-mesh contacts
            # with this tabletop scene and the reference model's arm meshes.
            for geom_id in range(self.model.ngeom):
                if self.model.geom_group[geom_id] == 3 and self.model.geom_bodyid[geom_id] != 0:
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0
        self.initial_size = self.model.geom_size.copy()
        self.initial_mass = self.model.body_mass.copy()
        self.initial_inertia = self.model.body_inertia.copy()
        self.initial_friction = self.model.geom_friction.copy()
        self.initial_gain = self.model.actuator_gainprm.copy()
        self.initial_bias = self.model.actuator_biasprm.copy()
        self.initial_damping = self.model.dof_damping.copy()
        self.overhead_camera_id = self.model.camera("overhead").id
        self.initial_camera_quat = self.model.cam_quat[self.overhead_camera_id].copy()
        self.action_space = (spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
                             if control_mode == "cartesian" else spaces.Box(
                                 self.ctrl.limits[:, 0].astype(np.float32),
                                 self.ctrl.limits[:, 1].astype(np.float32), dtype=np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(36,), dtype=np.float32)
        self.renderer = None
        self.step_count = 0
        self.selected = "box"
        self.last_action = np.zeros(6)
        self.last_joint_delta = np.zeros(5)
        self.action_queue: list[np.ndarray] = []
        self.action_delay_steps = 0
        self.joint_sensor_bias = np.zeros(6)
        self.randomization_state: dict = {}
        self.grasped_once = False
        self.lifted_once = False
        self._success = False
        self._collision_count = 0
        self._settled_frames = 0
        self.observed_object_xyz: np.ndarray | None = None
        self._assisted_attached = False
        self._grasp_relative_pos = np.zeros(3)
        self._grasp_relative_quat = np.array([1.0, 0, 0, 0])

    def _randomize_model(self):
        cfg = STAGES[self.stage]
        self.model.geom_size[:] = self.initial_size
        self.model.body_mass[:] = self.initial_mass
        self.model.body_inertia[:] = self.initial_inertia
        self.model.geom_friction[:] = self.initial_friction
        self.model.actuator_gainprm[:] = self.initial_gain
        self.model.actuator_biasprm[:] = self.initial_bias
        self.model.dof_damping[:] = self.initial_damping
        self.model.cam_quat[self.overhead_camera_id] = self.initial_camera_quat
        for name, geom_id in self.geom_ids.items():
            active = name == self.selected
            self.model.geom_contype[geom_id] = 1 if active else 0
            self.model.geom_conaffinity[geom_id] = 1 if active else 0
            self.model.geom_rgba[geom_id, 3] = 1.0 if active else 0.0
        geom_id = self.geom_ids[self.selected]
        scale = self.np_random.uniform(1 - cfg.size_fraction, 1 + cfg.size_fraction)
        self.model.geom_size[geom_id, :2] *= scale
        if self.selected == "box":
            self.model.geom_size[geom_id, 2] *= scale
        mass = 0.025 * self.np_random.uniform(1 - cfg.mass_fraction, 1 + cfg.mass_fraction)
        self.model.body_mass[self.obj_id] = mass
        size = self.model.geom_size[geom_id]
        if self.selected == "box":
            self.model.body_inertia[self.obj_id] = mass / 3 * np.array([
                size[1] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[1] ** 2,
            ])
        else:
            radius, half_height = size[:2]
            transverse = mass * (3 * radius**2 + 4 * half_height**2) / 12
            self.model.body_inertia[self.obj_id] = [transverse, transverse, mass * radius**2 / 2]
        self.model.geom_friction[geom_id, 0] *= self.np_random.uniform(1 - cfg.friction_fraction, 1 + cfg.friction_fraction)
        gain_scale = self.np_random.uniform(1 - cfg.servo_gain_fraction, 1 + cfg.servo_gain_fraction, size=6)
        self.model.actuator_gainprm[:, 0] *= gain_scale
        self.model.actuator_biasprm[:, 1] *= gain_scale
        damping_scale = self.np_random.uniform(1 - cfg.joint_damping_fraction,
                                               1 + cfg.joint_damping_fraction, size=6)
        self.model.dof_damping[self.vids] *= damping_scale
        self.action_delay_steps = int(self.np_random.integers(0, cfg.action_delay_steps + 1))
        self.joint_sensor_bias = self.np_random.uniform(-cfg.joint_sensor_bias_rad,
                                                         cfg.joint_sensor_bias_rad, size=6)
        tilt = np.deg2rad(cfg.camera_tilt_deg)
        angles = self.np_random.uniform(-tilt, tilt, size=2)
        qx, qy, tilt_quat = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(qx, np.array([1.0, 0.0, 0.0]), float(angles[0]))
        mujoco.mju_axisAngle2Quat(qy, np.array([0.0, 1.0, 0.0]), float(angles[1]))
        mujoco.mju_mulQuat(tilt_quat, qx, qy)
        mujoco.mju_mulQuat(self.model.cam_quat[self.overhead_camera_id], self.initial_camera_quat, tilt_quat)
        self.randomization_state = {
            "configured": asdict(cfg), "object_scale": float(scale), "object_mass_kg": float(mass),
            "friction_scale": float(self.model.geom_friction[geom_id, 0] / self.initial_friction[geom_id, 0]),
            "servo_gain_scale": gain_scale.tolist(), "joint_damping_scale": damping_scale.tolist(),
            "action_delay_steps": self.action_delay_steps,
            "joint_sensor_bias_rad": self.joint_sensor_bias.tolist(),
            "camera_tilt_rad": angles.tolist(),
        }

    def _object_half_height(self):
        geom_id = self.geom_ids[self.selected]
        return float(self.model.geom_size[geom_id, 2] if self.selected == "box" else self.model.geom_size[geom_id, 1])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.selected = options.get("object_name", self.object_name)
        if self.selected is None:
            self.selected = str(self.np_random.choice(OBJECTS if self.stage in ("mixed", "robust") else ("box",)))
        if self.selected not in OBJECTS:
            raise ValueError(f"Unknown object: {self.selected}")
        self._randomize_model()
        mujoco.mj_resetData(self.model, self.data)
        cfg = STAGES[self.stage]
        spread = cfg.position_m
        ox = self.np_random.uniform(0.22 - spread, 0.22 + spread)
        oy = self.np_random.uniform(-0.09 - spread, -0.09 + spread)
        tx = self.np_random.uniform(0.27 - spread / 2, 0.27 + spread / 2)
        ty = self.np_random.uniform(0.115 - spread / 2, 0.115 + spread / 2)
        self.randomization_state["object_xy_m"] = [float(ox), float(oy)]
        self.randomization_state["tray_xy_m"] = [float(tx), float(ty)]
        self.model.body_pos[self.tray_id, :2] = [tx, ty]
        self.data.qpos[self.obj_qadr:self.obj_qadr + 3] = [ox, oy, self._object_half_height() + 0.002]
        self.data.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = [1, 0, 0, 0]
        self.data.qpos[self.qids] = [0, 0.25, -0.55, 0.3, 0, self.gripper_map.to_joint(100)]
        self.data.ctrl[:] = self.data.qpos[self.qids]
        mujoco.mj_forward(self.model, self.data)
        self.ctrl.reset(self.data)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.step_count = 0
        self.last_action[:] = 0
        self.last_joint_delta[:] = 0
        self.action_queue = []
        self.grasped_once = False
        self.lifted_once = False
        self._success = False
        self._collision_count = 0
        self._settled_frames = 0
        self.observed_object_xyz = None
        self._assisted_attached = False
        return self._observation(), self._info()

    def _set_assisted_grasp(self, active: bool):
        geom_id = self.geom_ids[self.selected]
        if active:
            # A kinematic grasp proxy is used until the real gripper geometry
            # can be measured; retain the object's closing pose in local axes.
            body_quat = self.data.xquat[self.gripper_body_id]
            obj_quat = self.data.xquat[self.obj_id]
            inv_quat = body_quat * np.array([1, -1, -1, -1])
            rotation = self.data.xmat[self.gripper_body_id].reshape(3, 3)
            self._grasp_relative_pos = rotation.T @ (self.data.xpos[self.obj_id] - self.data.xpos[self.gripper_body_id])
            mujoco.mju_mulQuat(self._grasp_relative_quat, inv_quat, obj_quat)
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
        else:
            self.model.geom_contype[geom_id] = 1
            self.model.geom_conaffinity[geom_id] = 1
        self._assisted_attached = active

    def _follow_gripper(self):
        rotation = self.data.xmat[self.gripper_body_id].reshape(3, 3)
        obj_pos = self.data.xpos[self.gripper_body_id] + rotation @ self._grasp_relative_pos
        obj_quat = np.zeros(4)
        mujoco.mju_mulQuat(obj_quat, self.data.xquat[self.gripper_body_id], self._grasp_relative_quat)
        self.data.qpos[self.obj_qadr:self.obj_qadr + 3] = obj_pos
        self.data.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = obj_quat
        self.data.qvel[self.obj_vadr:self.obj_vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def _object_xyz(self):
        return self.data.xpos[self.obj_id].copy()

    def _ee_xyz(self):
        return self.data.site_xpos[self.ee_id].copy()

    def _tray_xyz(self):
        return self.data.xpos[self.tray_id].copy() + np.array([0, 0, 0.024])

    def _observation(self):
        obj = self._object_xyz() if self.observed_object_xyz is None else self.observed_object_xyz.copy()
        tray = self._tray_xyz()
        if self.observation_noise and STAGES[self.stage].coordinate_noise_m > 0:
            noise = STAGES[self.stage].coordinate_noise_m
            obj += self.np_random.normal(0, noise, size=3)
            tray += self.np_random.normal(0, noise / 2, size=3)
        one_hot = np.array([self.selected == name for name in OBJECTS], dtype=float)
        joint_qpos = self.data.qpos[self.qids].copy()
        if self.observation_noise:
            joint_qpos += self.joint_sensor_bias
        obs = np.concatenate([
            joint_qpos, self.data.qvel[self.vids], self._ee_xyz(), obj, tray,
            self.data.qvel[self.obj_vadr:self.obj_vadr + 3], one_hot,
            self.last_action, np.array([self.grasped_once, self.lifted_once, self._assisted_attached]),
        ])
        return obs.astype(np.float32)

    def _contact_state(self):
        object_geom = self.geom_ids[self.selected]
        robot_contact = False
        table_contact = False
        tray_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if object_geom not in pair:
                continue
            other = pair[0] if pair[1] == object_geom else pair[1]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if name.startswith("tray_"):
                tray_contact = True
            elif name == "table":
                table_contact = True
            elif self.model.geom_bodyid[other] in self.gripper_body_ids:
                robot_contact = True
        return robot_contact, table_contact, tray_contact

    def _two_pad_contact(self):
        if self.physics_variant != "pick101_contact":
            return False
        object_geom = self.geom_ids[self.selected]
        pad_ids = {self.model.geom("static_finger_pad").id,
                   self.model.geom("moving_finger_pad").id}
        touched = set()
        for i in range(self.data.ncon):
            pair = {self.data.contact[i].geom1, self.data.contact[i].geom2}
            if object_geom in pair:
                touched.update(pair & pad_ids)
        return touched == pad_ids

    def _info(self):
        robot_contact, table_contact, tray_contact = self._contact_state()
        obj = self._object_xyz()
        tray = self._tray_xyz()
        ee = self._ee_xyz()
        in_tray = np.linalg.norm(obj[:2] - tray[:2]) < 0.043 and obj[2] < 0.075
        return {
            "object_name": self.selected,
            "object_xyz": obj.tolist(),
            "tray_xyz": tray.tolist(),
            "ee_xyz": ee.tolist(),
            "robot_contact": robot_contact,
            "two_pad_contact": self._two_pad_contact(),
            "table_contact": table_contact,
            "tray_contact": tray_contact,
            "grasped_once": self.grasped_once,
            "assisted_grasp_active": self._assisted_attached,
            "lifted_once": self.lifted_once,
            "success": self._success,
            "in_tray": in_tray,
            "collision_count": self._collision_count,
            "step_count": self.step_count,
            "joint_qpos": self.data.qpos[self.qids].tolist(),
            "joint_qvel": self.data.qvel[self.vids].tolist(),
            "randomization": self.randomization_state,
        }

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be six finite values")
        self.action_queue.append(action.copy())
        delayed = (self.action_queue.pop(0) if len(self.action_queue) > self.action_delay_steps else
                   (self.data.ctrl[:6].copy() if self.control_mode == "joint" else np.zeros(6)))
        old_q = self.data.qpos[self.qids[:5]].copy()
        old_obj = self._object_xyz()
        old_ee = self._ee_xyz()
        old_distance = np.linalg.norm(old_ee - old_obj)
        old_tray_distance = np.linalg.norm(old_obj[:2] - self._tray_xyz()[:2])
        was_grasped = self.grasped_once
        was_lifted = self.lifted_once
        old_collision_count = self._collision_count
        old_ctrl = self.data.ctrl[:6].copy()
        if self.control_mode == "joint":
            self.ctrl.apply_joint_target(self.data, delayed, self.substeps)
            action_for_reward = np.clip((self.data.ctrl[:6] - old_ctrl) /
                                        self.ctrl.settings.max_joint_step_rad, -1, 1)
            closing_command = self.gripper_map.to_percent(self.data.ctrl[5]) < self.gripper_map.to_percent(old_ctrl[5]) - 1
        else:
            self.ctrl.apply(self.data, delayed, self.substeps)
            action_for_reward = action
            closing_command = delayed[5] < -0.05
        if self._assisted_attached:
            self._follow_gripper()
        self.step_count += 1
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            names = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     for g in (contact.geom1, contact.geom2)}
            if ("table" in names or any(n.startswith("tray_") for n in names)) and any(
                    self.model.geom_bodyid[g] in self.gripper_body_ids for g in (contact.geom1, contact.geom2)):
                self._collision_count += 1
                break
        obj = self._object_xyz()
        ee = self._ee_xyz()
        tray = self._tray_xyz()
        robot_contact, table_contact, _ = self._contact_state()
        jaw_percent = self.gripper_map.to_percent(self.data.qpos[self.qids[5]])
        lifted = obj[2] > self._object_half_height() + 0.045
        near_obj = np.linalg.norm(ee - obj) < 0.055
        xy_aligned = np.linalg.norm(ee[:2] - obj[:2]) < 0.025
        if self.assist_grasp and not self._assisted_attached and closing_command and jaw_percent < 85 and near_obj and xy_aligned:
            self._set_assisted_grasp(True)
        elif self._assisted_attached and jaw_percent > 75:
            self._set_assisted_grasp(False)
        valid_contact = (self._two_pad_contact() if self.physics_variant == "pick101_contact"
                         else robot_contact)
        self.grasped_once |= (self._assisted_attached or valid_contact) and jaw_percent < 60 and near_obj
        self.lifted_once |= lifted and self.grasped_once
        in_tray = np.linalg.norm(obj[:2] - tray[:2]) < 0.043 and obj[2] < self._object_half_height() + 0.03
        released = jaw_percent > 75 and not self._assisted_attached
        self._settled_frames = self._settled_frames + 1 if in_tray and released and self.lifted_once else 0
        self._success = self._settled_frames >= 3
        joint_delta = self.data.qpos[self.qids[:5]] - old_q
        jerk_proxy = np.linalg.norm(joint_delta - self.last_joint_delta)
        smooth_penalty = 0.01 * np.linalg.norm(action_for_reward - self.last_action) + 0.03 * jerk_proxy
        speed_penalty = 0.002 * np.linalg.norm(self.data.qvel[self.vids[:5]])
        distance = np.linalg.norm(ee - obj)
        if was_lifted:
            reward = 4.0 * (old_tray_distance - np.linalg.norm(obj[:2] - tray[:2]))
        else:
            reward = 3.0 * (old_distance - distance) - 0.02 * distance
        reward -= smooth_penalty + speed_penalty + 0.002
        reward -= 0.2 * float(self._collision_count > old_collision_count)
        reward += 2.0 * float(self.grasped_once and not was_grasped)
        reward += 4.0 * float(self.lifted_once and not was_lifted)
        if self.stage == "reach":
            self._success = distance < 0.035
        elif self.stage == "grasp":
            self._success = bool(self.grasped_once)
        elif self.stage == "lift":
            self._success = bool(self.lifted_once)
        if self._success:
            reward += 40.0
        if obj[2] < -0.01 or abs(obj[1]) > 0.24 or obj[0] > 0.43:
            reward -= 3.0
        self.last_action[:] = action_for_reward
        self.last_joint_delta[:] = joint_delta
        terminated = self._success or obj[2] < -0.02
        truncated = self.step_count >= self.max_steps
        info = self._info()
        info["reward_terms"] = {"distance": float(2.0 * (old_distance - distance)),
                                "smoothness": float(-smooth_penalty), "speed": float(-speed_penalty)}
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.renderer.update_scene(self.data, camera="overhead")
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
