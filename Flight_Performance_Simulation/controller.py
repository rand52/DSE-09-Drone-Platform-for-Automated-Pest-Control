"""controller.py -- Actuator mapping for the Chameleon ducted-fan drone."""

from __future__ import annotations
import mujoco
import numpy as np

MAX_PITCH_RATE = 300.0   # deg/s


class FlightController:
    def __init__(self, model, data, body_name="drone",
                 max_thrust=15.0, mass=0.12, gravity=9.81,
                 max_pitch_rate=MAX_PITCH_RATE):
        self.model = model
        self.data = data
        self.max_thrust = float(max_thrust)
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.max_pitch_rate = float(max_pitch_rate)

        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model.")

        self.act_reel  = self._actuator_id("reel_motor")
        self.act_brake = self._actuator_id("friction_brake")
        self._reel_range  = model.actuator_ctrlrange[self.act_reel].copy()
        self._brake_range = model.actuator_ctrlrange[self.act_brake].copy()
        self._vel6 = np.zeros(6, dtype=float)

        # Body-axis thrust direction (world frame unit vector).
        # Caller should set this to point toward the moth before first step.
        self.thrust_dir = np.array([0.0, 0.0, 1.0])

        jnt_adr = model.body_jntadr[self.body_id]  #added for thrust
        self._free_qposadr = int(model.jnt_qposadr[jnt_adr])  #added for thrust
        self._free_dofadr  = int(model.jnt_dofadr[jnt_adr])   #added for thrust

    def _actuator_id(self, name):
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"Actuator '{name}' not found in model.")
        return aid

    def rotate_thrust_toward(self, desired_dir, dt):
        """Slew thrust_dir toward desired_dir at most MAX_PITCH_RATE deg/s (Rodrigues)."""
        desired = np.asarray(desired_dir, dtype=float)
        norm = np.linalg.norm(desired)
        if norm < 1e-9:
            return
        desired = desired / norm

        cos_a = float(np.clip(np.dot(self.thrust_dir, desired), -1.0, 1.0))
        angle = np.arccos(cos_a)
        if angle < 1e-7:
            self.thrust_dir = desired
            return

        max_angle = np.radians(self.max_pitch_rate) * dt
        step = min(angle, max_angle)

        axis = np.cross(self.thrust_dir, desired)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(axis, self.thrust_dir)) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
        else:
            axis = axis / axis_norm

        c, s = np.cos(step), np.sin(step)
        v = self.thrust_dir
        self.thrust_dir = (c * v
                           + s * np.cross(axis, v)
                           + (1.0 - c) * np.dot(axis, v) * axis)
        self.thrust_dir /= np.linalg.norm(self.thrust_dir)

    @property
    def hover_thrust(self):
        return self.mass * self.gravity

    def _thrust_dir_to_quat(self) -> np.ndarray:  #added for thrust
        """Quaternion [w,x,y,z] that rotates body +X onto thrust_dir."""  #added for thrust
        x = np.array([1.0, 0.0, 0.0])  #added for thrust
        d = self.thrust_dir  #added for thrust
        dot = float(np.clip(np.dot(x, d), -1.0, 1.0))  #added for thrust
        if dot > 1.0 - 1e-7:  #added for thrust
            return np.array([1.0, 0.0, 0.0, 0.0])  #added for thrust
        if dot < -1.0 + 1e-7:  #added for thrust
            return np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg around Y  #added for thrust
        axis = np.cross(x, d)  #added for thrust
        axis /= np.linalg.norm(axis)  #added for thrust
        half = np.arccos(dot) * 0.5  #added for thrust
        return np.array([np.cos(half), *(np.sin(half) * axis)])  #added for thrust

    def attitude_hold_torque(self, kp=0.6, kd=0.08):
        # Lock body +X (top cap) exactly to thrust_dir by writing qpos directly.  #added for thrust
        # Rate-limiting is handled upstream by rotate_thrust_toward, not here.    #added for thrust
        q = self._thrust_dir_to_quat()  #added for thrust
        self.data.qpos[self._free_qposadr + 3 : self._free_qposadr + 7] = q  #added for thrust
        self.data.qvel[self._free_dofadr  + 3 : self._free_dofadr  + 6] = 0.0  #added for thrust
        return np.zeros(3)  #added for thrust

    def apply_drone_wrench(self, thrust_magnitude, drag_force, attitude_hold=True):
        """Apply thrust along body axis (thrust_dir) + drag. Returns thrust vector."""
        thrust = float(np.clip(thrust_magnitude, 0.0, self.max_thrust)) * self.thrust_dir
        net_force = thrust + np.asarray(drag_force, dtype=float)
        self.data.xfrc_applied[self.body_id, 0:3] = net_force
        self.data.xfrc_applied[self.body_id, 3:6] = (
            self.attitude_hold_torque() if attitude_hold else 0.0)
        return thrust

    def set_reel_torque(self, value):
        self.data.ctrl[self.act_reel] = float(
            np.clip(value, self._reel_range[0], self._reel_range[1]))

    def set_brake_friction(self, value):
        self.data.ctrl[self.act_brake] = float(
            np.clip(value, self._brake_range[0], self._brake_range[1]))

    def release_spool(self):
        self.set_reel_torque(0.0)
        self.set_brake_friction(0.0)
