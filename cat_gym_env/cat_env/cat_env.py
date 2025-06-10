"""
cat_env.py
Author: Jessica Anz
References: https://github.com/openai/gym & https://gymnasium.farama.org/introduction/create_custom_env/
https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py
Description: Main environment script for the cat robot training
"""

# Imports
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import importlib.resources as pkg_resources
from scipy.spatial.transform import Rotation as R

# Main training class
class CatEnv(gym.Env):

    # Set render mode
    metadata = {"render_modes": ["human"], "render_fps": 60}

    # Initialize training environment
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Open robot model
        with pkg_resources.path("cat_env.assets", "cat.xml") as xml_path:
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Initialize variables for actions & observations
        # Sets actions between -1 and 1 for each motor
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.model.nu,), dtype=np.float32)
        
        # Define observation space as joint positions, joint velocities, and 2 custom observations
        obs_dim = self.model.nq + self.model.nv + 2

        # Define custom observations type for the phase signals (to promote sinusoidal motion in legs)
        self.observation_space = spaces.Box(low=np.full(obs_dim, -np.inf, dtype=np.float32),
                                            high=np.full(obs_dim, np.inf, dtype=np.float32),
                                            dtype=np.float32)

        # Initialize tracking variables
        self.prev_xpos = None
        self.phase = 0.0
        self.step_counter = 0
        self.max_steps = 1000

        # Define what joints to track phase of (hips)
        self.joint_idx_hip1 = 0  # front left hip
        self.joint_idx_hip3 = 4  # back left hip

    # Reset the environment
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)

        # Save data
        self.prev_xpos = self.data.qpos[0].copy()
        self.phase = 0.0
        self.step_counter = 0
        obs = self._get_obs()
        return obs, {}

    # Simulate each time step
    def step(self, action):
        self.step_counter += 1

        # Apply action and step simulation
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Get current state data
        xpos = self.data.qpos[0] # position
        xvel = self.data.qvel[0] # velocity
        quat = self.data.qpos[3:7] # rotation
        euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=False)
        roll, pitch, yaw = euler
        z_pos = self.data.qpos[2] # height of robot
        hip1_vel = self.data.qvel[self.joint_idx_hip1] # hip velocity
        hip3_vel = self.data.qvel[self.joint_idx_hip3]


        ########## DEFINE REWARDS ##########
        
        # FORWARD VELOCITY REWARD
        velocity_reward = 1.0 * xvel
        if xvel > 0.00001: # Check if velocity is positive (small positive number)
            fwd_bonus = 1.0 # add bonus
        else:
            fwd_bonus = -0.5

        # GAIT SYMMETRY REWARD
        gait_symmetry = 5.0 * -np.square(hip1_vel + hip3_vel)

        # FOOT CLEARANCE REWARD
        foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lowerleg_fl")
        foot_pos = self.data.xpos[foot_body_id]
        foot_z = foot_pos[2]
        foot_clearance_reward = 5.0 * max(0.0, foot_z - 0.01)


        ########## DEFINE PENALTIES ##########

        # INSTABILITY PENALTY
        tilt_penalty = -5.0 * (roll**2 + pitch**2)

        # CONTROL EFFORT PENALTY
        ctrl_penalty = -0.5 * np.sum(np.square(self.data.ctrl))

        # HEIGHT PENALTY
        z_target = 0.15  # Robot standing height
        z_dev_penalty = -5.0 * np.square(z_pos - z_target)

        # FALL CHECK
        fallen = (z_pos < 0.12)
        if fallen:
            terminated = True
            fall_penalty = -100.0
        else:
            terminated = False
            fall_penalty = 0.0


        ########## COMBINE REWARDS & PENALTIES ##########
        
        # Final reward
        reward = (
            velocity_reward +
            fwd_bonus +
            tilt_penalty +
            ctrl_penalty +
            z_dev_penalty +
            gait_symmetry +
            foot_clearance_reward +
            fall_penalty
        )

        # Print to check current rewards
        print(f"Rwd: {reward:.4f} | Vel: {velocity_reward:.4f}, Bonus: {fwd_bonus:.4f}, Tilt: {tilt_penalty:.4f}, "
              f"Ctrl: {ctrl_penalty:.4f}, ZDev: {z_dev_penalty:.4f}, Gait: {gait_symmetry:.4f}, "
              f"Foot: {foot_clearance_reward:.4f}, Fall: {fall_penalty}")

        # Check if episode is done
        truncated = self.step_counter >= self.max_steps

        # Returns data for this time step
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Add sine phase signals to encourage rhythmic behavior
        self.phase += 0.1  # CPG phase progression
        gait_phase = np.array([np.sin(self.phase), np.sin(self.phase + np.pi)])
        obs = np.concatenate([self.data.qpos, self.data.qvel, gait_phase])
        return obs.astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, "viewer"):
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
