"""
cat_env.py
Author: Jessica Anz
Reference: https://github.com/openai/gym & https://gymnasium.farama.org/introduction/create_custom_env/
Description: Main environment script for the cat robot training
"""

# Imports
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import importlib.resources as pkg_resources

# Main training class
class CatEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Open robot model
        with pkg_resources.path("cat_env.assets", "cat.xml") as xml_path:
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Initialize variables for actions & observations
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.model.nu,), dtype=np.float32)
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=np.full(obs_dim, -np.inf, dtype=np.float32),
                                            high=np.full(obs_dim, np.inf, dtype=np.float32),
                                            dtype=np.float32)
        self.prev_xpos = None

    def reset(self):
        # Reset environment
        mujoco.mj_resetData(self.model, self.data)
        self.prev_xpos = self.data.qpos[0].copy()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Take action
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Reward: forward movement in x-direction
        xpos = self.data.qpos[0]
        reward = xpos - self.prev_xpos
        self.prev_xpos = xpos

        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs.astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, "viewer"):
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()


    def close(self):
        # write if needed in future!
        pass
