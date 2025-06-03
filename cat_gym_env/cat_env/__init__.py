"""
__init__.py
Author: Jessica Anz
Reference: https://gymnasium.farama.org/api/registry/
Description: Registers the cat training environment
"""

from gymnasium.envs.registration import register

register(
    id="CatWalk-v0",
    entry_point="cat_env.cat_env:CatEnv",
    max_episode_steps=1000,
)
