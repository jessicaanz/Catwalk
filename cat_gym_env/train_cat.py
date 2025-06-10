"""
train_cat.py
Author: Jessica Anz
Reference: https://gymnasium.farama.org/introduction/basic_usage/
Description: Runs PPO training of the cat walking policy
"""

# Imports
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import cat_env

# Make training environment
env = gym.make("CatWalk-v0")

# Check that the environment works
check_env(env, warn=True)

# Create PPO model with MLP policy
model = PPO("MlpPolicy", env, verbose=1)

# Train model for total timesteps
model.learn(total_timesteps=300_000)

# Save trained policy
model.save("ppo_catwalk")
print("Completed training!")
