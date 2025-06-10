"""
run_training.py
Author: Jessica Anz
Reference: https://gymnasium.farama.org/introduction/basic_usage/
Description: Visualize the trained walking policy
"""

# Imports
import gymnasium as gym
import cat_env
import time
from stable_baselines3 import PPO

# Set to true for the policy to reset when the cat falls
reset = True

# Make training environment and load model
env = gym.make("CatWalk-v0", render_mode="human")
model = PPO.load("ppo_catwalk")

# Reset environment
obs, _ = env.reset()

# Open viewer
env.render()

# Loop through actions of trained policy
for i in range(5000):

    # Print info for each action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Reward: {reward:.3f}, Action: {action}")

    # Render to Mujoco
    env.render()

    # Restarts policy when cat falls
    if reset:
        if terminated or truncated:
            obs, _ = env.reset()

# Close window when done
env.close()