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

# Make training environment and load model
env = gym.make("CatWalk-v0", render_mode="human")
model = PPO.load("ppo_catwalk")

# Reset environment
obs, _ = env.reset()

# Open viewer
env.render()

# Wait for user to adjust camera
print("Viewer opened. Adjust the camera, then press Enter to start the motion...")
input()
time.sleep(2)

# Loop through actions of trained policy
for _ in range(1000):
    # Print info for each action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Reward: {reward:.3f}, Action: {action}")

    # Render to Mujococ
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()
env.close()
input("Press Enter to close the viewer...")
