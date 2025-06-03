"""
test_env.py
Author: Jessica Anz
Reference: https://gymnasium.farama.org/introduction/basic_usage/
Description: Tests the OpenAI gym setup for the quadruped cat robot
"""

# Imports
import gymnasium as gym
import cat_env

# Create environment
env = gym.make("CatWalk-v0")

# Reset environment
obs, info = env.reset()

# Loop through actions and print observations
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print("Observation:", obs)
    if done or truncated:
        break
env.close()
