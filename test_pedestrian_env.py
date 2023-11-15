import gymnasium as gym
import gym_mobile_agent

env = gym.make("MobileAgent-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
