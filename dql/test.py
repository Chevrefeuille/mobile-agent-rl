import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
import gym_mobile_agent

import torch

from dql import DeepQLearning


if __name__ == "__main__":
    test_env = gym.make("MobileAgent-v0", render_mode="human")
    wrapped_test_env = NormalizeObservation(FlattenObservation(test_env))

    dql = DeepQLearning(wrapped_test_env, eval_env=None, name="MobileAgent")
    dql.load_weights("dql/models/mobile_agent_dql_v1/checkpoints/00037000.pt")

    print(dql.perform_episode(wrapped_test_env, truncate=False))
