import gymnasium as gym
import gym_mobile_agent
from gymnasium.wrappers import FlattenObservation, NormalizeObservation

import torch

from dql import DeepQLearning


if __name__ == "__main__":
    train_env = gym.make("MobileAgent-v0")
    wrapped_train_env = NormalizeObservation(FlattenObservation(train_env))

    eval_env = gym.make("MobileAgent-v0")
    wrapped_eval_env = NormalizeObservation(FlattenObservation(eval_env))

    dql = DeepQLearning(
        wrapped_train_env, wrapped_eval_env, name="mobile_agent_dql_v1", gamma=0.9
    )

    dql.train(n_episodes=1000)
