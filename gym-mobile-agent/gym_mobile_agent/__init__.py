from gymnasium.envs.registration import register

register(
    id="MobileAgent-v0",
    entry_point="gym_mobile_agent.envs:MobileAgentEnv",
    max_episode_steps=300,
)
