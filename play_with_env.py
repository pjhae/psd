
import gym
import numpy as np


# from envs.register import register_custom_envs
# register_custom_envs()

# Environment
env = gym.make("Ant-v3")

state = env.reset()

while True:

    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
    env.render()

