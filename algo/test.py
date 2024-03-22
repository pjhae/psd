import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from model_metra import Phi, Lambda
import matplotlib.pyplot as plt

from utils_sac import VideoRecorder
from utils_metra import generate_skill_disc, generate_skill_cont, add_noise_to_skill

from envs.register import register_custom_envs

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Ant-v3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=0.01, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=12345, metavar='N',
                    help='random seed (default: 12345)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=1024, metavar='N',
                    help='hidden size (default: 1024)')

parser.add_argument('--gradient_steps_per_epoch', type=int, default=50, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--episodes_per_epoch', type=int, default=8, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--skill_dim', type=int, default=2, metavar='N',
                    help='dimension of skill (default: 8)')
parser.add_argument('--radius_dim', type=int, default=3, metavar='N',
                    help='dimension of radius (default: 3)')
parser.add_argument('--radius_latent_dim', type=int, default=2, metavar='N',
                    help='dimension of radius latent (default: 3)')

parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: True)')
args = parser.parse_args()

register_custom_envs()

env_name = 'Ant-v3'
num_epi = 20000

# Environment
env = gym.make(env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Radius dim
radius_dim = args.radius_dim
radius_latent_dim = args.radius_latent_dim

# Agent
agent = SAC(env.observation_space.shape[0] + radius_dim, env.action_space, args)

agent.load_checkpoint("checkpoints/sac_checkpoint_{}_{}".format(env_name, num_epi), True )


avg_reward = 0.
avg_step = 0.
episodes = 10
while True:
    state = env.reset()
    radius = generate_skill_disc(radius_dim)
    # radius = np.array([0,1,0])
    state = np.concatenate([state, radius])
    episode_reward = 0
    step = 0
    done = False
    print(radius)
    while not done:
        
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        env.render()
        episode_reward += reward
        step += 1
        next_state = np.concatenate([next_state, radius])
        state = next_state

    print('episode_reward :' ,reward)
    print('episode_step :' ,step)
    avg_reward += episode_reward
    avg_step += step
    avg_reward += episode_reward
    avg_step += step

