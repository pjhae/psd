import random
import numpy as np
import os
import pickle

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class TrajectoryReplayMemory:
    def __init__(self, capacity, trajectory_length, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.capacity = capacity
        self.trajectory_length = trajectory_length
        self.buffer = []
        self.position = 0

    def push(self, trajectory):
        # trajectory is list of following tuples : (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = trajectory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        trajectories = random.sample(self.buffer, batch_size)
        # trajectories is list of list : [[[state, action, reward, next_state, done], ...], ...]

        # stack trajectory component
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for traj in trajectories:
            state, action, reward, next_state, done = zip(*traj)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # convert to numpy array/ add batch dim
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)