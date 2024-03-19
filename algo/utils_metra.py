import math
import torch
import os
import sys

import imageio
import numpy as np


def generate_skill(dim, eval_idx = -1):

    vector = np.full(dim, -1/(dim-1))

    if eval_idx != -1:
        vector[eval_idx] = 1
    else:
        idx = np.random.randint(dim)
        vector[idx] = 1
    
    return vector


def generate_skill_cont(dim):

    while True:
        vector = np.random.normal(0, 1, dim)
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            break

    normalized_vector = vector / norm
    
    return normalized_vector


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


# def add_noise_to_skill(skill, noise_std=0.1):
#     noisy_skill = skill + np.random.normal(0, noise_std, skill.shape)
#     return normalize_vector(noisy_skill)


def add_noise_to_skill(skill, noise_scale, step):
    if step % 4 in [0, 1]:
        noisy_skill = skill + (noise_scale, -noise_scale)
    else:
        noisy_skill = skill + (-noise_scale, noise_scale)
    return normalize_vector(noisy_skill)