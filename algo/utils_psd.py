import math
import torch
import os
import sys

import imageio
import numpy as np

def onehot2radius(state_batch, radius_dim):
    radius_onehot_batch = state_batch[:,-radius_dim:]
    radius_candidate = np.array([10,50,100])

    radius_batch = np.dot(radius_onehot_batch, radius_candidate)

    return radius_batch

