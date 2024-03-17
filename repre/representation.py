import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

import numpy as np

from repre.models import Psi
from repre.utils import generate_data

# Run the code
batch_size = 200
trajectory_length = 300
file_path = './repre/time_series_data.npy' # optional
data = generate_data(batch_size, trajectory_length, file_path)


for i in range(100):
    pass