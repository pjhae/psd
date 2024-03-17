import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from torch.optim import Adam

import datetime
import numpy as np
import matplotlib.pyplot as plt

from repre.models import Psi
from repre.utils import generate_data, get_minibatch

# Device
device = torch.device("cuda")

# Tensorboard
writer = SummaryWriter('repre/runs/{}_representation'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

# Generate data
batch_size = 100000
trajectory_length = 200
file_path = './repre/time_series_data.npy' # optional
data = generate_data(batch_size, trajectory_length, file_path)
num_samples, len_trajectory, num_features = data.shape

# Psi
psi = Psi(num_features, latent_dim = 2).to(device)

# Period
L = 6

# Training loop
updates = 0
for i in range(50000):
    samples = get_minibatch(data, L=L, num_samples=128)
    total_loss, loss_max, loss_min, loss_const_1, loss_const_2 = psi.update_parameters(samples, L=L)

    writer.add_scalar('loss/total', total_loss, updates)
    writer.add_scalar('loss/max', loss_max, updates)
    writer.add_scalar('loss/min', loss_min, updates)
    writer.add_scalar('loss/const_L)', loss_const_1, updates)
    writer.add_scalar('loss/const_1', loss_const_2, updates)

    updates += 1

samples = get_minibatch(data, L=L, num_samples=3000)
minibatch_before, minibatch_before_prime, minibatch_after, minibatch_after_prime = samples
data = psi.forward_np(minibatch_before)

plt.figure(figsize=(12, 8))  # 그림 크기 조정
plt.plot(data[:, 0], data[:, 1], 'o')  # 'o'는 데이터 포인트를 동그란 마커로 표시합니다.

# 그래프 제목과 축 라벨 설정
plt.title('2D NumPy Data Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')

# # 플롯 표시
plt.show()
writer.add_figure('numpy_plot', plt.gcf())