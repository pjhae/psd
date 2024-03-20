import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

from repre.models import Psi
from repre.utils import generate_data, get_minibatch, get_evalbatch, plot_graph

# for argument
from arguments import parse_args
args = parse_args()

# Device
device = torch.device("cuda")

# Tensorboard
writer = SummaryWriter('repre/runs/{}_representation'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

# Generate data
data_size = args.data_size
trajectory_length = args.traj_length
file_path = './repre/time_series_data.npy' # optional
data = generate_data(data_size, trajectory_length, file_path)
num_samples, len_trajectory, num_features = data.shape

# Psi
psi = Psi(num_features, args.latent_dim).to(device)

# Training loop
updates = 0

for epoch in range(args.num_epoch):

    samples = get_minibatch(data, args.batch_size)
    total_loss, loss_max, loss_min, loss_const_1, loss_const_2 = psi.update_parameters(samples, args)

    writer.add_scalar('loss/total', total_loss, updates)
    writer.add_scalar('loss/max', loss_max, updates)
    writer.add_scalar('loss/min', loss_min, updates)
    writer.add_scalar('loss/const_L)', loss_const_1, updates)
    writer.add_scalar('loss/const_1', loss_const_2, updates)

    if epoch % args.num_epoch_for_eval == 0:
        minibatch, minibatch_eval = get_evalbatch(data, args.eval_batch_size, args)
        encoded_data = psi.forward_np(minibatch)
        encoded_data_eval = psi.forward_np(minibatch_eval)

        plot_graph(encoded_data, encoded_data_eval, args.latent_dim, writer, updates)

    updates += 1






    