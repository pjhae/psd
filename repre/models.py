import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import numpy as np
import time

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Psi(nn.Module):
    def __init__(self, num_inputs, latent_dim):
        super(Psi, self).__init__()
        
        self.lr = 0.0001
        self.skill_dim = latent_dim
        self.hidden_dim = 512
        self.device = torch.device("cuda")
        
        # Psi architecture
        self.linear1 = nn.Linear(num_inputs, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.skill_dim)

        self.apply(weights_init_)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr )

    def forward(self, state):
        
        state = torch.from_numpy(state).float().to(self.device)
        # state = state[:, :-self.skill_dim]

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

    def forward_np(self, state):
        
        state = torch.from_numpy(state).float().to(self.device)
        # state = state[:-self.skill_dim]
        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1.detach().cpu().numpy()

    def update_parameters_original(self, memory, batch_size, lambda_value, epsilon=1e-3):
        state_batch, _,  _, next_state_batch, _ = memory.sample(batch_size=batch_size)
        
        z_batch = state_batch[:, -self.skill_dim:]

        psi_s = self.forward(state_batch)
        psi_next_s = self.forward(next_state_batch)
        
        loss = -(psi_next_s - psi_s).mul(torch.from_numpy(z_batch).to(self.device).detach()).sum(1) - lambda_value.detach() * torch.min(torch.tensor(epsilon).detach(), 1 - (psi_s - psi_next_s).pow(2).sum(1))

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean().item()
    
    def update_parameters(self, samples, args):

        L = args.period
        lambda_value = 1
        epsilon=1e-5

        minibatch_before, minibatch_before_prime, minibatch_after, minibatch_after_prime = samples
        
        psi_before = self.forward(minibatch_before)
        psi_before_prime = self.forward(minibatch_before_prime)
        psi_after = self.forward(minibatch_after)
        psi_after_prime = self.forward(minibatch_after_prime)

        loss_max = -torch.norm(psi_after-psi_before, p=2, dim=-1)
        loss_min = torch.norm((psi_after+psi_before)/2, p=2, dim=-1)
        loss_const_1 = -lambda_value * torch.min(torch.tensor(epsilon).detach(), L - torch.norm(psi_after-psi_before, p=2, dim=-1))
        loss_const_2 = -lambda_value * torch.min(torch.tensor(epsilon).detach(), L*np.sin(np.pi/(2*L)) - torch.norm(psi_before_prime-psi_before, p=2, dim=-1))
        loss = loss_max + loss_min + loss_const_1 + loss_const_2

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean().item(), loss_max.mean().item(), loss_min.mean().item(), loss_const_1.mean().item(), loss_const_2.mean().item()