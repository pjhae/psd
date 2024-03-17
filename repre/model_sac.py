import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Psi(nn.Module):
    def __init__(self, num_inputs, args):
        super(Psi, self).__init__()
        
        self.lr = args.lr
        self.skill_dim = args.skill_dim
        self.hidden_dim = args.hidden_size
        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        # Psi architecture
        self.linear1 = nn.Linear(num_inputs, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.skill_dim)

        self.apply(weights_init_)

        self.optimizer = torch.optim.Adam(self.parameters(), lr= self.lr )

    def forward(self, state):
        
        state = torch.from_numpy(state).float().to(self.device)
        state = state[:, :-self.skill_dim]

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

    def forward_np(self, state):
        
        state = torch.from_numpy(state).float().to(self.device)
        state = state[:-self.skill_dim]

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1.detach().cpu().numpy()
