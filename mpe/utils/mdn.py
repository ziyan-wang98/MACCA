import torch
import torch.nn as nn
from .util import init, get_clones


"""MDN modules."""

class Dynamic_MDN(nn.Module):
    def __init__(self, args, state_dim, action_dim, num_agent, agent_type_num=None, guassian_num=2, hidden_size = 256):
        super(Dynamic_MDN, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim +num_agent, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3 * guassian_num)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_ReLU]
        self.guassian_num = guassian_num
    def forward(self, state, action, agent_id, agent_type=None, sample=False):
        x = self.activate_func(self.fc1(torch.cat([state, action, agent_id], -1)))
        x = self.activate_func(self.fc2(x))
        x = self.fc3(x)
        pi, sigma, mu = torch.split(x, self.guassian_num, -1)
        pi = torch.softmax(pi, -1)
        sigma = torch.exp(sigma)
        sigma *= torch.ones_like(sigma) * 2.7
        if sample:
            return self.sample(pi, sigma, mu), mu, sigma, pi.log()
        return None, mu, sigma, pi.log()
    def sample(self, pi, sigma, mu):
        return torch.sum(pi * torch.normal(mu, sigma), -1)