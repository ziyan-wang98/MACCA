import torch.nn as nn
import torch.nn.functional as F

import torch

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, discrete_action=True, policy_ln=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
            policy_ln (bool): use LayerNorm (input + hidden) instead of BatchNorm
        """
        super(MLPNetwork, self).__init__()
        self.policy_ln = policy_ln

        if policy_ln:
            self.in_fn = nn.LayerNorm(input_dim)
        else:
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        if policy_ln:
            self.ln1 = nn.LayerNorm(hidden_dim); self.ln2 = nn.LayerNorm(hidden_dim)
        self.nonlin = nonlin

        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        nx = self.in_fn(X)
        if self.policy_ln:
            h1 = self.nonlin(self.ln1(self.fc1(nx)))
            h2 = self.nonlin(self.ln2(self.fc2(h1)))
        else:
            h1 = self.nonlin(self.fc1(nx))
            h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))

        return out

class DoubleMLPNetwork(nn.Module):
    """
    Twin-Q MLP critic. With critic_ln=True it uses LayerNorm (input + after each hidden
    linear) instead of BatchNorm1d — the standard offline-RL stabilization trick that bounds
    Q extrapolation and prevents the early-peak-then-diverge failure on suboptimal data.
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, discrete_action=True, critic_ln=False):
        super(DoubleMLPNetwork, self).__init__()
        self.critic_ln = critic_ln

        if critic_ln:
            self.in_fn = nn.LayerNorm(input_dim)
        else:
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.fc4 = nn.Linear(input_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, out_dim)

        if critic_ln:
            self.ln1 = nn.LayerNorm(hidden_dim); self.ln2 = nn.LayerNorm(hidden_dim)
            self.ln4 = nn.LayerNorm(hidden_dim); self.ln5 = nn.LayerNorm(hidden_dim)

        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.fc6.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def _branch(self, X, which):
        nx = self.in_fn(X)
        if which == 1:
            fa, fb, fc = self.fc1, self.fc2, self.fc3
            la, lb = (self.ln1, self.ln2) if self.critic_ln else (None, None)
        else:
            fa, fb, fc = self.fc4, self.fc5, self.fc6
            la, lb = (self.ln4, self.ln5) if self.critic_ln else (None, None)
        if self.critic_ln:
            h1 = self.nonlin(la(fa(nx)))
            h2 = self.nonlin(lb(fb(h1)))
        else:
            h1 = self.nonlin(fa(nx))
            h2 = self.nonlin(fb(h1))
        return self.out_fn(fc(h2))

    def forward(self, X):
        return self._branch(X, 1), self._branch(X, 2)

    def Q1(self, X):
        return self._branch(X, 1)
