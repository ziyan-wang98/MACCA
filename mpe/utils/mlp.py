import torch
import torch.nn as nn
from .util import init, get_clones #, orthogonal_init

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x


# class Reward_MLP(nn.Module):
#     def __init__(self, args, state_dim, action_dim, num_agent):
#         super(Reward_MLP, self).__init__()

#         self._use_feature_normalization = args.use_feature_normalization
#         self._use_orthogonal = args.use_orthogonal
#         self._use_ReLU = args.use_ReLU
#         self._stacked_frames = args.stacked_frames
#         self._layer_N = args.layer_N
#         self.hidden_size = args.hidden_size

#         if self._use_feature_normalization:
#             self.feature_norm = nn.LayerNorm(state_dim + action_dim)

#         self.reward_mlp = MLPLayer(state_dim + action_dim, self.hidden_size,
#                               self._layer_N, self._use_orthogonal, self._use_ReLU)

#     def forward(self, state, action):
#         if self._use_feature_normalization:
#             x = self.feature_norm(torch.concat([state, action], -1))
#             x = self.reward_mlp(x)
#         else:
#             x = self.reward_mlp(torch.concat([state, action], -1))
#         return x

# class Reward_MLP(nn.Module):
#     def __init__(self, args, state_dim, action_dim, agent_num, agent_type_num=None):
#         super(Reward_MLP, self).__init__()
#         args.use_ReLU = True
#         # TODO add agent type
#         self.fc1 = nn.Linear(state_dim + action_dim + agent_num, args.hidden_dim)
#         self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
#         self.fc3 = nn.Linear(args.hidden_dim, 1)
#         self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_ReLU]

#     def forward(self, state, action, agent_id, agent_type=None):
#         x = self.activate_func(self.fc1(torch.cat([state, action, agent_id], -1)))
#         x = self.activate_func(self.fc2(x))
#         x = self.fc3(x) # num_agent, 1 | shared reward function
#         return x

class Reward_MLP(nn.Module):
    def __init__(self, args, state_dim, action_dim, agent_num, agent_type_num=None):
        super(Reward_MLP, self).__init__()
        args.use_ReLU = True
        self.paper_rew = getattr(args, 'paper_rew', False)
        in_dim = state_dim + action_dim + agent_num
        if self.paper_rew:
            # paper psi_r: 3 fully-connected layers of width 256 + a single-output layer
            W = 256
            self.fc1 = nn.Linear(in_dim, W)
            self.fc2 = nn.Linear(W, W)
            self.fc3 = nn.Linear(W, W)
            self.fc4 = nn.Linear(W, 1)
        else:
            self.fc1 = nn.Linear(in_dim, args.hidden_dim)
            self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fc33 = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fc4 = nn.Linear(args.hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_ReLU]

    def forward(self, state, action, agent_id, agent_type=None):
        x = self.activate_func(self.fc1(torch.cat([state, action, agent_id], -1)))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        if not self.paper_rew:
            x = self.activate_func(self.fc33(x))
        x = self.fc4(x)  # num_agent, 1 | shared reward function
        return x
