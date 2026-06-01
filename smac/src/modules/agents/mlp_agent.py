import torch.nn as nn
import torch.nn.functional as F
import torch

class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)

    def init_hidden(self):
        return None

    def forward(self, inputs, h=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        if self.agent_return_logits:
            actions = self.fc3(x)
        else:
            actions = F.tanh(self.fc3(x))
        return actions
    

class Reward_MLP(nn.Module):
    def __init__(self, args, state_dim, action_dim, agent_num, agent_type_num=None):
        super(Reward_MLP, self).__init__()
        self.agent_num = agent_num
        self.embedding = nn.Embedding(agent_num, args.embed_dim)
        self.fc1 = nn.Linear(state_dim + action_dim + args.embed_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc33 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][True]

    def forward(self, state, action, agent_id, agent_type=None):
        agent_id_embed = self.embedding(agent_id)
        x = self.activate_func(self.fc1(torch.cat([state, action, agent_id_embed], -1)))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        x = self.activate_func(self.fc33(x))
        x = self.fc4(x)  # num_agent, 1 | shared reward function
        return x
    
    
class ActionDependentCausalEncoder(nn.Module):
    def __init__(self, args, action_dim, hidden_dim, output_dim):
        super(ActionDependentCausalEncoder, self).__init__()
        self.device = args.device
        self.fcn = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  
            nn.Sigmoid() 
        ).to(self.device)
        
    def forward(self, joint_actions):
        original_shape = (joint_actions.shape)

        # Reshape to fit the network
        joint_actions = joint_actions.view(-1, joint_actions.shape[-1])
        joint_actions = joint_actions.to(self.device)
        output = self.fcn(joint_actions)

        # Reshape back
        output = output.view(*original_shape[:-1], -1)
        # [batch_size, episode_length, num_agents, output_dim]
        return output

