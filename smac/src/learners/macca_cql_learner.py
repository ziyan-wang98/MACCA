import copy
from torch.distributions import Categorical
from torch.optim.rmsprop import RMSprop
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
from modules.critics.discrete_critic import DoubleMLPNetwork
import torch as th
from utils.rl_utils import build_td_lambda_targets, build_target_q, build_td_lambda_targets, build_q_lambda_targets
from torch.optim import Adam
from modules.mixers.qmix import QMixer
from utils.th_utils import get_parameters_num
import numpy as np
from .cq_learner import CQLearner
from modules.agents.mlp_agent import Reward_MLP
from modules.agents.mlp_agent import ActionDependentCausalEncoder
from utils.th_utils import check
import torch.nn.functional as F
import torch.nn as nn

from modules.agents.mlp_agent import Reward_MLP
from modules.agents.mlp_agent import ActionDependentCausalEncoder
from utils.th_utils import check

from envs.starcraft.smac_maps import get_map_params


class MACCA_CQLearner(CQLearner):
    def __init__(self, mac, scheme, logger, args):
        super(MACCA_CQLearner, self).__init__(mac, scheme, logger, args)
        self.args = args
        
        self.num_agents = self.args.n_agents
        self.n_agents = self.args.n_agents
        
        self.action_dim = self.args.n_actions
        
        self.joint_state_dim = self.args.state_shape
        
        self.continuous = self.args.continuous
        
        # TODO: What if the action space for each agent is different? Like 2s3z?
        self.joint_action_dim = self.action_dim * self.num_agents
        
        ## MACCA specific parameters
        self.r_lr = self.args.r_lr
        self.s_lr = self.args.s_lr
        
        self.sr_sparsity_coef = self.args.sr_sparsity_coef
        self.ar_sparsity_coef = self.args.ar_sparsity_coef
        self.temperature = self.args.temperature
        self.threshold_value = self.args.threshold_value
        self.opti_eps = self.args.opti_eps
        
        
        self.tpdv = dict(dtype=th.float32, device=self.args.device)
        self.tpdv_long = dict(dtype=th.long, device=self.args.device)
        
        
        self.use_gumbel = self.args.use_gumbel
        
        if self.use_gumbel:
            self.get_nnParam = self.get_Param_prob
            self.get_structure_param = self.get_structure_param_prob
            self.cal_sparsity_loss = self.cal_sparsity_loss_prob
        else:
            self.get_nnParam = self.get_Param_threshold
            self.get_nnParam_action = self.get_Param_threshold_action
            self.cal_sparsity_loss = self.cal_sparsity_loss_threshold
            self.get_structure_param = self.get_structure_param_threshold


        self.build_model(args)
        self.to_device(self.args.device)
        
    
    @staticmethod
    def get_Param_prob(shape):
        v_0 = (1. * np.ones(shape, np.float32)[..., np.newaxis])
        v_1 = (-1. * np.ones(shape, np.float32)[..., np.newaxis])
        #return torch.nn.Parameter(th.from_numpy(np.concatenate([v_0, v_1], -1)))
        return th.nn.Parameter(th.randn_like(th.from_numpy(np.concatenate([v_0, v_1], -1))))
    
    @staticmethod
    def get_Param_threshold(shape):
        v_0 = (1. * np.ones(shape, np.float32)[..., np.newaxis])
        v_1 = (-1. * np.ones(shape, np.float32)[..., np.newaxis])
        #return torch.nn.Parameter(torch.from_numpy(np.concatenate([v_0, v_1], -1)))
        return th.nn.Parameter(th.randn_like(th.from_numpy(np.concatenate([v_0, v_1], -1)))[..., 0])
    
    @staticmethod
    def get_Param_threshold_action(shape):
        v_0 = (1. * np.ones(shape, np.float32)[..., np.newaxis])
        v_1 = (-1. * np.ones(shape, np.float32)[..., np.newaxis])
        #return torch.nn.Parameter(torch.from_numpy(np.concatenate([v_0, v_1], -1)))
        return th.nn.Parameter(th.randn_like(th.from_numpy(np.concatenate([v_0, v_1], -1)))[..., 0])
    

    def build_model(self, args):
        self.causal_encoder = th.nn.ParameterDict({
            "S2R": self.get_nnParam([self.joint_state_dim,1]),
            "A2R": self.get_nnParam_action([self.joint_action_dim, 1]),
        })  
        
        # Initialize the action-dependent causal encoder
        self.a2r_action_dependent_causal_encoder = ActionDependentCausalEncoder(args, self.joint_action_dim, self.args.embed_dim, self.causal_encoder['A2R'].shape[0])
        
        # predict joint reward p(\bm r_t | \bm s_t, \bm a_t)
        self.reward_predictor = Reward_MLP(args, self.joint_state_dim, self.joint_action_dim, self.num_agents)
        
        self.causal_optimizer = th.optim.Adam([{'params': self.causal_encoder.parameters(), 'lr': self.s_lr},
                                               {'params': self.reward_predictor.parameters(), 'lr': self.r_lr},
                                               {'params': self.a2r_action_dependent_causal_encoder.parameters(), 'lr': self.s_lr}
                                               ],
                                                lr=self.r_lr, eps=self.opti_eps,
                                                weight_decay=0)
        self.create_sparsity_label()
    
    def to_device(self, device):
        self.causal_encoder.to(device)
        self.reward_predictor.to(device)
    
    
    def create_sparsity_label(self, ):
        self.s2r_label = th.ones([self.joint_state_dim, 1]).to(**self.tpdv_long)
        self.a2r_label = th.ones([self.joint_action_dim, 1]).to(**self.tpdv_long)
        
    def get_structure_param_prob(self, param, test):
        # TODO add temperature adjustment
        if test:
            return (param[..., 0] > param[..., 1]).float()
        else:
            # logits = th.softmax(param, -1).log()
            gumbel_samples = th.nn.functional.gumbel_softmax(param, self.temperature, hard=True)[..., 0]
            return gumbel_samples
    
    def get_structure_param_threshold(self, param, test):
         # TODO add temperature adjustment
        if test:
            return (th.abs(param) > self.threshold_value).float()
        else:
            # return param
            return th.ones_like(param) * param
        
        # return th.zeros_like(param)
        # return th.randn_like(param)
        
    def cal_sparsity_loss_prob(self, ):
        
        s2r_loss = F.cross_entropy(self.causal_encoder['S2R'].view(-1, 2), self.s2r_label.view(-1), reduction='none') 
        a2r_loss = F.cross_entropy(self.causal_encoder['A2R'].view(-1, 2), self.a2r_label.view(-1), reduction='none') 
        
        return s2r_loss.mean(), a2r_loss.mean()
    
    def cal_sparsity_loss_threshold(self, ):
       
        s2r_loss = th.abs(self.causal_encoder['S2R']).mean() 
        a2r_loss = th.abs(self.causal_encoder['A2R']).mean() 
    
        return s2r_loss, a2r_loss
    
    def cal_sparsity_rate(self, ):
        s2r_mask = self.get_structure_param(self.causal_encoder['S2R'], test=True)
        a2r_mask = self.get_structure_param(self.causal_encoder['A2R'], test=True)
        return s2r_mask.mean(), a2r_mask.mean()
    
    def causal_graph_update(self, sample, info=dict(
        Reward_loss=None,
        Sparsity_loss=None,
        Sparsity_rate=None,
    )):
        
        state = sample["state"] # (bs, ts, s_dim)
        if self.args.reorder_state:
            state = self.reorder_state_v2(state)
        shared_obs_batch_causal = state
        actions_batch_causal = sample["actions"]
        Rs_batch = sample["reward"]
        # Rs_batch = np.repeat(Rs_batch[:, :, np.newaxis], self.n_agents, axis=-2)  # (bs, ts, na, 1)
        type_id = [i for i in range(self.num_agents)]
        
        # ID for each agent
        agent_id = [i for i in range(self.num_agents)]
        agent_id_tensor = th.tensor(agent_id, dtype=th.long, device=self.args.device)
        agent_id_tensor = agent_id_tensor.unsqueeze(0).unsqueeze(0)  # Now its shape is [1, 1, 5]
        agent_id_tensor = agent_id_tensor.repeat(shared_obs_batch_causal.shape[0], shared_obs_batch_causal.shape[1], 1)
        
        shared_obs_batch_causal = check(shared_obs_batch_causal).to(**self.tpdv) # mini_batch_size, num_agent, joint_state_dim
        actions_batch_causal = check(actions_batch_causal).to(**self.tpdv) # mini_batch_size, num_agent, joint_action_dim
        Rs_batch = check(Rs_batch).to(**self.tpdv)  # mini_batch_size, num_agent, 1
        
        redistributed_reward = self.redistribute_reward(shared_obs_batch_causal, actions_batch_causal, info=dict(test=False, agent_id=agent_id_tensor, type_id=type_id)) # mini_batch_size, num_agent, 1

        
        # For log
        self.Rs_batch = Rs_batch
        self.Prd_rewards = redistributed_reward
        self.Prd_rewards_sum = redistributed_reward.sum(1)
        
        info.update({
            'Reward_loss': self.cal_reward_loss(redistributed_reward.sum(2), Rs_batch),
            'Sparsity_loss': self.cal_sparsity_loss(),
            'Sparsity_rate': self.cal_sparsity_rate()
        })
        
        # Upadate the causal graph
        s2r_loss, a2r_loss = info['Sparsity_loss']
        info['s2r_rate'] = info['Sparsity_rate'][0]
        info['a2r_rate'] = info['Sparsity_rate'][1]
        info['agent_id'] = agent_id_tensor
        info['s2r_loss'] = s2r_loss
        info['a2r_loss'] = a2r_loss
        info['Sparsity_loss'] = 0
        info['Sparsity_loss'] += self.sr_sparsity_coef * s2r_loss + self.ar_sparsity_coef * a2r_loss 
        total_loss = info['Reward_loss'] + info['Sparsity_loss']
        
        self.causal_optimizer.zero_grad()
        total_loss.backward()
        self.causal_optimizer.step()
        return info
    def reorder_state_v2(self, state):
        bs, ts = state.shape[0], state.shape[1]
        self.map_params = get_map_params(self.args.env_args['map_name'])
        self._agent_race = self.map_params["a_race"]
        self._bot_race = self.map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = self.map_params["unit_type_bits"]
        self.n_enemies = self.map_params["n_enemies"]
        
        
        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits
        # Get the sizes of ally_state and enemy_state
        self.size_ally = self.n_agents * nf_al
        self.size_enemy = self.n_enemies * nf_en

        state_repeated = state.unsqueeze(2).repeat(1, 1, self.num_agents, 1)  # Shape: (bs, ts, num_agents, state_dim)

        # Extract ally_state and enemy_state
        ally_state = state_repeated[:, :, :, :self.size_ally].view(bs, ts, self.n_agents, self.n_agents, -1)
        enemy_state = state_repeated[:, :, :, self.size_ally:self.size_ally + self.size_enemy]
        # import pdb; pdb.set_trace()
        # Reorder ally_state and last_action using list comprehension
        indexes = [list(range(agent_id, agent_id + 1)) + list(range(agent_id)) + list(range(agent_id + 1, self.n_agents)) for agent_id in range(self.n_agents)]

        # indexes = [list(range(agent_id, self.n_agents)) + list(range(agent_id)) for agent_id in range(self.n_agents)]
        ally_state_reordered = th.cat([ally_state[:, :, i, indexes[i]] for i in range(self.n_agents)], dim=2)

        if self.args.env_args['state_last_action']:
            last_action = state_repeated[:, :, :, self.size_ally + self.size_enemy:].view(bs, ts, self.n_agents, self.n_agents, -1)
            last_action_reordered = th.cat([last_action[:, :, i, indexes[i]] for i in range(self.n_agents)], dim=2)
        else:
            last_action_reordered = []
        # Concatenate the reordered states: ally_state, enemy_state, last_action
        reordered_states = th.cat((ally_state_reordered.view(bs, ts, self.n_agents, -1), 
                                    enemy_state, last_action_reordered.view(bs, ts, self.n_agents, -1)), dim=-1)

        return reordered_states
    def reorder_state(self, state):
        bs, ts = state.shape[0], state.shape[1]
        # Repeat the state for each agent
        state_repeated = np.repeat(state[:, :, np.newaxis, :].cpu(), self.num_agents, axis=-2)  # (bs, ts, num_agents, s_dim)
        reordered_states = np.zeros_like(state_repeated)  # (bs, ts, num_agents, s_dim)
        # Iterate through agents to reorder the states
        for agent_id in range(self.num_agents):
            for b in range(bs):
                for t in range(ts):
                    reordered_states[b, t, agent_id] = self.reorder_state_for_agent(state_repeated[b, t, agent_id], agent_id)
        return th.tensor(reordered_states, dtype=th.float32).to(self.args.device)
    
    def reorder_state_for_agent(self, state, agent_id):
        """Reorders a given state from the perspective of a specific agent."""
        
        self.map_params = get_map_params(self.args.env_args['map_name'])
        self._agent_race = self.map_params["a_race"]
        self._bot_race = self.map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = self.map_params["unit_type_bits"]
        self.n_enemies = self.map_params["n_enemies"]
        
        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits
        # Get the sizes of ally_state and enemy_state
        size_ally = self.n_agents * nf_al
        size_enemy = self.n_enemies * nf_en
        
        # Reshape state back to ally_state and enemy_state
        ally_state = state[:size_ally].reshape(self.n_agents, nf_al)
        enemy_state = state[size_ally:size_ally + size_enemy].reshape(self.n_enemies, nf_en)
        
        # If state_last_action is True, extract the last_action part
        if self.args.env_args['state_last_action']:
            last_action = state[size_ally + size_enemy:]
            size_last_action_per_agent = last_action.shape[0] // self.n_agents
            last_action = last_action.reshape(self.n_agents, size_last_action_per_agent)

            # Reorder last_action
            last_action_reordered = np.concatenate((last_action[agent_id:agent_id+1], 
                                                    last_action[:agent_id], 
                                                    last_action[agent_id+1:]), axis=0)
        else:
            last_action_reordered = []


        ally_state_reordered = np.concatenate((ally_state[agent_id:agent_id+1], 
                                                ally_state[:agent_id], 
                                                ally_state[agent_id+1:]), axis=0)

        # Reorder enemy_state (if necessary)
        # This part may vary depending on how you define the "perspective" of a specific agent
        enemy_state_reordered = enemy_state  # Here, we just keep the original order

        # Concatenate the reordered states: ally_state, enemy_state, last_action
        state_reordered = np.concatenate((ally_state_reordered.flatten(), 
                                      enemy_state_reordered.flatten(),
                                      last_action_reordered.flatten()), axis=0)
        
        return state_reordered
        
    def cal_reward_loss(self, redistributed_reward, reward_batch):
        return nn.functional.mse_loss(redistributed_reward, reward_batch)
    
    def redistribute_reward(self, state, action, info):
        test = info['test']
        agent_id = info['agent_id']
        # type_id = info['type_id']
        # state
        
  
        s2r_mask = self.get_structure_param(self.causal_encoder['S2R'], test=test) # joint_state_dim, num_agent
        a2r_mask = self.get_structure_param(self.causal_encoder['A2R'], test=test) # joint_action_dim, num_agent
        
        # s2r_mask[:,-self.num_agents:] = 1.0
        
        # convert states and actions into joint state and joint action
        state_tensor = state
        if not self.args.reorder_state:
            state_tensor = state_tensor.view(state.shape[0], state.shape[1], self.num_agents, -1).repeat(1, 1, 1, self.num_agents)
        if self.continuous:
            action_tensor = action
        else:
            action_tensor = self.get_onehot_for_action(action)
        
        # Embedding the joint action
        if self.args.use_action_embedding:
            action_tensor = action_tensor.to(self.args.device)
            mask_a2r_em = self.a2r_action_dependent_causal_encoder(action_tensor)
            a2r_mask = a2r_mask.transpose(0,1).expand_as(mask_a2r_em)
            a2r_mask = th.mul(a2r_mask, mask_a2r_em)
            
        return self.reward_predictor(
            th.mul((s2r_mask.transpose(0,1)), state_tensor), # state: batch_size, num_agent, joint_state_dim
            # th.mul((a2r_mask.transpose(0,1)), action_tensor), # action: batch_size, num_agent, joint_action_dim
            th.mul(a2r_mask, action_tensor), # action: batch_size, num_agent, joint_action_dim
            agent_id
            ) # reward: batch_size, num_agent , 1

    def get_onehot_for_action(self, action):
        # action (batch_size, time_size, num_agents, 1) into (batch_size, time_size, num_agents, action_dim)
        action_tensor = th.squeeze(action, -1)  # shape (batch_size, time_size, num_agents)
        action_tensor = th.tensor(action_tensor, dtype=th.int64, device=self.args.device)
        action_onehot = th.nn.functional.one_hot(action_tensor, num_classes=self.action_dim).float()  # shape (batch_size, time_size, num_agents, action_dim)
        return action_onehot.repeat(1, 1, 1, self.n_agents) # shape (batch_size, time_size, num_agents, action_dim * num_agents)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        
        self.loss_dict = self.causal_graph_update(batch)
        
        ## Causal graph update
        sample = batch
        team_reward = batch['reward'][:, :-1]
        state = batch['state']
        re_team_reward = np.repeat(team_reward.cpu().numpy()[:, :, np.newaxis], self.n_agents, axis=-2)  # (bs, ts, na, 1)

        
        # ID for each agent
        agent_id = [i for i in range(self.num_agents)]
        agent_id_tensor = th.tensor(agent_id, dtype=th.long, device=self.args.device)
        agent_id_tensor = agent_id_tensor.unsqueeze(0).unsqueeze(0)  # Now its shape is [1, 1, 5]
        agent_id_tensor = agent_id_tensor.repeat(state.shape[0], state.shape[1], 1)
        
        
        info = dict(test=False, agent_id = agent_id_tensor, type_id = None)
        if self.args.reorder_state:
            state = self.reorder_state_v2(state)
            state = th.tensor(state, dtype=th.float32).to(self.args.device)
        
        indiviudal_rewards = self.redistribute_reward(state, batch["actions"], info)
        
        
        
        
        # team_rewards = batch["reward"][:, :-1]
        
        # team_rewards = team_rewards.unsqueeze(2).expand(-1, -1, self.num_agents, -1)
        # import pdb; pdb.set_trace()
        team_rewards = indiviudal_rewards
        
        
        bs, ts, num_agents = team_rewards.shape[0], team_rewards.shape[1], team_rewards.shape[2]

        total_cql_loss = 0
        total_masked_td_error = 0
        
        for agent_id in range(num_agents):
            rewards = team_rewards[:, :, agent_id, :]
            # Get the relevant quantities
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]
        
            # Calculate estimated Q-Values
            self.mac.agent.train()
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time bs,ts,na,action

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions)[:, :, agent_id, :]

            # Calculate the Q-Values necessary for the target
            with th.no_grad():
                self.target_mac.agent.train()
                target_mac_out = []
                self.target_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)

                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            if self.args.use_sarsa:
                target_max_qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                self.args.cql_alpha=0.0
                self.args.global_cql_alpha=0.0
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])


            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

            # Mixer
            # chosen_action_qtotals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            chosen_action_qtotals = chosen_action_qvals

            td_error = (chosen_action_qtotals - targets.detach())
            td_error2 = 0.5 * td_error.pow(2)

            mask = mask.expand_as(td_error2)
            masked_td_error = td_error2 * mask
            
            total_masked_td_error += masked_td_error

            # important sampling for PER

            n_agents = mac_out.shape[2]
            n_actions = mac_out.shape[3]
    
            sample_actions_num = self.args.raw_sample_actions
            bs = actions.shape[0]
            ts = actions.shape[1]
            sample_enough =False
            repeat_avail_actions = th.repeat_interleave(avail_actions[:,:-1].unsqueeze(0),repeats=sample_actions_num,dim=0)#san,bs,ts,na,ad
            # while not sample_enough:
            total_random_actions = th.randint(low=0,high=n_actions,size=(sample_actions_num,bs,ts,n_agents,1)).to(self.args.device)#san,bs,ts,na,1
            chosen_if_avail = th.gather(repeat_avail_actions, dim=-1, index=total_random_actions).min(-2)[0]#san,bs,ts,1
            # chosen_if_avail = th.repeat_interleave(chosen_if_avail.unsqueeze(-1),repeats=n_agents,dim=-2)#san,bs,ts,na,1
            # total_random_actions = total_random_actions[]
            repeat_mac_out = th.repeat_interleave(mac_out[:,:-1].unsqueeze(0),repeats=sample_actions_num,dim=0)#san,bs,ts,na,ad
            random_chosen_action_qvals = th.gather(repeat_mac_out, dim=-1, index=total_random_actions).squeeze(-1)#san,bs,ts,na
            repeat_state = th.repeat_interleave(batch["state"][:, :-1].unsqueeze(0),repeats=sample_actions_num,dim=0)#san,bs,ts,sd
            random_chosen_action_qvals = random_chosen_action_qvals.view(bs*sample_actions_num,ts,-1)
            repeat_state = repeat_state.view((bs*sample_actions_num,ts,-1))
            random_chosen_action_qtotal = self.mixer(random_chosen_action_qvals,repeat_state).view(sample_actions_num,bs,ts,1)#san,bs,ts,1
            negative_sampling = th.logsumexp(random_chosen_action_qtotal*chosen_if_avail,dim=0)#bs,ts,1
            dataset_expec = chosen_action_qtotals#bs,ts,1
            cql_loss = self.args.global_cql_alpha * ((negative_sampling-dataset_expec)* mask).sum()/mask.sum()
            
            total_cql_loss += cql_loss
            

        L_td = total_masked_td_error.sum() / mask.sum()
        loss = total_cql_loss + L_td

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        
        if th.any(th.isnan(cql_loss)) or th.any(th.isnan(grad_norm)):
            print('there is nan!!!!!!!!!!!!')

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # self.logger.log_stat("q_taken_mean", (chosen_action_qtotals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qtotals * mask).sum().item()/(mask_elems), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems), t_env)
            self.logger.log_stat("q_local_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("cql_loss", (((negative_sampling-dataset_expec)* mask).sum().item()/mask_elems), t_env)
            self.logger.log_stat("negtive_sampling_mean", ((negative_sampling* mask).sum().item()/mask_elems), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    
