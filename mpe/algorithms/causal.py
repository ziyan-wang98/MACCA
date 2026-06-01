import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from utils.mlp import Reward_MLP
from utils.mdn import Dynamic_MDN
from utils.util import check

def _t2n(x):
    return x.detach().cpu().numpy()


class CausalAgent(object):
    def __init__(self,
                 args,
                 device=torch.device("cpu")):
        self.device = device

        self.r_lr = args.r_lr
        self.d_lr = args.d_lr
        self.s_lr = args.s_lr
        
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.block_causal_structure = False
        self._use_causal_reward_assignment = args.use_causal
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv_long = dict(dtype=torch.long, device=device)
        
        self.ss_sparsity_coef = args.ss_sparsity_coef
        self.ss_sparsity_coef_aux = args.ss_sparsity_coef_aux
        self.as_sparsity_coef = args.as_sparsity_coef
        self.sr_sparsity_coef = args.sr_sparsity_coef
        self.ar_sparsity_coef = args.ar_sparsity_coef
        self.temperature = args.temperature


        # * self.state_dim: state dim of single agent, self.action_dim: action dim of single agent
        # * self.joint_state_dim: joint state dim, self.joint_action_dim: joint action dim
        # * self.num_agents: number of agents
        # * self.observation_space: observation space of whole world
        
       
        if self.block_causal_structure:
            #TODO: add block state.
            raise NotImplementedError
            self.num_agents = args.num_agents
            self.state_dim = (args.shared_state_dim // args.num_agents)
            self.action_dim = args.single_action_dim
            self.share_obs_space = args.share_obs_space
            
            self.joint_state_dim = self.state_dim * self.num_agents
            self.joint_action_dim = self.action_dim * self.num_agents
        else:
            self.num_agents = args.num_agents
            self.action_dim = args.single_action_dim
            self.joint_state_dim =  args.shared_state_dim
            self.joint_action_dim = self.action_dim * self.num_agents
        

        self.use_gumbel = False
        if self.use_gumbel:
            self.get_nnParam = self.get_Param_prob
            self.get_structure_param = self.get_structure_param_prob
            self.cal_sparsity_loss = self.cal_sparsity_loss_prob
        else:
            self.get_nnParam = self.get_Param_threshold
            # self.get_nnParam_action = self.get_Param_threshold_action
            self.get_nnParam_action = self.get_Param_threshold
            self.cal_sparsity_loss = self.cal_sparsity_loss_threshold
            self.get_structure_param = self.get_structure_param_threshold
            self.threshold_value =0.1
        self.dynamic_mask = getattr(args, 'dynamic_mask', False)
        self.build_model(args)
        self.to_device(self.device)
        
        
        
    @staticmethod
    def get_Param_prob(shape):
        v_0 = (1. * np.ones(shape, np.float32)[..., np.newaxis])
        v_1 = (-1. * np.ones(shape, np.float32)[..., np.newaxis])
        #return torch.nn.Parameter(torch.from_numpy(np.concatenate([v_0, v_1], -1)))
        return torch.nn.Parameter(torch.randn_like(torch.from_numpy(np.concatenate([v_0, v_1], -1))))
    

    # @staticmethod
    # def get_Param_threshold(shape):
    #     return torch.nn.Parameter(torch.randn(shape))
    

        
    @staticmethod
    def get_Param_threshold(shape):
        v_0 = (1. * np.ones(shape, np.float32)[..., np.newaxis])
        v_1 = (-1. * np.ones(shape, np.float32)[..., np.newaxis])
        #return torch.nn.Parameter(torch.from_numpy(np.concatenate([v_0, v_1], -1)))
        return torch.nn.Parameter(torch.randn_like(torch.from_numpy(np.concatenate([v_0, v_1], -1)))[..., 0])
    
    @staticmethod
    def get_Param_threshold_action(shape):
        v_0 = (1. * np.ones(shape, np.float32)[..., np.newaxis])
        v_1 = (-1. * np.ones(shape, np.float32)[..., np.newaxis])
        #return torch.nn.Parameter(torch.from_numpy(np.concatenate([v_0, v_1], -1)))
        return torch.nn.Parameter(torch.randn_like(torch.from_numpy(np.concatenate([v_0, v_1], -1)))[..., 0])

    def build_model(self, args):
        # build causal structure
        
        if self.block_causal_structure:
            self.causal_encoder = torch.nn.ModuleDict({
            "S2S": torch.nn.ParameterList([
                self.get_nnParam([self.state_dim, self.state_dim])
                for _ in range(self.num_agents * self.num_agents )
            ]),
            "A2S": torch.nn.ParameterList([
                self.get_nnParam([self.action_dim, self.state_dim])
                for _ in range(self.num_agents  * self.num_agents )
            ]),
            "S2R": torch.nn.ParameterList([
                self.get_nnParam([self.state_dim, 1])
                for _ in range(self.num_agents  * self.num_agents)
            ]),
            "A2R": torch.nn.ParameterList([
                self.get_nnParam_action([self.action_dim, 1])
                for _ in range(self.num_agents  * self.num_agents)
            ])
            })
        else:
            self.causal_encoder = torch.nn.ParameterDict({
                "S2S": self.get_nnParam([self.joint_state_dim, self.joint_state_dim]),
                "A2S": self.get_nnParam([self.joint_action_dim, self.joint_state_dim]),
                "S2R": self.get_nnParam([self.joint_state_dim,1]),
                "A2R": self.get_nnParam_action([self.joint_action_dim, 1]),
            })  

        # predict joint reward p(\bm r_t | \bm s_t, \bm a_t)
        self.reward_predictor = Reward_MLP(args, self.joint_state_dim, self.joint_action_dim, self.num_agents)

        # predict joint state p(\bm s_{t+1} | \bm s_t, \bm a_t)
        self.dynamic_module = Dynamic_MDN(args, self.joint_state_dim, self.joint_action_dim, self.num_agents)

        # paper's dynamic causal-structure predictor psi_g(s_t, a_t, i) (Eq 3): per-sample, per-agent
        # masks over the joint state/action dims. Opt-in via --dynamic_mask; default keeps the static params.
        if self.dynamic_mask:
            g_in = self.joint_state_dim + self.joint_action_dim + self.num_agents
            self.g_s2r = nn.Sequential(
                nn.Linear(g_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, self.joint_state_dim))
            self.g_a2r = nn.Sequential(
                nn.Linear(g_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, self.joint_action_dim))
            self._last_mask_s = None
            self._last_mask_a = None

        # optimizer
        param_groups = [{'params': self.causal_encoder.parameters(), 'lr': self.s_lr},
                        {'params': self.reward_predictor.parameters(), 'lr': self.r_lr},
                        {'params': self.dynamic_module.parameters(), 'lr': self.d_lr}]
        if self.dynamic_mask:
            param_groups.append({'params': list(self.g_s2r.parameters()) + list(self.g_a2r.parameters()), 'lr': self.s_lr})
        self.causal_optimizer = torch.optim.Adam(param_groups,
                                                lr=self.r_lr, eps=self.opti_eps,
                                                weight_decay=1e-5)
       
        # create sparsity label
        self.create_sparsity_label()
        
        
        # # lambda for an L2 reg coefficient that decays over time
        # l2_decay = lambda epoch: 1 / (1 + 0.01 * epoch)

        # # build a LambdaLR scheduler bound to the optimizer
        # self.scheduler = LambdaLR(self.causal_optimizer, lr_lambda=l2_decay)
        self.scheduler = None
        

    def to_device(self, device):
        self.causal_encoder.to(device)
        self.reward_predictor.to(device)
        self.dynamic_module.to(device)
        if getattr(self, 'dynamic_mask', False):
            self.g_s2r.to(device)
            self.g_a2r.to(device)
    
    
    def create_sparsity_label(self, ):
        self.s2r_label = torch.ones([self.joint_state_dim, 1]).to(**self.tpdv_long)
        # self.a2r_label = torch.ones([self.joint_action_dim, 1]).to(**self.tpdv_long)
        # self.s2r_label = torch.zeros([self.joint_state_dim, 1]).to(**self.tpdv_long)
        self.a2r_label = torch.zeros([self.joint_action_dim, 1]).to(**self.tpdv_long)
        self.s2s_label = torch.ones([self.joint_state_dim, self.joint_state_dim]).to(**self.tpdv_long)
        self.a2s_label = torch.ones([self.joint_action_dim, self.joint_state_dim]).to(**self.tpdv_long)
        self.s2s_mask = torch.eye(self.joint_state_dim).to(**self.tpdv)

    def get_structure_param_prob(self, param, test):
         # TODO add temperature adjustment
        if test:
            return (param[..., 0] > param[..., 1]).float()
        else:
            # logits = torch.softmax(param, -1).log()
            gumbel_samples = torch.nn.functional.gumbel_softmax(param, self.temperature, hard=True)[..., 0]
            return gumbel_samples
        
        # return torch.ones_like(param[..., 0])
    def get_structure_param_threshold(self, param, test):
         # TODO add temperature adjustment
        if test:
            return (torch.abs(param) > self.threshold_value).float()
        else:
            # return param
            return torch.ones_like(param) * param
        # return torch.zeros_like(param)

        # return torch.randn_like(param)
        
    def get_causal_structure(self, ):
        S2R, A2R = self.get_structure_param_threshold(self.causal_encoder['S2R'], test=False), \
            self.get_structure_param_threshold(self.causal_encoder['A2R'], test=False)
        causal_structure = {
            "S2R": S2R,
            "A2R": A2R,
        }
        return causal_structure    
    
    
    def cal_sparsity_loss_prob(self, ):
       
        s2s_loss = F.cross_entropy(self.causal_encoder['S2S'].view(-1, 2), self.s2s_label.view(-1), reduction='none') 
        s2s_loss_aux = s2s_loss * self.s2s_mask.view(-1)
        s2s_loss = s2s_loss * ((1 - self.s2s_mask).view(-1))
        a2s_loss = F.cross_entropy(self.causal_encoder['A2S'].view(-1, 2), self.a2s_label.view(-1), reduction='none') 
        s2r_loss = F.cross_entropy(self.causal_encoder['S2R'].view(-1, 2), self.s2r_label.view(-1), reduction='none') 
        a2r_loss = F.cross_entropy(self.causal_encoder['A2R'].view(-1, 2), self.a2r_label.view(-1), reduction='none') 
        
       
        
        return s2s_loss.sum() / (self.joint_state_dim * self.joint_state_dim - self.joint_state_dim), \
            s2s_loss_aux.sum() / self.joint_state_dim , a2s_loss.mean(), s2r_loss.mean(), a2r_loss.mean()
    

    def cal_sparsity_loss_threshold(self, ):
        if getattr(self, 'dynamic_mask', False):
            zero = torch.zeros((), **self.tpdv)
            s2r_loss = self._last_mask_s.abs().mean() if self._last_mask_s is not None else zero
            a2r_loss = self._last_mask_a.abs().mean() if self._last_mask_a is not None else zero
            return zero, zero, zero, s2r_loss, a2r_loss

        s2s_loss = torch.abs(self.causal_encoder['S2S']).view(-1)
        s2s_loss_aux = s2s_loss * self.s2s_mask.view(-1)
        s2s_loss = s2s_loss * ((1 - self.s2s_mask).view(-1))
        a2s_loss = torch.abs(self.causal_encoder['A2S']).mean()
        s2r_loss = torch.abs(self.causal_encoder['S2R']).mean()
        a2r_loss = torch.abs(self.causal_encoder['A2R']).mean()
        
       
        
        return s2s_loss.sum() / (self.joint_state_dim * self.joint_state_dim - self.joint_state_dim), \
            s2s_loss_aux.sum() / self.joint_state_dim , a2s_loss, s2r_loss, a2r_loss
    

    def cal_dynamic_loss(self, mu, sigma, logpi, z):

        
        # mu,sigma,logpi: [mini_batch_size, joint_state_dim, num_guassian]
        # z : [mini_batch_size, joint_state_dim]
        
        v = logpi + self.lognormal(z, mu, sigma)
        v = torch.logsumexp(v, -1, keepdims=True)
        acc = (((mu * logpi).sum(-1) - z).abs() / z.abs()).mean()
        
        return -torch.mean(v), acc

    def lognormal(self, z, mean, sigma):
        logSqrtTwoPI = torch.log(torch.sqrt(torch.tensor(2.0 * np.pi)))
        return -0.5 * ((z.unsqueeze(-1) - mean) / sigma) ** 2 - sigma.log() - logSqrtTwoPI

    def cal_sparsity_rate(self, ):
        if getattr(self, 'dynamic_mask', False):
            zero = torch.zeros((), **self.tpdv)
            s2r_rate = (self._last_mask_s > 0.5).float().mean() if self._last_mask_s is not None else zero
            a2r_rate = (self._last_mask_a > 0.5).float().mean() if self._last_mask_a is not None else zero
            return zero, zero, zero, s2r_rate, a2r_rate
        # import pdb; pdb.set_trace()
        s2s_mask = self.get_structure_param(self.causal_encoder['S2S'], test=True)
        s2s_mask_aux = s2s_mask * self.s2s_mask
        s2s_mask = s2s_mask * (1-self.s2s_mask)
        a2s_mask = self.get_structure_param(self.causal_encoder['A2S'], test=True)
        s2r_mask = self.get_structure_param(self.causal_encoder['S2R'], test=True)
        a2r_mask = self.get_structure_param(self.causal_encoder['A2R'], test=True)
        return s2s_mask.sum() / (s2s_mask.shape[0] * s2s_mask.shape[0] - s2s_mask.shape[0]), \
            s2s_mask_aux.sum() / self.s2s_mask.sum(), a2s_mask.mean(), s2r_mask.mean(), a2r_mask.mean()

    def get_compact_obs(self, share_obs):
        
        # import pdb; pdb.set_trace()
        # get s2pi
        s2s_mask = self.get_structure_param(self.causal_encoder['S2S'], test=True)
        s2s_loss_aux = s2s_mask * self.s2s_mask
        s2r_mask = self.get_structure_param(self.causal_encoder['S2R'], test=True)
        
        a2r_mask = self.get_structure_param(self.causal_encoder['A2R'], test=True)
        a2s_mask = self.get_structure_param(self.causal_encoder['A2S'], test=True)
        
    
        # s2r_mask_expanded = s2r_mask.unsqueeze(-1).unsqueeze(-1)

        # # Use torch.cat() to concatenate s2r_mask_expanded and s2s_mask along the last dimension
        # mask_concat = torch.cat((s2r_mask_expanded, s2s_mask.unsqueeze(-1)), dim=-1)

        # # Use torch.max() to take the maximum value along the last dimension
        # comp_mask = torch.max(mask_concat, dim=-1)[0]

        # # Use torch.bmm() to multiply share_obs and comp_mask
        # compact_obs = torch.bmm(share_obs, comp_mask)
                

        # s2s_mask_expanded = self.get_structure_param(self.causal_encoder['S2S'], test=True).unsqueeze(-1) # state, state, 1
        # s2r_mask_expanded = self.get_structure_param(self.causal_encoder['S2R'], test=True).unsqueeze(0) # 1, state, num_agent
        # torch.max(torch.concat(\
        #     self.get_structure_param(self.causal_encoder['S2R'], test=True).unsqueeze(1), \
        #         s2s_mask_expanded * s2r_mask_expanded))[0] # state, state, num_agent

        comp_mask = torch.stack([\
            torch.max(torch.concat((s2r_mask[:, i:i+1], s2r_mask[:, i].unsqueeze(0) * s2s_mask), -1), 1)[0] for i in range(self.num_agents)])       
        
        # mask: num_agent, state_dim
        # obs: num_agent, state_dim
        share_obs_tensor =check(share_obs).to(**self.tpdv)
        
        # import pdb; pdb.set_trace()
        
        compact_obs = torch.mul(share_obs_tensor, comp_mask)
        
        return share_obs
        
        return _t2n(compact_obs)
    
    
    def causal_graph_update_gt(self, sample, info=dict(
        Reward_loss=None,
        Sparsity_loss=None,
        Dynamic_loss=None,
        Sparsity_rate=None,
    )):
        shared_obs_batch_causal, actions_batch_causal, next_share_obs_batch_causal, Rs_batch, gt_Rs_batch, agent_i, agent_id, type_id = sample


        shared_obs_batch_causal = check(shared_obs_batch_causal).to(**self.tpdv) # mini_batch_size, num_agent, joint_state_dim
        actions_batch_causal = check(actions_batch_causal).to(**self.tpdv)
        # remove the dimension of the agents
        # import pdb; pdb.set_trace()
        next_share_obs_batch_causal = check(next_share_obs_batch_causal[:, 0]).to(**self.tpdv)
        
        Rs_batch = check(Rs_batch).to(**self.tpdv)
        gt_Rs_batch = check(gt_Rs_batch).to(**self.tpdv)
        redistributed_reward = self.redistribute_reward(shared_obs_batch_causal, actions_batch_causal, info=dict(test=False, agent_id=agent_id, type_id=type_id)) # mini_batch_size, num_agent, 1
        # next_share_obs_batch_prediction = self.predict_next_state(shared_obs_batch_causal, actions_batch_causal, agent_id, type_id)
        
        # For log
        self.Rs_batch = Rs_batch
        self.gt_Rs_batch = gt_Rs_batch
        self.Prd_rewards = redistributed_reward
        self.Prd_rewards_sum = redistributed_reward.sum(1)
        
        info.update({
            # 'Reward_loss': self.cal_reward_loss(redistributed_reward.sum(1), Rs_batch),
            'Reward_loss': self.cal_reward_loss(redistributed_reward[:,agent_i], gt_Rs_batch),
            # 'Reward_loss': self.cal_reward_loss(redistributed_reward[:,0], Rs_batch),
            'Sparsity_loss': self.cal_sparsity_loss(),
            # 'Dynamic_loss': self.cal_dynamic_loss(*next_share_obs_batch_prediction[1:], next_share_obs_batch_causal),
            'Sparsity_rate': self.cal_sparsity_rate()
        })


        # TODO optimization
        s2s_loss, s2s_loss_aux, a2s_loss, s2r_loss, a2r_loss = info['Sparsity_loss']
        
        info['s2s_loss'] = s2s_loss
        info['s2s_loss_aux'] = s2s_loss_aux
        info['a2s_loss'] = a2s_loss
        info['s2r_loss'] = s2r_loss
        info['a2r_loss'] = a2r_loss
        info['Sparsity_loss'] = 0
        # info['Sparsity_loss'] += self.ss_sparsity_coef * s2s_loss + self.ss_sparsity_coef_aux * s2s_loss_aux+ self.as_sparsity_coef * a2s_loss 
        info['Sparsity_loss'] += self.sr_sparsity_coef * s2r_loss + self.ar_sparsity_coef * a2r_loss 

        total_loss = info['Reward_loss'] + info['Sparsity_loss'] # +  info['Dynamic_loss'][0]
        #total_loss = info['Reward_loss'] + info['Sparsity_loss'] +  info['Dynamic_loss'][0]
        
        # import pdb; pdb.set_trace()
        
        self.causal_optimizer.zero_grad()
        total_loss.backward()
        self.causal_optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return info
    
    
    
    def causal_graph_update(self, sample, info=dict(
        Reward_loss=None,
        Sparsity_loss=None,
        Dynamic_loss=None,
        Sparsity_rate=None,
    )):
        shared_obs_batch_causal, actions_batch_causal, next_share_obs_batch_causal, Rs_batch, agent_id, type_id = sample


        shared_obs_batch_causal = check(shared_obs_batch_causal).to(**self.tpdv) # mini_batch_size, num_agent, joint_state_dim
        actions_batch_causal = check(actions_batch_causal).to(**self.tpdv)
        # remove the dimension of the agents
        # import pdb; pdb.set_trace()
        next_share_obs_batch_causal = check(next_share_obs_batch_causal[:, 0]).to(**self.tpdv)
        
        Rs_batch = check(Rs_batch).to(**self.tpdv)
        redistributed_reward = self.redistribute_reward(shared_obs_batch_causal, actions_batch_causal, info=dict(test=False, agent_id=agent_id, type_id=type_id)) # mini_batch_size, num_agent, 1
        # next_share_obs_batch_prediction = self.predict_next_state(shared_obs_batch_causal, actions_batch_causal, agent_id, type_id)
        
        # For log
        self.Rs_batch = Rs_batch
        self.Prd_rewards = redistributed_reward
        self.Prd_rewards_sum = redistributed_reward.sum(1)
        
        info.update({
            'Reward_loss': self.cal_reward_loss(redistributed_reward.sum(1), Rs_batch),
            # 'Reward_loss': self.cal_reward_loss(redistributed_reward[:,0], Rs_batch),
            'Sparsity_loss': self.cal_sparsity_loss(),
            # 'Dynamic_loss': self.cal_dynamic_loss(*next_share_obs_batch_prediction[1:], next_share_obs_batch_causal),
            'Sparsity_rate': self.cal_sparsity_rate()
        })


        # TODO optimization
        s2s_loss, s2s_loss_aux, a2s_loss, s2r_loss, a2r_loss = info['Sparsity_loss']
        
        info['s2s_loss'] = s2s_loss
        info['s2s_loss_aux'] = s2s_loss_aux
        info['a2s_loss'] = a2s_loss
        info['s2r_loss'] = s2r_loss
        info['a2r_loss'] = a2r_loss
        info['Sparsity_loss'] = 0
        # info['Sparsity_loss'] += self.ss_sparsity_coef * s2s_loss + self.ss_sparsity_coef_aux * s2s_loss_aux+ self.as_sparsity_coef * a2s_loss 
        info['Sparsity_loss'] += self.sr_sparsity_coef * s2r_loss + self.ar_sparsity_coef * a2r_loss 

        total_loss = info['Reward_loss'] + info['Sparsity_loss'] # +  info['Dynamic_loss'][0]
        #total_loss = info['Reward_loss'] + info['Sparsity_loss'] +  info['Dynamic_loss'][0]
        
        # import pdb; pdb.set_trace()
        
        self.causal_optimizer.zero_grad()
        total_loss.backward()
        self.causal_optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return info
    
    def predict_next_state(self, state, action, agent_id, agent_type=None):
        
        # state (min_batch_size, num_agents, state_dim)
        # action (mini_batch_size, num_agent, 1)
        s2s_mask = self.get_structure_param(self.causal_encoder['S2S'], test=False) # joint_state_dim_old, joint_state_dim_new
        a2s_mask = self.get_structure_param(self.causal_encoder['A2S'], test=False) # joint_action_dim, joint_state_dim_new 
       
        # convert states and actions into joint state and joint action
        # state of (n_rollout_thread or batch_size, num_agent, joint_state_dim) -> joint state of (batch_size, joint_state_dim)
        state_tensor = torch.tensor(state[:, 0], dtype=torch.float32, device=self.device)
        
        #state_tensor = torch.from_numpy(state[:, 0]).to(self.device).clone().detach()

        continuous = True
        if continuous:
            action_tensor = action[:, 0]
        else:
            # action of (n_rollout_thread or batch_size, num_agent, 1) -> joint state of (batch_size, joint_action_dim)
            action_tensor = self.get_onehot_for_action(action).view(-1, self.joint_action_dim) # action: batch_size, joint_action_dim   
        agent_id_input = torch.zeros(agent_id.shape[0], s2s_mask.shape[0], self.num_agents)

            
        return self.dynamic_module(
            torch.mul(s2s_mask, torch.unsqueeze(state_tensor, -1)).transpose(1, 2), # state: batch_size, joint_state_dim_new, joint_state_dim_old
            torch.mul(a2s_mask, torch.unsqueeze(action_tensor, -1)).transpose(1, 2),  # action: batch_size, num_agent, joint_action_dim
            agent_id_input, agent_type
            ) # next state: batch_size, joint_state_dim_new , 1



    def cal_reward_loss(self, redistributed_reward, reward_batch):
        return nn.functional.mse_loss(redistributed_reward, reward_batch)

    def get_critic_input(self,share_obs_batch):
        if self._use_causal_reward_assignment:
            print("Generate compact share obs batch")
            return self.get_compact_obs(share_obs_batch)
        else:
            return share_obs_batch
    
    def ppo_update(self, sample, update_actor=True):
        
        ori_sample, causal_sample = sample
        
        self.loss_dict = self.causal_graph_update(sample)

        share_obs_batch = ori_sample[0]
        
        compact_share_obs_batch = share_obs_batch
        # compact_share_obs_batch = self.get_critic_input(share_obs_batch)
        return super().ppo_update((compact_share_obs_batch, *ori_sample[1:]), update_actor)
        
    def _redistribute_reward_dynamic(self, state, action, agent_id, type_id, test):
        # state/action: [batch, num_agent, joint_state_dim/joint_action_dim] (joint repeated per agent),
        # agent_id: [batch, num_agent, num_agent] (identity). psi_g predicts per-sample, per-agent masks.
        state_tensor = check(state).to(**self.tpdv)
        action_tensor = check(action).to(**self.tpdv)
        agent_id_t = check(agent_id).to(**self.tpdv)
        g_in = torch.cat([state_tensor, action_tensor, agent_id_t], -1)
        mask_s = torch.sigmoid(self.g_s2r(g_in))  # [batch, num_agent, joint_state_dim]
        mask_a = torch.sigmoid(self.g_a2r(g_in))  # [batch, num_agent, joint_action_dim]
        self._last_mask_s, self._last_mask_a = mask_s, mask_a
        if test:
            mask_s = (mask_s > 0.5).float()
            mask_a = (mask_a > 0.5).float()
        return self.reward_predictor(
            torch.mul(mask_s, state_tensor),
            torch.mul(mask_a, action_tensor),
            agent_id, type_id)

    def redistribute_reward(self, state, action, info):
        test = info['test']
        agent_id = info['agent_id']
        type_id = info['type_id']
        if getattr(self, 'dynamic_mask', False):
            return self._redistribute_reward_dynamic(state, action, agent_id, type_id, test)
        # state
        s2r_mask = self.get_structure_param(self.causal_encoder['S2R'], test=test) # joint_state_dim, num_agent
        a2r_mask = self.get_structure_param(self.causal_encoder['A2R'], test=test) # joint_action_dim, num_agent
        
        s2r_mask[:,-self.num_agents:] = 1.0
        
        # convert states and actions into joint state and joint action
        # state of (n_rollout_thread or batch_size, num_agent, joint_state_dim) 
        # import pdb; pdb.set_trace()
        # state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        state_tensor = state
        # id = torch.tensor(state[])
        continuous = True
        if continuous:
            action_tensor = action
        else:
            # action of (n_rollout_thread or batch_size, num_agent, 1) -> joint state of (batch_size, num_agent, joint_action_dim)
            action_tensor =  self.get_onehot_for_action(action).view(-1, self.joint_action_dim).unsqueeze(1).repeat(1, 3, 1)
            
        
        # action: batch_size, joint_action_dim
        return self.reward_predictor(
            torch.mul((s2r_mask.transpose(0,1)), state_tensor), # state: batch_size, num_agent, joint_state_dim
            torch.mul((a2r_mask.transpose(0,1)), action_tensor),  # action: batch_size, num_agent, joint_action_dim
            agent_id, type_id
            ) # reward: batch_size, num_agent , 1
        
    def get_onehot_for_action(self, action):
        # action (min_batch_size, num_agents, 1) into (mini_batch_size, num_agent * action_dim)
        action_tensor = torch.tensor(action[..., 0], dtype=torch.float32, device=self.device)
        return torch.nn.functional.one_hot(action_tensor.long(), \
            num_classes=self.action_dim).float()
        

 