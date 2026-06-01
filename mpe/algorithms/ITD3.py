import torch
import torch.nn.functional as F
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients
from utils.agents import TD3Agent, DDPGAgent
import itertools
import numpy as np
import random
import wandb


MSELoss = torch.nn.MSELoss()

class ITD3(object):
    def __init__(
        self, 
        args, 
        env,
        agent_init_params, 
        alg_types, 
        adv_init_params=None,
        gamma=0.95, 
        tau=0.01, 
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        gaussian_noise_std=None, 
        agent_max_actions=None, 
        cql=False, cql_alpha=None, lse_temp=1.0, num_sampled_actions=None, cql_sample_noise_level=0.2,
        omar=None, omar_coe=None,
        omar_mu=None, omar_sigma=None, omar_num_samples=None, omar_num_elites=None, omar_iters=None, batch_size=None, 
        env_id=None, num_steps = None,
    ):
        self.args = args
        self.env = env
        self.env_id = env_id
        self.is_mamujoco = True if self.env_id == 'HalfCheetah-v2' else False
        self.is_smac = True if self.env_id == 'StarCraft2' else False

        assert (ma == agent_max_actions[0] for ma in agent_max_actions)
        self.max_action = agent_max_actions[0]
        self.min_action = -self.max_action
        
        self.nagents = len(alg_types)
        self.alg_types = alg_types


        if args.IL:
            self.agents = [TD3Agent(
                lr=lr,
                discrete_action=discrete_action,
                hidden_dim=hidden_dim,
                gaussian_noise_std=gaussian_noise_std,
                critic_ln=getattr(args, 'critic_ln', False),
                policy_ln=getattr(args, 'policy_ln', False),
                **params
            ) for params in agent_init_params]
        else:
            self.agents = [TD3Agent(
                lr=lr,
                discrete_action=discrete_action,
                hidden_dim=hidden_dim,
                gaussian_noise_std=gaussian_noise_std,
                critic_ln=getattr(args, 'critic_ln', False),
                policy_ln=getattr(args, 'policy_ln', False),
                **params
            ) for params in agent_init_params]




        if self.env_id in ['simple_tag', 'simple_world']:
            self.num_predators = len(agent_init_params)
            self.num_preys = len(adv_init_params)
            self.preys = [DDPGAgent(lr=lr, discrete_action=discrete_action, hidden_dim=hidden_dim, **params) for params in adv_init_params]

        self.niter = 0

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action

        self.pol_dev, self.trgt_pol_dev, self.critic_dev, self.trgt_critic_dev = 'cpu', 'cpu', 'cpu', 'cpu' 

        self.omar = omar
        self.num_steps = num_steps
        self.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        
        if self.omar:
            self.omar_coe = omar_coe

            self.omar_iters = omar_iters
            self.omar_num_samples = omar_num_samples
            self.init_omar_mu, self.init_omar_sigma = omar_mu, omar_sigma
            # self.omar_mu = torch.cuda.FloatTensor(batch_size, self.agent_init_params[0]['num_out_pol']).zero_() + self.init_omar_mu
            self.omar_mu = (torch.zeros(batch_size, self.agent_init_params[0]['num_out_pol']) + self.init_omar_mu).float().to(self.device)
            self.omar_sigma = (torch.zeros(batch_size, self.agent_init_params[0]['num_out_pol']) + self.init_omar_sigma).float().to(self.device)
            # self.omar_sigma = torch.cuda.FloatTensor(batch_size, self.agent_init_params[0]['num_out_pol']).zero_() + self.init_omar_sigma
            self.omar_num_elites = omar_num_elites

        self.cql = cql
        if self.cql:
            self.cql_alpha = cql_alpha
            self.cql_sample_noise_level = cql_sample_noise_level
            self.lse_temp = lse_temp
            self.num_sampled_actions = num_sampled_actions
        

        self.causal = args.use_causal
        if self.causal:
            # self.causal_coefs = dict(s2r = args.s2r_coef, a2r = args.a2r_coef, s2s = args.s2s_coef, a2s = args.a2s_coef)
            # self.compact = False
            # define causal structure
            if self.is_mamujoco:
                args.num_agents = self.nagents
                args.shared_state_dim = self.env.get_state_size() * self.nagents
                args.action_dim = np.array([box.shape[0] for box in self.env.action_space]).sum()
                args.single_action_dim = self.env.action_space[0].shape[0]
                from algorithms.causal import CausalAgent
                self.causal_agent = CausalAgent(args, self.device)
            elif self.is_smac:
                args.num_agents = self.nagents
                #TODO: Check this
                args.is_continuous = False
                args.shared_state_dim = self.env.get_state_size() * self.nagents
                args.action_dim = np.array([box.shape[0] for box in self.env.action_space]).sum()
                args.single_action_dim = self.env.action_space[0].shape[0]
                from algorithms.causal import CausalAgent
                self.causal_agent = CausalAgent(args, self.device)
            else: 
                args.num_agents = self.nagents
                args.shared_state_dim = np.array([box.shape[0] for box in self.env.observation_space]).sum()
                args.action_dim = np.array([box.shape[0] for box in self.env.action_space]).sum()
                args.single_action_dim = self.env.action_space[0].shape[0]
                from algorithms.causal import CausalAgent
                self.causal_agent = CausalAgent(args, self.device)
                
    def get_structure(self):
        if self.causal:
            return self.causal_agent.get_causal_structure()
        else:
            raise NotImplementedError
            

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        res = []
        
        # import pdb; pdb.set_trace()
        
        for i, obs in zip(range(self.nagents), observations):
            if self.env_id in ['simple_world', 'simple_tag']:
                if i < self.num_predators:
                    predator_action = self.agents[i].step(obs, explore=explore)
                    res.append(predator_action)
                else:
                    prey_action = self.preys[i - self.num_predators].step(obs, explore=False)
                    res.append(prey_action)
            else:
                action = self.agents[i].step(obs, explore=explore)
                res.append(action)
        return res

    def step_test_cem(self, observations, n_samples=64, n_iters=2, sigma=0.5, n_elites=8):
        """TRICK (test-time): critic-guided action selection. For each agent, run a small CEM
        around the policy's action using the trained (per-agent) critic to pick the highest-Q
        action — test-time optimization that lifts eval performance without changing training.
        Opt-in via --test_cem; preys keep their fixed policy."""
        res = []
        mx = float(getattr(self, 'max_action', 1.0))
        for i, obs in zip(range(self.nagents), observations):
            if self.env_id in ['simple_world', 'simple_tag'] and i >= self.num_predators:
                res.append(self.preys[i - self.num_predators].step(obs, explore=False))
                continue
            ag = self.agents[i]
            cdev = next(ag.critic.parameters()).device     # critic may be on a different device than the policy
            with torch.no_grad():
                mu0 = ag.policy(obs).clamp(-mx, mx)          # (B, A) on the policy/obs device
                odev = mu0.device
                mu = mu0.to(cdev); obs_c = obs.to(cdev)
                B, A = mu.shape
                best = mu.clone()
                best_q = ag.critic.Q1(torch.cat((obs_c, mu), dim=1))   # (B,1)
                cur_mu, cur_sig = mu, sigma
                for _ in range(n_iters):
                    cand = (cur_mu.unsqueeze(1) + cur_sig * torch.randn(B, n_samples, A, device=cdev)).clamp(-mx, mx)
                    obs_rep = obs_c.unsqueeze(1).repeat(1, n_samples, 1).reshape(-1, obs_c.shape[1])
                    q = ag.critic.Q1(torch.cat((obs_rep, cand.reshape(-1, A)), dim=1)).reshape(B, n_samples)
                    topq, topi = q.topk(min(n_elites, n_samples), dim=1)
                    elite = torch.gather(cand, 1, topi.unsqueeze(-1).repeat(1, 1, A))
                    cur_mu = elite.mean(1); cur_sig = elite.std(1) + 1e-3
                    bq, bidx = q.max(1)
                    bcand = torch.gather(cand, 1, bidx.view(-1, 1, 1).repeat(1, 1, A)).squeeze(1)
                    imp = (bq.unsqueeze(1) > best_q)
                    best = torch.where(imp, bcand, best)
                    best_q = torch.where(imp, bq.unsqueeze(1), best_q)
            res.append(best.to(odev))
        return res

    def calc_gaussian_pdf(self, samples, mu=0):
        pdfs = 1 / (self.cql_sample_noise_level * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * self.cql_sample_noise_level**2) )
        pdf = torch.prod(pdfs, dim=-1)
        return pdf

    def get_policy_actions(self, state, network):
        action = network(state)

        formatted_action = action.unsqueeze(1).repeat(1, self.num_sampled_actions, 1).view(action.shape[0] * self.num_sampled_actions, action.shape[1])

        random_noises = torch.FloatTensor(formatted_action.shape[0], formatted_action.shape[1])

        random_noises = random_noises.normal_() * self.cql_sample_noise_level
        random_noises_log_pi = self.calc_gaussian_pdf(random_noises).view(action.shape[0], self.num_sampled_actions, 1).to(self.device)
        random_noises = random_noises.to(self.device)

        noisy_action = (formatted_action + random_noises).clamp(-self.max_action, self.max_action)

        return noisy_action, random_noises_log_pi

    def compute_softmax_acs(self, q_vals, acs):
        max_q_vals = torch.max(q_vals, 1, keepdim=True)[0]
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = torch.exp(norm_q_vals)
        a_mult_e = acs * e_beta_normQ
        numerators = a_mult_e
        denominators = e_beta_normQ

        sum_numerators = torch.sum(numerators, 1)
        sum_denominators = torch.sum(denominators, 1)

        softmax_acs = sum_numerators / sum_denominators

        return softmax_acs
    
    

    def update(self, sample, agent_i, t, parallel=False):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next observations, and episode end masks) 
                    sampled randomly from the replay buffer. Each is a list with entries corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch): If passed in, important quantities will be logged
        """
        if self.is_mamujoco:
            states, obs, acs, rews, next_states, next_obs, dones = sample
        elif self.is_smac:
            #TODO: Check this
            import pdb; pdb.set_trace()
            obs, states, rews, dones, infos, avaliable_actions  = sample
        else:
            obs, acs, rews, next_obs, dones = sample


        # import pdb; pdb.set_trace()
        # check rews[0] is equal rews[1]
        # assert torch.equal(rews[0], rews[1])
        
        
        # sum up rewards for each agent
        if not self.is_mamujoco:
            ori_res = rews
            rews = torch.stack(rews, dim=1).sum(-1)
            rews = [rews for _ in range(self.nagents)]
        
        # # modified by yudi
        # rews = torch.stack(rews, dim=1).mean(-1)
        # rews = [rews for _ in range(self.nagents)]
            
        # print('rews', rews)
        curr_agent = self.agents[agent_i]
        curr_agent.critic_optimizer.zero_grad()
            
        
        
        IL = 1
        if IL:
            trgt_acs = curr_agent.target_policy(next_obs[agent_i])
            trgt_vf_in = torch.cat((next_obs[agent_i], trgt_acs), dim=1)
        else:
            trgt_acs = []
            for i in range(self.nagents):
                trgt_acs.append(curr_agent.target_policy(next_obs[i]))
            trgt_vf_in = torch.cat((next_obs,trgt_acs), dim=1)
        
        next_q_value1, next_q_value2 = curr_agent.target_critic(trgt_vf_in) 
        next_q_value = torch.min(next_q_value1, next_q_value2)
        
        self.loss_dict = dict()

        if self.causal:
            # TODO: redistributed rewards
            
            if self.is_mamujoco:
                joint_state = torch.transpose(torch.unsqueeze(torch.cat(states, -1), 0).repeat((len(states), 1, 1)), 0, 1)
                joint_action = torch.transpose(torch.unsqueeze(torch.cat(acs, -1), 0).repeat((len(acs), 1, 1)), 0, 1)
                joint_next_state = torch.transpose(torch.unsqueeze(torch.cat(next_states, -1), 0).repeat((len(next_states), 1, 1)), 0, 1)
            else:
                joint_state = torch.transpose(torch.unsqueeze(torch.cat(obs, -1), 0).repeat((len(obs), 1, 1)), 0, 1)
                joint_action = torch.transpose(torch.unsqueeze(torch.cat(acs, -1), 0).repeat((len(acs), 1, 1)), 0, 1)
                joint_next_state = torch.transpose(torch.unsqueeze(torch.cat(next_obs, -1), 0).repeat((len(next_obs), 1, 1)), 0, 1)
            
            
            # rews: sample from the env,  [tensor(shape=2) for i in range(num_agents)] team reward
            # 
            # team_R = Rs[causal_indices_step, causal_indices_rollout]
            
            team_R = torch.stack(rews).mean(0).unsqueeze(-1)
            
            if self.is_mamujoco:
                team_ori_R = team_R
            else:
                team_ori_R = ori_res[agent_i].unsqueeze(-1)
            
            agent_id_causal = torch.eye(self.nagents).unsqueeze(0).repeat(joint_state.shape[0], 1, 1).to(self.device)
            
            if self.is_mamujoco:
                type_id = 0
                causal_sample = joint_state, joint_action, joint_next_state, team_R, agent_id_causal, type_id
                self.loss_dict.update(self.causal_agent.causal_graph_update(causal_sample))
            else:
                type_id = (np.array(self.env.agent_types)=='agents').astype(np.int)
                causal_sample = joint_state, joint_action, joint_next_state, team_R, agent_id_causal, type_id
                ground_truth_causal_sample = joint_state, joint_action, joint_next_state, team_R, team_ori_R, agent_i, agent_id_causal, type_id
                
                ground_truth_test = 0
                if not ground_truth_test:
                    self.loss_dict.update(self.causal_agent.causal_graph_update(causal_sample))
                else:
                    self.loss_dict.update(self.causal_agent.causal_graph_update_gt(ground_truth_causal_sample))

            
            
            info = dict(test=False, agent_id = agent_id_causal, type_id= type_id)
            
            use_pred_single_reward = 1
            use_pred_single_ratio = 0
            use_pr_plus_teamr = 0
            
            if use_pred_single_reward:  
                #(batch_size, num_agents, 1) -> [tensor(shape=2) for i in range(num_agents)]
                # agent id (batch_size, num_agent, num_agent)
                pred_single_rews = self.causal_agent.redistribute_reward(joint_state, joint_action, info).chunk(self.nagents, 1)
                pred_single_rews = [re[:, 0, 0] for re in pred_single_rews]
                rews = pred_single_rews
               
            elif use_pred_single_ratio:
                pred_rewards = self.causal_agent.redistribute_reward(joint_state, joint_action, info)
                pred_team_rewards = pred_rewards[:, :, 0].sum(dim=-1, keepdim=True)
                pred_ratio = pred_rewards / pred_team_rewards.unsqueeze(1)
                pred_single_rews = pred_ratio * team_R.unsqueeze(1)
                pred_single_rews = pred_single_rews.chunk(self.nagents, 1)
                pred_single_rews = [re[:, 0, 0] for re in pred_single_rews]
                rews = pred_single_rews
            elif use_pr_plus_teamr:
                with torch.no_grad():
                    pred_single_rews = self.causal_agent.redistribute_reward(joint_state, joint_action, info).chunk(self.nagents, 1)
                    # pred_single_rews = [re[:, 0, 0] for re, re_team in zip(pred_single_rews, rews)]
                    # if t <= 3e4:
                    # rews = pred_single_rews
                        
                    decay_rate = (1 - t/2e5)
                    pred_single_rews = [re[:, 0, 0] * decay_rate + (1 - decay_rate) * re_team for re, re_team in zip(pred_single_rews, rews)]
                    # pred_single_rews = [torch.rand_like(re).to(self.device) + 1e5 for re in pred_single_rews]
                    # for i in range(self.nagents):
                    #    pred_single_rews[i] = (1-lemdba) * pred_single_rews[i] + lemdba * team_R.squeeze(-1)
                    # rews = [ r/3 for r in rews]
                    # rews = rews
                    rews = pred_single_rews
                

        # rews = [ r/10000 for r in rews]
        # Trick: reward (scale) normalization by a running std — stabilizes Q-targets and the
        # CQL conservative term on suboptimal data (offline RL is very sensitive to reward scale).
        if getattr(self.args, 'rew_norm', False):
            if not hasattr(self, '_rew_std'):
                self._rew_std = 1.0
            with torch.no_grad():
                bstd = rews[agent_i].detach().float().std().item()
                self._rew_std = 0.99 * self._rew_std + 0.01 * max(bstd, 1e-3)
            rews = [r / self._rew_std for r in rews]

        target_value = rews[agent_i].view(-1, 1) + self.gamma * next_q_value * (1 - dones[agent_i].view(-1, 1))

        vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        
        actual_value1, actual_value2 = curr_agent.critic(vf_in) 

        vf_loss = MSELoss(actual_value1, target_value.detach()) + MSELoss(actual_value2, target_value.detach())
        
        if self.cql:
            if self.is_mamujoco:
                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.num_sampled_actions, 1).view(-1, obs[agent_i].shape[1])

                random_action = (torch.FloatTensor(acs[agent_i].shape[0] * self.num_sampled_actions, acs[agent_i].shape[1]).uniform_(-1, 1)).to(self.device)
                random_action_log_pi = np.log(0.5 ** random_action.shape[-1])
                curr_action, curr_action_log_pi = self.get_policy_actions(obs[agent_i], curr_agent.policy)
                new_curr_action, new_curr_action_log_pi = self.get_policy_actions(next_obs[agent_i], curr_agent.policy)

                random_vf_in = torch.cat((formatted_obs, random_action), dim=1)
                curr_vf_in = torch.cat((formatted_obs, curr_action), dim=1)
                new_curr_vf_in = torch.cat((formatted_obs, new_curr_action), dim=1)

                random_Q1, random_Q2 = curr_agent.critic(random_vf_in)
                curr_Q1, curr_Q2 = curr_agent.critic(curr_vf_in)
                new_curr_Q1, new_curr_Q2 = curr_agent.critic(new_curr_vf_in)

                random_Q1, random_Q2 = random_Q1.view(obs[agent_i].shape[0], self.num_sampled_actions, 1), random_Q2.view(obs[agent_i].shape[0], self.num_sampled_actions, 1)
                curr_Q1, curr_Q2 = curr_Q1.view(obs[agent_i].shape[0], self.num_sampled_actions, 1), curr_Q2.view(obs[agent_i].shape[0], self.num_sampled_actions, 1)
                new_curr_Q1, new_curr_Q2 = new_curr_Q1.view(obs[agent_i].shape[0], self.num_sampled_actions, 1), new_curr_Q2.view(obs[agent_i].shape[0], self.num_sampled_actions, 1)

                cat_q1 = torch.cat([random_Q1 - random_action_log_pi, new_curr_Q1 - new_curr_action_log_pi, curr_Q1 - curr_action_log_pi], 1)
                cat_q2 = torch.cat([random_Q2 - random_action_log_pi, new_curr_Q2 - new_curr_action_log_pi, curr_Q2 - curr_action_log_pi], 1)
                
                policy_qvals1 = torch.logsumexp(cat_q1 / self.lse_temp, dim=1) * self.lse_temp
                policy_qvals2 = torch.logsumexp(cat_q2 / self.lse_temp, dim=1) * self.lse_temp
            else:
                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.num_sampled_actions, 1).view(-1, obs[agent_i].shape[1])

                random_acs = (torch.FloatTensor(acs[agent_i].shape[0] * self.num_sampled_actions, acs[agent_i].shape[1]).uniform_(-1, 1)).to(self.device)
                random_acs_log_pi = np.log(0.5 ** random_acs.shape[-1])

                random_vf_in = torch.cat((formatted_obs, random_acs), dim=1)

                random_qvals1, random_qvals2 = curr_agent.critic(random_vf_in)

                random_qvals1 = random_qvals1.view(obs[agent_i].shape[0], self.num_sampled_actions)
                random_qvals2 = random_qvals2.view(obs[agent_i].shape[0], self.num_sampled_actions)

                policy_qvals1 = torch.logsumexp((random_qvals1 - random_acs_log_pi) / self.lse_temp, dim=1, keepdim=True) * self.lse_temp 
                policy_qvals2 = torch.logsumexp((random_qvals2 - random_acs_log_pi) / self.lse_temp, dim=1, keepdim=True) * self.lse_temp

            dataset_q_vals1 = actual_value1
            dataset_q_vals2 = actual_value2

            cql_term1 = (policy_qvals1 - dataset_q_vals1).mean()
            cql_term2 = (policy_qvals2 - dataset_q_vals2).mean()
            
            cql_term = cql_term1 + cql_term2
            vf_loss += self.cql_alpha * cql_term

        vf_loss.backward()
        
        
        
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # Trick: IQL-style expectile value baseline (opt-in --iql_tau). Learns V(s) as the
        # tau-expectile of Q(s,a_data) over the DATA -> the value of GOOD in-distribution actions,
        # without ever querying OOD actions (so no divergence like low cql_alpha). Used as the AWR
        # baseline below so adv = Q(s,a_data) - V(s) cleanly flags the p90 transitions to clone.
        # Lazy per-agent net => fully opt-in, the working CQL/OMAR/expert paths are untouched.
        self._iql_adv = None
        if getattr(self.args, 'iql_tau', 0.0) > 0:
            if not hasattr(self, '_value_nets'):
                self._value_nets, self._value_opts = {}, {}
            if agent_i not in self._value_nets:
                _odim = obs[agent_i].shape[1]
                vn = torch.nn.Sequential(
                    torch.nn.Linear(_odim, 256), torch.nn.LayerNorm(256), torch.nn.ReLU(),
                    torch.nn.Linear(256, 256), torch.nn.ReLU(),
                    torch.nn.Linear(256, 1)).to(self.device)
                self._value_nets[agent_i] = vn
                self._value_opts[agent_i] = torch.optim.Adam(vn.parameters(), lr=self.lr)
            vnet, vopt = self._value_nets[agent_i], self._value_opts[agent_i]
            with torch.no_grad():
                _q1d, _q2d = curr_agent.critic(torch.cat((obs[agent_i], acs[agent_i]), dim=1))
                _q_sa = torch.min(_q1d, _q2d)
            v_pred = vnet(obs[agent_i])
            _diff = _q_sa - v_pred
            _tau = self.args.iql_tau
            v_loss = (torch.abs(_tau - (_diff < 0).float()) * _diff ** 2).mean()
            vopt.zero_grad(); v_loss.backward(); vopt.step()
            with torch.no_grad():
                self._iql_adv = _q_sa - vnet(obs[agent_i])

        curr_agent.policy_optimizer.zero_grad()

        curr_pol_out = curr_agent.policy(obs[agent_i])
        curr_pol_vf_in = curr_pol_out
        
        vf_in = torch.cat((obs[agent_i], curr_pol_vf_in), dim=1)


        if self.omar:
            pred_qvals = curr_agent.critic.Q1(vf_in)

            if self.is_mamujoco:
                # self.omar_mu = torch.cuda.FloatTensor(acs[agent_i].shape[0], acs[agent_i].shape[1]).zero_() + self.init_omar_mu
                # self.omar_sigma = torch.cuda.FloatTensor(acs[agent_i].shape[0], acs[agent_i].shape[1]).zero_() + self.init_omar_sigma
                self.omar_mu = (torch.zeros(acs[agent_i].shape[0], acs[agent_i].shape[1]) + self.init_omar_mu).float().to(self.device)
                self.omar_sigma = (torch.zeros(acs[agent_i].shape[0], acs[agent_i].shape[1]) + self.init_omar_sigma).float().to(self.device)
                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.omar_num_samples, 1).view(-1, obs[agent_i].shape[1])

                last_top_k_qvals, last_elite_acs = None, None
                for iter_idx in range(self.omar_iters):
                    
                    try:
                        self.omar_sigma[self.omar_sigma <= 0] = 0.1
                        dist = torch.distributions.Normal(self.omar_mu, self.omar_sigma)
                    except:
                        import pdb; pdb.set_trace()
                    
                    ##([256, 20, 3])
                    cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1, 0, 2).clamp(-self.max_action, self.max_action)
                    
                    formatted_cem_sampled_acs = cem_sampled_acs.reshape(-1, cem_sampled_acs.shape[-1])

                    vf_in = torch.cat((formatted_obs, formatted_cem_sampled_acs), dim=1)
                    all_pred_qvals = curr_agent.critic.Q1(vf_in).reshape(acs[agent_i].shape[0], -1, 1)

                    if iter_idx > 0:
                        all_pred_qvals = torch.cat((all_pred_qvals, last_top_k_qvals), dim=1)
                        cem_sampled_acs = torch.cat((cem_sampled_acs, last_elite_acs), dim=1)

                    top_k_qvals, top_k_inds = torch.topk(all_pred_qvals, self.omar_num_elites, dim=1)
                    elite_ac_inds = top_k_inds.repeat(1, 1, acs[agent_i].shape[1])
                    elite_acs = torch.gather(cem_sampled_acs, 1, elite_ac_inds)

                    last_top_k_qvals, last_elite_acs = top_k_qvals, elite_acs

                    updated_mu = torch.mean(elite_acs, dim=1)
                    updated_sigma = torch.std(elite_acs, dim=1)

                    self.omar_mu = updated_mu
                    self.omar_sigma = updated_sigma

                top_qvals, top_inds = torch.topk(all_pred_qvals, 1, dim=1)
                top_ac_inds = top_inds.repeat(1, 1, acs[agent_i].shape[1])
                top_acs = torch.gather(cem_sampled_acs, 1, top_ac_inds)

                cem_qvals = top_qvals
                pol_qvals = pred_qvals.unsqueeze(1)
                cem_acs = top_acs
                pol_acs = curr_pol_out.unsqueeze(1)

                candidate_qvals = torch.cat([pol_qvals, cem_qvals], 1)
                candidate_acs = torch.cat([pol_acs, cem_acs], 1)

                max_qvals, max_inds = torch.max(candidate_qvals, 1, keepdim=True)
                max_ac_inds = max_inds.repeat(1, 1, acs[agent_i].shape[1])

                max_acs = torch.gather(candidate_acs, 1, max_ac_inds).squeeze(1)
            elif self.is_smac:
                self.omar_mu = (torch.zeros(acs[agent_i].shape[0], acs[agent_i].shape[1]) + self.init_omar_mu).float().to(self.device)
                self.omar_sigma = (torch.zeros(acs[agent_i].shape[0], acs[agent_i].shape[1]) + self.init_omar_sigma).float().to(self.device)
                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.omar_num_samples, 1).view(-1, obs[agent_i].shape[1])
                last_top_k_qvals, last_elite_acs = None, None
                for iter_idx in range(self.omar_iters):
                    
                    try:
                        self.omar_sigma[self.omar_sigma <= 0] = 0.1
                        dist = torch.distributions.Normal(self.omar_mu, self.omar_sigma)
                    except:
                        import pdb; pdb.set_trace()
                    
                    ##([256, 20, 3])
                    cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1, 0, 2).clamp(-self.max_action, self.max_action)
                    
                    formatted_cem_sampled_acs = cem_sampled_acs.reshape(-1, cem_sampled_acs.shape[-1])

                    vf_in = torch.cat((formatted_obs, formatted_cem_sampled_acs), dim=1)
                    all_pred_qvals = curr_agent.critic.Q1(vf_in).reshape(acs[agent_i].shape[0], -1, 1)

                    if iter_idx > 0:
                        all_pred_qvals = torch.cat((all_pred_qvals, last_top_k_qvals), dim=1)
                        cem_sampled_acs = torch.cat((cem_sampled_acs, last_elite_acs), dim=1)

                    top_k_qvals, top_k_inds = torch.topk(all_pred_qvals, self.omar_num_elites, dim=1)
                    elite_ac_inds = top_k_inds.repeat(1, 1, acs[agent_i].shape[1])
                    elite_acs = torch.gather(cem_sampled_acs, 1, elite_ac_inds)

                    last_top_k_qvals, last_elite_acs = top_k_qvals, elite_acs

                    updated_mu = torch.mean(elite_acs, dim=1)
                    updated_sigma = torch.std(elite_acs, dim=1)

                    self.omar_mu = updated_mu
                    self.omar_sigma = updated_sigma

                top_qvals, top_inds = torch.topk(all_pred_qvals, 1, dim=1)
                top_ac_inds = top_inds.repeat(1, 1, acs[agent_i].shape[1])
                top_acs = torch.gather(cem_sampled_acs, 1, top_ac_inds)

                cem_qvals = top_qvals
                pol_qvals = pred_qvals.unsqueeze(1)
                cem_acs = top_acs
                pol_acs = curr_pol_out.unsqueeze(1)

                candidate_qvals = torch.cat([pol_qvals, cem_qvals], 1)
                candidate_acs = torch.cat([pol_acs, cem_acs], 1)

                max_qvals, max_inds = torch.max(candidate_qvals, 1, keepdim=True)
                max_ac_inds = max_inds.repeat(1, 1, acs[agent_i].shape[1])

                max_acs = torch.gather(candidate_acs, 1, max_ac_inds).squeeze(1)

            else:
                # self.omar_mu = torch.cuda.FloatTensor(acs[agent_i].shape[0], acs[agent_i].shape[1]).zero_() + self.init_omar_mu
                # self.omar_sigma = torch.cuda.FloatTensor(acs[agent_i].shape[0], acs[agent_i].shape[1]).zero_() + self.init_omar_sigma
                self.omar_mu = (torch.zeros(acs[agent_i].shape[0], acs[agent_i].shape[1]) + self.init_omar_mu).float().to(self.device)
                self.omar_sigma = (torch.zeros(acs[agent_i].shape[0], acs[agent_i].shape[1]) + self.init_omar_sigma).float().to(self.device)
                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.omar_num_samples, 1).view(-1, obs[agent_i].shape[1])

                for iter_idx in range(self.omar_iters):
                    
                    # try calulating the distribution here if not working then import pdb; pdb.set_trace() and check
                    try:
                        self.omar_sigma[self.omar_sigma <= 0] = 0.1
                        dist = torch.distributions.Normal(self.omar_mu, self.omar_sigma)
                    except:
                        import pdb; pdb.set_trace()
                    

                    cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1, 0, 2).clamp(-self.max_action, self.max_action)

                    # formatted_cem_sampled_acs = cem_sampled_acs.view(-1, cem_sampled_acs.shape[-1])
                    formatted_cem_sampled_acs = cem_sampled_acs.reshape(-1, cem_sampled_acs.shape[-1])
                    
                    vf_in = torch.cat((formatted_obs, formatted_cem_sampled_acs), dim=1)
                    all_pred_qvals = curr_agent.critic.Q1(vf_in)
                    all_pred_qvals = all_pred_qvals.view(acs[agent_i].shape[0], -1, 1)

                    updated_mu = self.compute_softmax_acs(all_pred_qvals, cem_sampled_acs)
                    self.omar_mu = updated_mu

                    updated_sigma = torch.sqrt(torch.mean((cem_sampled_acs - updated_mu.unsqueeze(1)) ** 2, 1))
                    self.omar_sigma = updated_sigma

                top_qvals, top_inds = torch.topk(all_pred_qvals, 1, dim=1)
                top_ac_inds = top_inds.repeat(1, 1, acs[agent_i].shape[1])
                top_acs = torch.gather(cem_sampled_acs, 1, top_ac_inds)

                cem_qvals = top_qvals
                pol_qvals = pred_qvals.unsqueeze(1)
                cem_acs = top_acs
                pol_acs = curr_pol_out.unsqueeze(1)

                candidate_qvals = torch.cat([pol_qvals, cem_qvals], 1)
                candidate_acs = torch.cat([pol_acs, cem_acs], 1)

                max_qvals, max_inds = torch.max(candidate_qvals, 1, keepdim=True)
                max_ac_inds = max_inds.repeat(1, 1, acs[agent_i].shape[1])

                max_acs = torch.gather(candidate_acs, 1, max_ac_inds).squeeze(1)
                        
            mimic_acs = max_acs.detach()
            
            mimic_term = F.mse_loss(curr_pol_out, mimic_acs)

            pol_loss = self.omar_coe * mimic_term - (1 - self.omar_coe) * pred_qvals.mean()

            pol_loss += (curr_pol_out ** 2).mean() * 1e-3
            # Trick: TD3+BC behavior-cloning anchor to the dataset action (Q-normalized weight).
            # Strong on suboptimal (medium/replay) data — keeps the policy in-distribution while Q improves it.
            if getattr(self.args, 'td3bc_coef', 0.0) > 0:
                lmbda = self.args.td3bc_coef / (pred_qvals.abs().mean().detach() + 1e-6)
                pol_loss = lmbda * pol_loss + F.mse_loss(curr_pol_out, acs[agent_i])
            # Trick: Advantage-Weighted Regression (AWR) — in-distribution policy improvement.
            # Reweights BC toward dataset actions with positive advantage Q(s,a_data)-Q(s,pi(s)),
            # i.e. the top-decile (p90) trajectories, which is exactly where the paper's non-expert
            # numbers sit. Unlike CEM it never queries OOD actions, so a conservative critic can't veto it.
            if getattr(self.args, 'awr_temp', 0.0) > 0:
                with torch.no_grad():
                    if getattr(self, '_iql_adv', None) is not None:
                        adv = self._iql_adv                       # IQL expectile-V advantage (cleaner)
                    else:
                        q_data = curr_agent.critic.Q1(torch.cat((obs[agent_i], acs[agent_i]), dim=1))
                        adv = q_data - pred_qvals                  # fallback: Q(s,a)-Q(s,pi)
                    w = torch.exp(adv / (self.args.awr_temp * (adv.std().detach() + 1e-6))).clamp(max=100.0)
                awr_bc = (w * ((curr_pol_out - acs[agent_i]) ** 2).mean(dim=1, keepdim=True)).mean()
                pol_loss = pol_loss + getattr(self.args, 'awr_coef', 1.0) * awr_bc

        else:
            #Caluler the ratio about the agents' rewards / the whole team rewards

            pol_loss = -curr_agent.critic.Q1(vf_in).mean()

            pol_loss += (curr_pol_out ** 2).mean() * 1e-3
            if getattr(self.args, 'td3bc_coef', 0.0) > 0:
                with torch.no_grad():
                    _q_abs = curr_agent.critic.Q1(vf_in).abs().mean()
                lmbda = self.args.td3bc_coef / (_q_abs + 1e-6)
                pol_loss = lmbda * pol_loss + F.mse_loss(curr_pol_out, acs[agent_i])
            if getattr(self.args, 'awr_temp', 0.0) > 0:
                with torch.no_grad():
                    if getattr(self, '_iql_adv', None) is not None:
                        adv = self._iql_adv
                    else:
                        q_pol = curr_agent.critic.Q1(vf_in)
                        q_data = curr_agent.critic.Q1(torch.cat((obs[agent_i], acs[agent_i]), dim=1))
                        adv = q_data - q_pol
                    w = torch.exp(adv / (self.args.awr_temp * (adv.std().detach() + 1e-6))).clamp(max=100.0)
                awr_bc = (w * ((curr_pol_out - acs[agent_i]) ** 2).mean(dim=1, keepdim=True)).mean()
                pol_loss = pol_loss + getattr(self.args, 'awr_coef', 1.0) * awr_bc
        

        
        
        
        if self.args.use_wandb:
            self.loss_dict.update({
                        "pol_loss": pol_loss.item(),
                        })
            if self.omar:
                self.loss_dict.update({"mimic_term": mimic_term.item(), "pred_qvals": pred_qvals.mean().item(),})
            self.log_train()
            
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been performed for each agent)
        """
        if self.env_id in ['simple_tag', 'simple_world']:
            end_idx = self.num_predators
        else:
            end_idx = len(self.agents)

        for i, a in enumerate(self.agents[:end_idx]):
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau) 
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train() 
            a.target_critic.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)

            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.policy = fn(p.policy)

            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device: 
            for a in self.agents: 
                a.target_policy = fn(a.target_policy)

            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.target_policy = fn(p.target_policy)

            self.trgt_pol_dev = device 
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.policy = fn(p.policy)
            self.pol_dev = device

    @classmethod
    def init_from_env(cls, args, env, env_id, data_type, env_info=None, agent_alg="td3", adversary_alg="ddpg", gamma=0.95, \
            tau=0.01, lr=0.01, hidden_dim=64, cql=False, batch_size=None, lse_temp=None, num_sampled_actions=None, \
            gaussian_noise_std=None, omar=None, omar_mu=None, omar_sigma=None, omar_num_samples=None, omar_num_elites=None, \
            omar_iters=None, num_steps = None,):
        """
        Instantiate instance of this class from multi-agent environment
        """
        IL = args.IL
        
        
        if env_id in ['simple_tag', 'simple_world']:
            alg_types = [agent_alg if atype == 'adversary' else adversary_alg for atype in env.agent_types]
        elif env_id in ['simple_spread']:
            alg_types = [agent_alg for atype in env.agent_types]
        elif env_id in ['HalfCheetah-v2']:
            alg_types = [agent_alg for _ in range(env_info['n_agents'])]
        else:
            alg_types = [agent_alg for _ in range(env_info['n_agents'])]

        agent_init_params = []
        all_n_actions = []
        agent_max_actions = []
        adv_init_params = []

        if env_id == 'HalfCheetah-v2':
            for agent_idx, algtype in zip(range(len(alg_types)), alg_types):
                acsp = env_info['action_spaces'][agent_idx]

                num_in_pol = env_info['obs_shape']
                num_out_pol = acsp.shape[0]
                num_in_critic = num_in_pol + num_out_pol

                agent_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})
                
                agent_max_actions.append(acsp.high[0])
                all_n_actions.append(acsp.shape[0])
        elif env_id == 'StarCraft2':  
            for agent_idx, algtype in zip(range(len(alg_types)), alg_types):
                acsp = env_info['action_spaces'][agent_idx]

                num_in_pol = env_info['obs_shape']
                num_out_pol = acsp.shape[0]
                num_in_critic = num_in_pol + num_out_pol

                agent_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})
                
                agent_max_actions.append(acsp.high[0])
                all_n_actions.append(acsp.shape[0])
            
        else:
            for acsp, obsp, algtype, agent_type in zip(env.action_space, env.observation_space, alg_types, env.agent_types):
                if IL:
                    num_in_pol = obsp.shape[0]
                    num_out_pol = acsp.shape[0]
                    num_in_critic = num_in_pol + num_out_pol
                else:
                    num_in_pol = obsp.shape[0]
                    num_out_pol = acsp.shape[0]
                    import pdb; pdb.set_trace()
                    num_in_critic = np.sum([box.shape[0] for box in env.observation_space]) + \
                                            np.sum([box.shape[0] for box in env.action_space])
                    
                if env_id in ['simple_spread']:
                    agent_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})
                    agent_max_actions.append(acsp.high[0])
                else:
                    if agent_type == 'adversary':
                        agent_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})
                        agent_max_actions.append(acsp.high[0])
                    elif agent_type == 'agent':
                        adv_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})

                all_n_actions.append(acsp.shape[0])

            for i in range(1, len(all_n_actions)):
                assert (all_n_actions[i] == all_n_actions[0])

        env_config_map = {
            'simple_spread': {
                # 'random': {'omar_coe': 1.0, 'cql_alpha': 0.5},
                
                'random': {'omar_coe': 0.5, 'cql_alpha': 5.0},
                
                'medium-replay': {'omar_coe': 1.0, 'cql_alpha': 1.0},
                # 'medium': {'omar_coe': 1.0, 'cql_alpha': 5.0}, # by default
                'medium': {'omar_coe': 0.995, 'cql_alpha': 5.0}, # by yudi
                'expert': {'omar_coe': 1.0, 'cql_alpha': 5.0}, # by default
                # 'expert': {'omar_coe': 0.9, 'cql_alpha': 5.0}, # by yudi
            },
            'simple_tag': {
                'random': {'omar_coe': 0.9, 'cql_alpha': 0.5},
                'medium-replay': {'omar_coe': 0.9, 'cql_alpha': 0.5},
                'medium': {'omar_coe': 0.7, 'cql_alpha': 5.0},
                'expert': {'omar_coe': 0.9, 'cql_alpha': 5.0},
            },
            'simple_world': {
                # 'random': {'omar_coe': 1.0, 'cql_alpha': 0.5},
                
                'random': {'omar_coe': 0.5, 'cql_alpha': 5.0},
                
                'medium-replay': {'omar_coe': 0.7, 'cql_alpha': 1.0},
                'medium': {'omar_coe': 0.1, 'cql_alpha': 0.5},
                'expert': {'omar_coe': 0.9, 'cql_alpha': 5.0},
            },
            'HalfCheetah-v2': {
                'random': {'omar_coe': 0.8, 'cql_alpha': 1.0},
                'medium-replay': {'omar_coe': 0.7, 'cql_alpha': 5.0},
                'medium': {'omar_coe': 0.7, 'cql_alpha': 1.0},
                'expert': {'omar_coe': 0.5, 'cql_alpha': 5.0},
            }
        }
        omar_coe = env_config_map[env_id][data_type]['omar_coe']
        cql_alpha = env_config_map[env_id][data_type]['cql_alpha']
        # allow CLI overrides (sentinel < 0 means "use the env_config_map default")
        if getattr(args, 'omar_coe', -1.0) is not None and getattr(args, 'omar_coe', -1.0) >= 0:
            omar_coe = args.omar_coe
        if getattr(args, 'cql_alpha_o', -1.0) is not None and getattr(args, 'cql_alpha_o', -1.0) >= 0:
            cql_alpha = args.cql_alpha_o
        cql = True if omar else cql

        init_dict = {
            'args': args,
            'env': env,
            'env_id': env_id,
            'gamma': gamma, 
            'tau': tau, 
            'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'adv_init_params': adv_init_params, 
            'discrete_action': False,
            'cql': cql, 'cql_alpha': cql_alpha, 'lse_temp': lse_temp, 'num_sampled_actions': num_sampled_actions,
            'batch_size': batch_size,
            'gaussian_noise_std': gaussian_noise_std,
            'agent_max_actions': agent_max_actions,
            'omar': omar, 'omar_coe': omar_coe,
            'omar_iters': omar_iters, 'omar_mu': omar_mu, 'omar_sigma': omar_sigma, 'omar_num_samples': omar_num_samples, 'omar_num_elites': omar_num_elites,
            'num_steps': num_steps
        }
        
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance

    def load_pretrained_preys(self, filename):
        if not torch.cuda.is_available():
            save_dict = torch.load(filename, map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(filename)

        if self.env_id in ['simple_tag', 'simple_world']:
            prey_params = save_dict['agent_params'][self.num_predators:]

        for i, params in zip(range(self.num_preys), prey_params):
            self.preys[i].load_params_without_optims(params)

        for p in self.preys:
            p.policy.eval()
            p.target_policy.eval()

    def log_train(self,):
        train_infos = self.loss_dict
        wandb_train_infos = {}
 
        if self.causal:
            wandb_train_infos["Reward_loss"] = train_infos["Reward_loss"].item()
        
            # train_infos["Buffer Prd_rewards Sum (size: {})".format(str(list(self.buffer.sum_Prs.shape)))] = np.mean(self.buffer.sum_Prs)
            # train_infos["Buffer Rs (size: {})".format(str(list(self.buffer.Rs.shape)))] = np.mean(self.buffer.Rs)
            # train_infos["R_MSE"] = train_infos["Reward_loss"][1].item()

            wandb_train_infos["Sparsity_loss"] = train_infos["Sparsity_loss"].item()
            # wandb_train_infos["Dynamic_loss"] = train_infos["Dynamic_loss"][0].item()
        
                    
            # train_infos["Mean Buffer' Rewards"] = np.mean(self.buffer.rewards)
            
            # for i in range(self.num_agents):
            #     if self._pred_r_ratio:
            #         train_infos["Ratio Agent {}".format(i)] = self.buffer.ratio[i].item()
            #     train_infos["Buffer Prd_rewards Agent {}".format(i)] = np.mean(self.buffer.Prs[:,:,i,:])
            #     train_infos["Trainer Prd_rewards Agent {}".format(i)] = self.trainer.Prd_rewards[:,i,:].mean().item()
            
            # train_infos[" Rs_batch"] = self.trainer.Rs_batch.mean().item()
            # train_infos["Tainer Prd_rewards Sum (size: {})".format(str(list(self.trainer.Prd_rewards_sum.shape)))] = self.trainer.Prd_rewards_sum.mean().item()
            
            
            # Sparsity_rate
            wandb_train_infos["s2s_rate"] , wandb_train_infos["s2s_aux_rate"], wandb_train_infos["a2s_rate"], wandb_train_infos["s2r_rate"], wandb_train_infos["a2r_rate"] = \
                    [v.item() for v in train_infos["Sparsity_rate"]]
            
            # Sparsity_loss
            wandb_train_infos["s2s_loss"] = train_infos["s2s_loss"].item()
            wandb_train_infos["s2s_loss_aux"] = train_infos["s2s_loss_aux"].item()
            wandb_train_infos["a2s_loss"] = train_infos["a2s_loss"].item()
            wandb_train_infos["s2r_loss"] = train_infos["s2r_loss"].item()
            wandb_train_infos["a2r_loss"] = train_infos["a2r_loss"].item()
            
            
            #OMAR ploy loss
            
       

            # print("### Tainer Reward ###")
            
            
            # print("Tainer Rs_batch {}. (size: {})".format(train_infos["Tainer Rs_batch (size: {})".format(str(list(self.trainer.Rs_batch.shape)))], str(list(self.trainer.Rs_batch.shape))))
            # print("Tainer Sum Prd_rewards {}. (size: {})".format(train_infos["Tainer Prd_rewards Sum (size: {})".format(str(list(self.trainer.Prd_rewards_sum.shape)))], str(list(self.trainer.Prd_rewards_sum.shape))))
            # print("Tainer Reward_loss is {}.".format(train_infos["Reward_loss"]))
            
            # print("### Buffer Reward ###")
            # print("Buffer Sum Predict Rewards is {}. (size: {})".format(train_infos["Buffer Prd_rewards Sum (size: {})".format(str(list(self.buffer.sum_Prs.shape)))], str(list(self.buffer.sum_Prs.shape))))
            # print("Buffer Sum Team Rewards is {}. (size: {})".format(train_infos["Buffer Rs (size: {})".format(str(list(self.buffer.Rs.shape)))], str(list(self.buffer.Rs.shape))))
            # print("Buffer Reward_MSE is {}.".format(train_infos["Buffer R MSE"]))
        
            # print("### Sparsity and Dyn ###")
            # print("Dynamic_loss is {}.".format(train_infos["Dynamic_loss"]))
            # print("Sparsity_loss is {}.".format(train_infos["Sparsity_loss"]))
        
        
        wandb_train_infos["pol_loss"] = train_infos["pol_loss"]
        if self.omar:
            wandb_train_infos["pred_qvals"] = train_infos["pred_qvals"]
            wandb_train_infos["mimic_term"] = train_infos["mimic_term"]
        wandb.log(wandb_train_infos)
      
        # for k, v in train_infos.items():
            # if self.use_wandb:
                # wandb.log({k: v}, step=total_num_steps)
            # else:
                # self.writter.add_scalars(k, {k: v}, total_num_steps)