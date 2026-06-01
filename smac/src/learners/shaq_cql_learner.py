import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, SGD
from modules.mixers.shaq import SHAQMixer
from .cq_learner import CQLearner

class SHAQCQLLearner(CQLearner):
    def __init__(self, mac, scheme, logger, args):
        super(SHAQCQLLearner, self).__init__(mac, scheme, logger, args)
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.last_mixer_update_episode = 0
        self.last_sample_coalition_episode = 0
        self.mixer = SHAQMixer(args)
        self.params_mixer = list(self.mixer.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_mixer = RMSprop(params=self.params_mixer, lr=args.alpha_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        one_hot_actions = th.nn.functional.one_hot(actions, num_classes=self.args.n_actions)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        total_cql_loss = 0
        total_masked_td_error = 0
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        
        
        test = True
        if test:
            n_agents = actions.shape[2]
            n_actions = mac_out.shape[3]
            sample_actions_num = self.args.raw_sample_actions
            bs = actions.shape[0]
            ts = actions.shape[1]
            
            cql_loss = 0
            
            for agent_i in range(n_agents):
                # CQL Loss for each agent: Start
                repeat_avail_actions = th.repeat_interleave(avail_actions[:, :-1, agent_i].unsqueeze(0), repeats=sample_actions_num, dim=0) # shape = (sample_actions_num, bs, ts, na)
                total_random_actions = th.randint(low=0, high=n_actions, size=(sample_actions_num, bs, ts, 1)).to(mac_out.device) # shape = (sample_actions_num, bs, ts, 1)
                # chosen_if_avail = th.gather(repeat_avail_actions, dim=-1, index=total_random_actions).min(1)[0]
                chosen_if_avail = th.gather(repeat_avail_actions, dim=-1, index=total_random_actions).squeeze(-1).min(dim=0)[0]
                repeat_mac_out = th.repeat_interleave(mac_out[:, :-1, agent_i].unsqueeze(0), repeats=sample_actions_num, dim=0)
                random_chosen_action_qvals = th.gather(repeat_mac_out, dim=-1, index=total_random_actions).squeeze(-1)
                random_chosen_action_qvals = random_chosen_action_qvals.view(sample_actions_num, bs, ts)
                negative_sampling = th.logsumexp(random_chosen_action_qvals*chosen_if_avail, dim=0)
                dataset_expec = chosen_action_qvals[:, :, agent_i]
                agent_cql_loss = self.args.global_cql_alpha * ((negative_sampling-dataset_expec)* mask[:, :, 0]).sum() / mask.sum()
                cql_loss += agent_cql_loss
        else:
            # CQL Loss: Start
            n_actions = mac_out.shape[3]
            sample_actions_num = self.args.raw_sample_actions
            bs = actions.shape[0]
            ts = actions.shape[1]
            repeat_avail_actions = th.repeat_interleave(avail_actions[:,:-1].unsqueeze(0), repeats=sample_actions_num, dim=0)
            total_random_actions = th.randint(low=0, high=n_actions, size=(sample_actions_num, bs, ts, 1)).to(mac_out.device)
            chosen_if_avail = th.gather(repeat_avail_actions, dim=-1, index=total_random_actions).min(2)[0]
            repeat_mac_out = th.repeat_interleave(mac_out[:,:-1].unsqueeze(0), repeats=sample_actions_num, dim=0)
            random_chosen_action_qvals = th.gather(repeat_mac_out, dim=-1, index=total_random_actions).squeeze(-1)
            random_chosen_action_qvals = random_chosen_action_qvals.view(bs*sample_actions_num, ts, -1)
            negative_sampling = th.logsumexp(random_chosen_action_qvals*chosen_if_avail, dim=0)
            dataset_expec = chosen_action_qvals
            cql_loss = self.args.global_cql_alpha * ((negative_sampling-dataset_expec)* mask).sum() / mask.sum()
            # CQL Loss: End
        
        
        # generate a filter for selecting the agents with the max-action
        _mac_out_detach = mac_out.clone().detach()
        _mac_out_detach[avail_actions == 0] = -9999999
        _cur_max_actions = _mac_out_detach[:, :-1].max(dim=3, keepdim=True)[1].squeeze(3)
        max_filter = (actions.detach().squeeze(3)==_cur_max_actions).float()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        
        
        
        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals, w_est = self.mixer(batch["state"][:, :-1], one_hot_actions, chosen_action_qvals, max_filter, target=False, manual_alpha_estimates=self.args.manual_alpha_estimates)
            target_max_qvals = self.mixer(batch["state"][:, 1:], one_hot_actions, target_max_qvals, max_filter, target=True, manual_alpha_estimates=self.args.manual_alpha_estimates)

        N = getattr(self.args, "n_step", 1)
        if N == 1:
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        else:
            # N step Q-Learning targets
            n_rewards = th.zeros_like(rewards)
            gamma_tensor = th.tensor([self.args.gamma**i for i in range(N)], dtype=th.float, device=n_rewards.device)
            steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(N).long()
            for i in range(batch.max_seq_length - 1):
                n_rewards[:,i,0] = ((rewards * mask)[:,i:i+N,0] * gamma_tensor[:(batch.max_seq_length - 1 - i)]).sum(dim=1)
            indices = th.linspace(0, batch.max_seq_length-2, steps=batch.max_seq_length-1, device=steps.device).unsqueeze(1).long()
            n_targets_terminated = th.gather(target_max_qvals*(1-terminated),dim=1,index=steps.long()+indices-1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data +  CQL Loss
        loss = (masked_td_error ** 2).sum() / mask.sum() + cql_loss
        
        # Optimise
        self.optimiser.zero_grad()
        self.optimiser_mixer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        grad_norm_mixer = th.nn.utils.clip_grad_norm_(self.params_mixer, self.args.grad_norm_clip)
        self.optimiser.step()
        self.optimiser_mixer.step()

        # Periodically update target Q-values
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("grad_norm_mixer", grad_norm_mixer, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            agent_utils = (th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("agent_utils", agent_utils, t_env)
            self.logger.log_stat("w_est", ( w_est * (1 - max_filter) * mask.expand_as(w_est) ).sum().item() / ( ( (1 - max_filter) * mask.expand_as(w_est) ).sum().item() ), t_env)
            self.log_stats_t = t_env


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")


    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()


    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))


    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))