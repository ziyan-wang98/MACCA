
import argparse
import torch
import time
import os, sys, tempfile
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from algorithms.ITD3 import ITD3
import datetime
import random
import copy
import shutil
import wandb
from tqdm import tqdm
from config import get_config
from utils.smac_env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

import h5py
import numpy as np
import time

from utils.h5 import read_h5
from starcraft2.StarCraft2_Env import StarCraft2Env
from starcraft2.smac_maps import get_map_params



"""Train script for SMAC."""

def make_train_env(config):
    def get_env_fn(rank):
        def init_env():
            if config.env_name == "StarCraft2":
                env = StarCraft2Env(config)
            else:
                print("Can not support the " + config.env_name + "environment.")
                raise NotImplementedError
            env.seed(config.seed + rank * 1000)
            return env

        return init_env

    env = StarCraft2Env(config)
    return env
    # if config.n_rollout_threads == 1:
    #     return ShareDummyVecEnv([get_env_fn(0)])
    # else:
    #     return ShareSubprocVecEnv([get_env_fn(i) for i in range(config.n_rollout_threads)])

def make_eval_env(config):
    def get_env_fn(rank):
        def init_env():
            if config.env_name == "StarCraft2":
                env = StarCraft2Env(config)
            else:
                print("Can not support the " + config.env_name + "environment.")
                raise NotImplementedError
            env.seed(config.seed * 50000 + rank * 10000)
            return env

        return init_env

    if config.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(config.n_eval_rollout_threads)])


def log_env(self, config, env_infos, t):
    for k, v in env_infos.items():
        if len(v) > 0:
            if config.use_wandb:
                wandb.log({k: np.mean(v)}, step=t)

def eval_policy(agent, config, t):
    debug = True
    if not debug:
        eval_episodes = config.eval_episodes
        # eval_envs = make_eval_env(config)
        eval_envs = StarCraft2Env(config)
        eval_win_rates = []
        all_episodes_rewards = []
        for ep_i in range(eval_episodes):
            if config.use_gpu:
                agent.prep_rollouts(device='gpu')
            else:
                agent.prep_rollouts(device='cpu')
            eval_envs.reset()
            terminated = False
            win_rates = []
            while not terminated:
                obs = eval_envs.get_obs()
                # state = env.get_state()
                # env.render()  # Uncomment for rendering
                torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
                torch_agent_actions = agent.step(torch_obs, explore=False)
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                actions = [ac.squeeze(0) for ac in agent_actions]

                result = eval_envs.step(actions)
                team_reward = np.arrary(result[2]).sum()
                terminated = all(result[3])
                episode_team_reward += team_reward
            all_episodes_team_rewards.append(episode_team_reward)
            win_rate = eval_envs.get_stats()['win_rate']
            win_rates.append(win_rate)
        mean_win_rate = np.mean(np.array(eval_win_rates))
        mean_episode_reward = np.mean(np.array(all_episodes_team_rewards))
        
        return mean_win_rate, mean_episode_reward
    else:
        eval_battles_won = 0
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        # eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
    
        while True:
            obs = eval_envs.get_obs()
            # state = env.get_state()
            # env.render()  # Uncomment for rendering
            torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
            torch_agent_actions = agent.step(torch_obs, explore=False)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [ac.squeeze(0) for ac in agent_actions]
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            # eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            # eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(config.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= config.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                log_env(eval_env_infos,config, t)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if config.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step= t)



def offline_train(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(config.n_training_threads)
    
    # env = make_train_env(config)
    env = StarCraft2Env(config)
    env_info = env.get_env_info()
    env.seed(config.seed)
    
    ma_agent = ITD3.init_from_env(
        config,
        env, config.env_id, config.data_type,
        tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim,
        cql=config.cql, lse_temp=config.lse_temp, batch_size=config.batch_size, num_sampled_actions=config.num_sampled_actions,
        omar=config.omar, omar_iters=config.omar_iters, omar_mu=config.omar_mu, omar_sigma=config.omar_sigma, omar_num_samples=config.omar_num_samples, omar_num_elites=config.omar_num_elites, 
        env_info=env_info, num_steps = config.num_steps,
    )
    
    actions_h, actions_onehot_h, avail_actions_h, filled_h, obs_h, reward_h, state_h, terminated_h = read_h5(config.dataset_dir)
    
    replay_buffer = ReplayBuffer(
        config.buffer_length, ma_agent.nagents,
        [env_info['obs_shape'] for _ in env.observation_space],
        [acsp.shape[0] for acsp in env.action_space],
        is_mamujoco=False,
        state_dims=[env_info['state_shape'] for _ in env.observation_space],
    )
    
    replay_buffer.load_h5(actions_h, actions_onehot_h, avail_actions_h, filled_h, obs_h, reward_h, state_h, terminated_h)

    wandb_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / config.env_name / config.map_name / config.algorithm_name / config.experiment_name
    if not wandb_dir.exists():
        os.makedirs(wandb_dir)
    if config.use_wandb:
        from utils.wandb_utils import init_wandb
        config.wandb_dir = wandb_dir
        config.wandb_group = config.map_name
        init_wandb(config)  
        
    for t in tqdm(range(config.num_steps + 1)):
        
        ####### EVALUATION ########
        if t % config.eval_interval == 0 or t == config.num_steps:
            
            
            eval_policy(ma_agent, config, t)
            
            
            
            # mean_win_rate, mean_episode_reward = eval_policy(ma_agent, config, t)
            # print('Step: {} \t Evaluation Win Rate: {}'.format(t, mean_win_rate))
            # print('Step: {} \t Evaluation Average Reward: {}'.format(t, mean_episode_reward))
            # if config.use_wandb:
            #     wandb.log({'eval_win_rate': mean_win_rate, 'eval_episode_reward': mean_episode_reward, 'step': t})
            #     eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
  

                
                
                
                
        
        ####### TRAINING ########
        if (t % config.steps_per_update) < config.n_rollout_threads:
            ma_agent.prep_training(device='gpu' if config.use_gpu else 'cpu')
            for u_i in range(config.n_rollout_threads):
                nagents = ma_agent.nagents
                for a_i in range(nagents):
                    sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                    ma_agent.update(sample, a_i, t)
                ma_agent.update_all_targets()
    env.close()
    if config.use_wandb:
        from utils.wandb_utils import finish_wandb
        finish_wandb(config)
        









if __name__ == "__main__":
    parser = get_config()
    
    
    
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_false', default=True)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    # parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_state_agent", action='store_false', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--use_stacked_frames", action='store_true', default=False)
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    
    config = parser.parse_args()
    
    
    if config.algorithm_name == 'causal_omar':
        config.use_causal = True
        config.omar = True
        config.cql = False
    elif config.algorithm_name == 'causal_cql':
        config.use_causal = True
        config.omar = False
        config.cql = True
    elif config.algorithm_name == 'cql':
        config.use_causal = False
        config.omar = False
        config.cql = True
    elif config.algorithm_name == 'omar':
        config.use_causal = False
        config.omar = True
        config.cql = False
    elif config.algorithm_name == 'itd3':
        config.use_causal = False
        config.omar = False
        config.cql = False
    else:
        NotImplementedError

    config.env_name = 'StarCraft2'
    config.map_name = '3m'
    config.env_id = 'StarCraft2'
    # config.dataset_dir = config.dataset_dir + '/' + config.smac_map + '.h5'
    # actions_h, actions_onehot_h, avail_actions_h, filled_h, obs_h, reward_h, state_h, terminated_h = read_h5(config.dataset_dir)
    
    offline_train(config)
    
  
    

    
    
    
    
    
    
    
    

