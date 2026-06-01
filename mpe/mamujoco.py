from multiagent_mujoco.src.multiagent_mujoco.mujoco_multi import MujocoMulti
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
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv

import h5py
import numpy as np
import time



def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
    
def eval_policy(agent, config, t, env_args=None):
    render = config.render
    seed = config.seed
    env_name = config.env_id
    eval_episodes = config.eval_episodes
    discrete_action = config.discrete_action
    env = MujocoMulti(env_args=env_args)
    env.seed(seed + 100)

    all_episodes_rewards = []
    for ep_i in range(eval_episodes):
        agent.prep_rollouts(device='cpu')

        env.reset()
        done = False
        episode_reward = 0.
        while not done:
            obs = env.get_obs()

            torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
            torch_agent_actions = agent.step(torch_obs, explore=False)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [ac.squeeze(0) for ac in agent_actions]

            reward, done, info = env.step(actions)
           
            episode_reward += reward

        all_episodes_rewards.append(episode_reward)
    
    mean_episode_reward = np.mean(np.array(all_episodes_rewards))
    return mean_episode_reward

def prepare_output_dir(log_dir, argv=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def offline_train(config):
    outdir = prepare_output_dir(config.dir + '/' + config.env_id, argv=sys.argv)
    
    print('\033[1;32mOutput files are saved in {} \033[1;0m'.format(outdir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if config.use_gpu:
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_num_threads(config.n_training_threads)
    
    env_args = {"scenario": "HalfCheetah-v2",
                  "agent_conf": "2x3",
                  "agent_obsk": 0,
                  "episode_limit": 1000}
    env = MujocoMulti(env_args=env_args)
    env_info = env.get_env_info()
    env.seed(config.seed)
    
    config.batch_size = 256
    config.hidden_dim = 256
    config.lr = 0.0003
    config.tau = 0.005
    config.gamma = 0.99
    
    config.omar_iters = 2
    config.omar_num_samples = 20
    config.omar_num_elites = 5 

    ma_agent = ITD3.init_from_env(
        config,
        env, config.env_id, config.data_type,
        tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim,
        cql=config.cql, lse_temp=config.lse_temp, batch_size=config.batch_size, num_sampled_actions=config.num_sampled_actions,
        omar=config.omar, omar_iters=config.omar_iters, omar_mu=config.omar_mu, omar_sigma=config.omar_sigma, omar_num_samples=config.omar_num_samples, omar_num_elites=config.omar_num_elites, 
        env_info=env_info, num_steps = config.num_steps,
    )
    
    replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space],
        )
    replay_buffer.load_batch_data(config.dataset_dir)

    wandb_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / config.env_name / config.map_name / config.algorithm_name / config.experiment_name
    if not wandb_dir.exists():
        os.makedirs(wandb_dir)
        
    if config.use_wandb:
        from utils.wandb_utils import init_wandb
        config.wandb_dir = wandb_dir
        config.wandb_group = config.map_name
        init_wandb(config)
        
    for t in tqdm(range(config.num_steps + 1)):
        if t % config.eval_interval == 0 or t == config.num_steps:
            eps_rewrad = eval_policy(ma_agent, config, t, env_args=env_args)
            mean_score = 0.
            for index in range(config.episode_length):
                sample_test = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                mean_reward = torch.mean(torch.cat(sample_test[3])).item()
                mean_score += mean_reward
            cnr = [3568.8, -284.0] 
            normalized_score = 100 * (eps_rewrad - cnr[1]) / (cnr[0]-cnr[1])
            print('Step: {} \t Episode Rewards Score: {}'.format(t, normalized_score))
            print('Step: {} \t Evaluation Average Reward: {}'.format(t, eps_rewrad))
            print('Step: {} \t Mean Reward Score in buffer: {}'.format(t, mean_reward))
            print('Step: {} \t Mean Return Score in buffer: {}'.format(t, mean_score))
            if config.use_wandb:
                wandb.log({"Rewards Score:": normalized_score}, step=t)
                wandb.log({"Evaluation Average Reward:": eps_rewrad}, step=t)
                wandb.log({"Mean Reward Score in buffer:": mean_reward}, step=t)
                wandb.log({"Mean Return Score in buffer:": mean_score}, step=t)
                
        if (t % config.steps_per_update) < config.n_rollout_threads:
            ma_agent.prep_training(device='gpu' if config.use_gpu else 'cpu')

            for u_i in range(config.n_rollout_threads):
                nagents = ma_agent.nagents if config.env_id in ['simple_spread', 'HalfCheetah-v2'] else ma_agent.num_predators

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
    config.num_steps = int(1e6)
    # config.steps_per_update = 5
    config.eval_interval = 5000
    config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.seed)
    config.env_id ='HalfCheetah-v2'
    config.env_name='HalfCheetah-v2'
    config.map_name='HalfCheetah-v2'
    offline_train(config)
