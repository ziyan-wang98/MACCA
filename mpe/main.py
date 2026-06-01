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
from utils.save import Saver
from multiagent_mujoco.src.multiagent_mujoco.mujoco_multi import MujocoMulti

import h5py

# try:
    
# except:
#     print ('MujocoMulti not installed')

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
    saver = config.saver
    
    if env_name in ['HalfCheetah-v2']:
        # Check the name
        # import pdb; pdb.set_trace()
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
    else:
        avg_predator_return = 0.
    
        env = make_parallel_env(env_name, 1, seed + 100, discrete_action)

        for ep_i in range(0, eval_episodes):
            obs = env.reset()
            agent.prep_rollouts(device='cuda' if config.use_gpu else 'cpu')

            for et_i in range(config.episode_length):
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(agent.nagents)]
                if getattr(config, 'test_cem', False):
                    torch_agent_actions = agent.step_test_cem(torch_obs)
                else:
                    torch_agent_actions = agent.step(torch_obs, explore=False)
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                
                next_obs, rewards, dones, infos = env.step(actions)
                if render:
                    if t >2e1 and rewards[0].mean() < 20.:
                        env.envs[0].render('human')
                        for i in range(int(5e6)):
                            m = 0
                        print(rewards[0])
                if env_name in ['simple_tag', 'simple_world']:
                    avg_predator_return += rewards[0][0]
                else:
                    avg_agent_reward = np.mean(rewards[0])
                    avg_predator_return += avg_agent_reward

                obs = next_obs
                
        
        avg_predator_return /= eval_episodes
        
        avg_predator_reward = avg_predator_return / config.episode_length
        
        return avg_predator_return, avg_predator_reward
    
def save_model(agent, config, t):
    if config.save_model:
        model_dir = os.path.join(config.dir, config.env_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if config.use_gpu:
            agent.save(os.path.join(model_dir, str(t) + '_actor.pth'), os.path.join(model_dir, str(t) + '_critic.pth'))
        else:
            agent.save(os.path.join(model_dir, str(t) + '_actor.pth'), os.path.join(model_dir, str(t) + '_critic.pth'))

def prepare_output_dir(log_dir, argv=None):
    os.makedirs(log_dir, exist_ok=True)
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

    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, config.discrete_action)
        env_args, env_info = None, None
    else:
        # config.env_id = 'half_cheetah'
        mujoco_env_args = {"scenario": "HalfCheetah-v2",
                  "agent_conf": "2x3",
                  "agent_obsk": 0,
                  "episode_limit": 1000}
        # env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
        env = MujocoMulti(env_args = mujoco_env_args)
        

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

    
    if config.env_id in ['simple_tag', 'simple_world']:
        pretrained_model_dir = './datasets/{}/pretrained_adv_model.pt'.format(config.env_id)
        ma_agent.load_pretrained_preys(pretrained_model_dir)

    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [obsp.shape[0] for obsp in env.observation_space],
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space],
        )
    else:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space],
        )
    replay_buffer.load_batch_data(config.dataset_dir)

    

    if config.use_gpu:
        print('Using GPU')
    
    if config.env_id == 'simple_spread':
        config.env_name = 'MA-Particle-Env'
        config.map_name = 'Cooperative_Navigation'
    elif config.env_id == 'simple_tag':
        config.env_name = 'MA-Particle-Env'
        config.map_name = 'Predator_Prey'
    elif config.env_id == 'simple_world':
        config.env_name = 'MA-Particle-Env'
        config.map_name = 'Multi-Agent_World'
    elif config.env_id == 'HalfCheetah-v2':
        config.env_name = 'HalfCheetah-v2'
    
    wandb_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / config.env_name / config.map_name / config.algorithm_name / config.experiment_name
    
    os.makedirs(wandb_dir, exist_ok=True)
        
    if config.use_wandb:
        from utils.wandb_utils import init_wandb
        config.wandb_dir = wandb_dir
        config.wandb_group = config.map_name
        init_wandb(config)
        
    # if config.pre_trained:
    
    saver = Saver(config)
        

    for t in tqdm(range(config.num_steps + 1)):
        if t % config.eval_interval == 0 or t == config.num_steps:
            eval_return, eval_rewrad = eval_policy(ma_agent, config, t, env_args=env_args)
            mean_score = 0.
            for index in range(config.episode_length):
                sample_test = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                mean_reward = torch.mean(torch.cat(sample_test[2])).item()
                mean_score += mean_reward
    
            if config.env_id == 'simple_spread':
                RR = 6.133883476257324
                ER = 20.802875518798828
                cnr = [516.8, 159.8] # 100.569 /357
            elif config.env_id == 'simple_tag':
                RR = 0.0
                ER = 3.0
                cnr = [90.637, -2.5]  # 100.569 /357
            elif config.env_id == 'simple_world':
                RR = 6.133883476257324
                ER = 20.802875518798828
                cnr = [34.661, -8.734]
            elif config.env_id == 'HalfCheetah-v2':
                RR = 6.133883476257324
                ER = 20.802875518798828
                cnr = [3568.8, -284.0] 
                
            
            
            normalized_score = 100 * (eval_rewrad - RR) / (ER-RR)
            normalized_score_test = 100 * (eval_return - cnr[1]) / (cnr[0]-cnr[1])     
                

            print('Step: {} \t Rewards Score: {}'.format(t, normalized_score))
            print('Step: {} \t Return Score: {}'.format(t, normalized_score_test))
            print('Step: {} \t Evaluation Average Reward: {}'.format(t, eval_rewrad))
            print('Step: {} \t Evaluation Average Return: {}'.format(t, eval_return))
            print('Step: {} \t Mean Reward Score in buffer: {}'.format(t, mean_reward))
            print('Step: {} \t Mean Return Score in buffer: {}'.format(t, mean_score))
            if config.use_wandb:
                wandb.log({"Rewards Score:": normalized_score}, step=t)
                wandb.log({"Return Score:": normalized_score_test}, step=t)
                wandb.log({"Evaluation Average Reward:": eval_rewrad}, step=t)
                wandb.log({"Evaluation Average Return:": eval_return}, step=t)
                wandb.log({"Mean Reward Score in buffer:": mean_reward}, step=t)
                wandb.log({"Mean Return Score in buffer:": mean_score}, step=t)
                
                    
                
        if (t % config.steps_per_update) < config.n_rollout_threads:
            
            # if t % 1000 == 0:
            #     if config.use_causal:
            #         saver.save_cs(ma_agent, t)
            
            ma_agent.prep_training(device='gpu' if config.use_gpu else 'cpu')

            for u_i in range(config.n_rollout_threads):
                nagents = ma_agent.nagents if config.env_id in ['simple_spread', 'HalfCheetah-v2'] else ma_agent.num_predators

                for a_i in range(nagents):
                    
                    sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                    
                    ma_agent.update(sample, a_i, t)

                ma_agent.update_all_targets()

    # === Showcase GIF of the learned MACCA policy (opt-in --save_gif): best of N rollouts ===
    # Headless matplotlib (Agg) renderer drawn from the MPE world entities (no OpenGL/pyglet/GLU needed).
    if getattr(config, 'save_gif', False):
        try:
            import imageio
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            def _render_mpl(raw_env):
                world = raw_env.world
                fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=80)
                for e in world.entities:
                    col = np.clip(np.array(getattr(e, 'color', [0.35, 0.35, 0.35]), dtype=float), 0, 1)
                    is_lm = ('landmark' in type(e).__name__.lower()) or (not getattr(e, 'movable', True))
                    ax.add_patch(plt.Circle(e.state.p_pos, e.size, color=col, alpha=0.55 if is_lm else 0.95,
                                            ec='k', lw=0.5, zorder=1 if is_lm else 2))
                ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect('equal'); ax.axis('off')
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3).copy()
                plt.close(fig)
                return frame

            gdir = 'gifs'; os.makedirs(gdir, exist_ok=True)
            genv = make_parallel_env(config.env_id, 1, config.seed + 7, config.discrete_action)
            ma_agent.prep_rollouts(device='cpu')
            best_frames, best_ret = None, -1e18
            for ep in range(8):
                obs = genv.reset(); frames = []; ret = 0.0
                for et in range(config.episode_length):
                    frames.append(_render_mpl(genv.envs[0]))
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(ma_agent.nagents)]
                    tacs = ma_agent.step(torch_obs, explore=False)
                    acs = [ac.data.numpy() for ac in tacs]
                    actions = [[ac[i] for ac in acs] for i in range(config.n_rollout_threads)]
                    obs, rews, dones, infos = genv.step(actions)
                    ret += float(rews[0][0] if config.env_id in ['simple_tag', 'simple_world'] else np.mean(rews[0]))
                frames.append(_render_mpl(genv.envs[0]))
                if ret > best_ret:
                    best_ret, best_frames = ret, frames
            imageio.mimsave(os.path.join(gdir, config.env_id + '.gif'), best_frames, duration=0.1)
            print('[GIF] saved gifs/{}.gif  (best rollout return={:.1f}, {} frames)'.format(config.env_id, best_ret, len(best_frames)))
            genv.close()
        except Exception as e:
            import traceback; traceback.print_exc(); print('[GIF] render failed:', e)

    env.close()
    if config.use_wandb:
        from utils.wandb_utils import finish_wandb
        finish_wandb(config)
        
if __name__ == '__main__':
    
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
        # break no algorithm name
        NotImplementedError
        
    if config.env_id in ['simple_spread', 'simple_tag','simple_world']:
        config.num_steps = 200000
        if config.env_id == 'simple_spread' and config.data_type == 'random':
            config.num_steps = 600000
        elif config.env_id == 'simple_tag' and config.data_type == 'medium-replay':
            config.num_steps = 100000
        elif config.env_id == 'simple_world' and config.data_type == 'random':
            config.num_steps = 600000
    elif config.env_id == 'HalfCheetah-v2':
        config.num_steps = int(1e6)
        config.steps_per_update = 10
        config.eval_interval = 5000

    if getattr(config, 'num_steps_override', 0) and config.num_steps_override > 0:
        config.num_steps = config.num_steps_override

    config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.seed)
    
    config.saver = saver = Saver(config)
    offline_train(config)

