import argparse
import time
import numpy as np
from utils.save import Saver

def get_config():
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # Original parameters
    parser.add_argument("--dir", help="Name of directory to store model/training contents", type=str, default='results')

    parser.add_argument("--env_id", help="Name of environment", type=str, default='simple_spread')
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=1, type=int)
    parser.add_argument("--discrete_action", action='store_true', default=False)
    parser.add_argument("--use_gpu", default=1, type=int)

    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument('--num_updates', default=1, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)

    parser.add_argument('--gaussian_noise_std', default=0.1, type=float)

    parser.add_argument("--data_type", default='medium', type=str)
    parser.add_argument('--dataset_dir', default='./datasets', type=str)

    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--eval_interval', default=1000, type=int)
    parser.add_argument('--num_steps', default=int(1e5), type=int)

    parser.add_argument('--cql', default=0, type=int)
    parser.add_argument('--cql_alpha', default=1.0, type=float)
    parser.add_argument("--lse_temp", default=1.0, type=float)
    parser.add_argument('--num_sampled_actions', default=10, type=int) 
    parser.add_argument('--cql_sample_noise_level', default=0.2, type=float)

    parser.add_argument('--omar', default=0, type=int)
    parser.add_argument('--omar_coe', default=-1.0, type=float, help="OMAR tau (mimic weight); <0 = use per-dataset default from env_config_map")
    parser.add_argument('--cql_alpha_o', default=-1.0, type=float, help="CQL alpha override; <0 = use per-dataset default from env_config_map")
    parser.add_argument('--omar_iters', default=3, type=int)
    parser.add_argument('--omar_mu', default=0., type=float)
    parser.add_argument('--omar_sigma', default=2.0, type=float)
    parser.add_argument('--omar_num_samples', default=10, type=int)
    parser.add_argument('--omar_num_elites', default=10, type=int)


    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='causal_omar', choices=["causal_omar", "omar", "causal_cql","cql","itd3"])
    parser.add_argument("--render", action='store_true', default=False, help="render the evalutation process or not.")
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    
    parser.add_argument("save_rews", action='store_false', default=True, help="whether to save the results or not.")
    

    # WandB
    # parser.add_argument("--user_name", type=str, default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument('--tag', help='the terminal tag in logger', type=str, default='')
    
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")
    parser.add_argument('--wandb-project', default='causal_offline', type=str, help='WandB "Project"')
    parser.add_argument('--wandb-entity', default='', type=str, help='WandB username (entity).')
    parser.add_argument('--wandb-job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb-tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb-key', default='', type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb-dir', default=None, type=str, help='the place to save WandB files')
    parser.add_argument('--wandb-experiment', default='', type=str, help='Identifier to specify the experiment')
    parser.add_argument('--timestamp', default=time.strftime('-(%Y-%m-%d-%H_%M_%S)') + '_' + str(np.random.randint(100)), type=str, help='Timestamp')

    parser.add_argument('--IL', default=1, type=int, help='whether to use Independent Critic')
    
    # causal credit assignment parameters
    parser.add_argument("--use_causal", action='store_true', default=False, help="Whether to use causal reward assignment.")
    
    parser.add_argument("--ss_sparsity_coef", type=float, default=0, help="Sparsity_loss : S2S")
    
    parser.add_argument("--ss_sparsity_coef_aux", type=float, default=0, help="Sparsity_loss : S2S_aux")
    
    parser.add_argument("--as_sparsity_coef", type=float, default=0, help="Sparsity_loss : A2S")
     
    # parser.add_argument("--sr_sparsity_coef", type=float, default=1e-8, help="Sparsity_loss : S2R")
    
    parser.add_argument("--sr_sparsity_coef", type=float, default=0.007, help="Sparsity_loss : S2R")
      
    # parser.add_argument("--ar_sparsity_coef", type=float, default=1e-2, help="Sparsity_loss : A2R")

    parser.add_argument("--ar_sparsity_coef", type=float, default=0.007, help="Sparsity_loss : A2R")
    
    parser.add_argument("--paper_rew", action='store_true', default=False, help="Use paper's reward predictor psi_r (3 FC layers x 256) instead of the default 5x hidden_dim.")
    parser.add_argument("--dynamic_mask", action='store_true', default=False, help="Use the paper's dynamic causal-structure predictor psi_g(s,a,i) (per-sample masks) instead of the static learnable params.")
    parser.add_argument("--critic_ln", action='store_true', default=False, help="Use LayerNorm critic (replaces BatchNorm) — offline-RL stabilization trick to prevent value extrapolation/divergence on suboptimal data.")
    parser.add_argument("--policy_ln", action='store_true', default=False, help="Use LayerNorm policy/actor (replaces BatchNorm).")
    parser.add_argument("--rew_norm", action='store_true', default=False, help="Normalize (scale) the redistributed rewards by a running std before the Q-target — offline-RL reward-scale trick.")
    parser.add_argument("--td3bc_coef", default=0.0, type=float, help="TD3+BC: >0 adds a Q-normalized behavior-cloning anchor to the dataset action in the actor loss (alpha). Strong on suboptimal data.")
    parser.add_argument("--awr_temp", default=0.0, type=float, help="AWR: >0 enables advantage-weighted regression in the actor loss (temperature, adv-std-normalized). Pulls policy toward high-advantage (p90) dataset actions, in-distribution.")
    parser.add_argument("--awr_coef", default=1.0, type=float, help="Weight on the AWR advantage-weighted-BC term (only used when --awr_temp>0).")
    parser.add_argument("--iql_tau", default=0.0, type=float, help="IQL expectile (0..1, e.g. 0.7-0.9): >0 trains a V(s) expectile-value net and uses adv=Q(s,a_data)-V(s) as the AWR baseline. In-distribution, no OOD queries.")
    parser.add_argument("--test_cem", action='store_true', default=False, help="Test-time critic-guided action selection (CEM around the policy action, pick highest-Q). Lifts eval performance without changing training.")
    parser.add_argument("--save_gif", action='store_true', default=False, help="After training, render the best of several rollouts of the learned MACCA policy to a showcase GIF in gifs/<env_id>.gif.")
    parser.add_argument("--num_steps_override", default=0, type=int, help=">0 overrides the hardcoded num_steps (for longer-training ablations).")
    parser.add_argument("--pre_trained", action='store_true', default=False, help="Pre-trained model or not.")
    # reward shaping ablation:
    parser.add_argument("--use_lambda_reward", action='store_true', default=False, help="Use lambda * pred_reward + (1-lamda) * true_reward")
    parser.add_argument("--use_pred_rewards", action='store_true', default=False, help="Use pred_reward predict by causal structure")
    parser.add_argument("--use_mean_R", action='store_true', default=False, help="Use mean true reward as agent's reward")
    parser.add_argument("--use_gr_r", action='store_true', default=False, help="Use ground truth reward as agent's reward")
    parser.add_argument("--use_sum_pref_r", action='store_true', default=False, help="Use sum of predicted reward as agent's reward")
    parser.add_argument("--pred_r_ratio", action='store_true', default=False, help="Use sum of true reward as agent's reward")
    # optimizer parameters
    # parser.add_argument("--lr", type=float, default=5e-4,
    #                     help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    
    
    
    #  parser.add_argument("--r_lr", type=float, default=1e-4,
    #                     help='learning rate (default: 5e-4)')
    parser.add_argument("--r_lr", type=float, default=5e-2,
                        help='learning rate (default: 5e-2)')
    
    parser.add_argument("--d_lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    
    parser.add_argument("--s_lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    
    parser.add_argument("--temperature", type=float, default=0.5, help='temp (default: 0.5)')

    
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    return parser
