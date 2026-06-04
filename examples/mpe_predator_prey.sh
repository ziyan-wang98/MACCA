#!/bin/bash
# MACCA-OMAR on MPE Predator-Prey (PP / simple_tag).
# Reproduces Table 1 PP-Expert (paper 111.0); seed 0 reaches ~138.
# Needs datasets/simple_tag/<quality>/ and the pretrained prey model pretrained_adv_model.pt.
# For non-expert data, add the stability flags:
#   --data_type medium --rew_sum_norm --causal_grad_clip 1.0 --causal_lr_decay
cd "$(dirname "$0")/../mpe"

python main.py \
  --env_id simple_tag \
  --data_type expert \
  --seed 0 \
  --algorithm_name causal_omar \
  --use_gpu 1 \
  --experiment_name macca_pp \
  --dataset_dir ./datasets
