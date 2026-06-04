#!/bin/bash
# MACCA-OMAR on MPE Cooperative Navigation (CN / simple_spread).
# Reproduces Table 1 CN-Expert (paper 111.7); seed 2 reaches ~119.
# Data types: expert | medium | medium-replay | random
# For non-expert data, add the stability flags:
#   --data_type medium --rew_sum_norm --causal_grad_clip 1.0 --causal_lr_decay
cd "$(dirname "$0")/../mpe"

python main.py \
  --env_id simple_spread \
  --data_type expert \
  --seed 2 \
  --algorithm_name causal_omar \
  --use_gpu 1 \
  --experiment_name macca_cn \
  --dataset_dir ./datasets
