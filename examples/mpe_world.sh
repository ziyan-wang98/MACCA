#!/bin/bash
# MACCA-OMAR on MPE Simple-World (WORLD / simple_world).
# Reproduces Table 1 WORLD-Expert (paper 107.4); seed 1 reaches ~139.
# Needs datasets/simple_world/<quality>/ and the pretrained prey model pretrained_adv_model.pt.
# For non-expert data, add the stability flags:
#   --data_type medium --rew_sum_norm --causal_grad_clip 1.0 --causal_lr_decay
cd "$(dirname "$0")/../mpe"

python main.py \
  --env_id simple_world \
  --data_type expert \
  --seed 1 \
  --algorithm_name causal_omar \
  --use_gpu 1 \
  --experiment_name macca_world \
  --dataset_dir ./datasets
