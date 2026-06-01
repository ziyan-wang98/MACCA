#!/bin/bash
# MACCA-OMAR on MPE Simple-World (WORLD / simple_world).
# Reproduces Table 1 WORLD-Expert (paper 107.4; this config reaches ~121 best-eval).
# Needs datasets/simple_world/<quality>/ and the pretrained prey model pretrained_adv_model.pt.
cd "$(dirname "$0")/../mpe"

python main.py \
  --env_id simple_world \
  --data_type expert \
  --seed 0 \
  --algorithm_name causal_omar \
  --use_gpu 1 \
  --experiment_name macca_world \
  --dataset_dir ./datasets
