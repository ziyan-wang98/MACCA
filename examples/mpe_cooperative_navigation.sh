#!/bin/bash
# MACCA-OMAR on MPE Cooperative Navigation (CN / simple_spread).
# Reproduces Table 1 CN-Expert (paper 111.7; this config reaches ~113 best-eval).
# Data types: expert | medium | medium-replay | random
cd "$(dirname "$0")/../mpe"

python main.py \
  --env_id simple_spread \
  --data_type expert \
  --seed 0 \
  --algorithm_name causal_omar \
  --use_gpu 1 \
  --experiment_name macca_cn \
  --dataset_dir ./datasets
