#!/bin/bash
# MACCA-OMAR on MPE Predator-Prey (PP / simple_tag).
# Reproduces Table 1 PP-Expert (paper 111.0; this config reaches ~113 best-eval).
# Needs datasets/simple_tag/<quality>/ and the pretrained prey model pretrained_adv_model.pt.
cd "$(dirname "$0")/../mpe"

python main.py \
  --env_id simple_tag \
  --data_type expert \
  --seed 0 \
  --algorithm_name causal_omar \
  --use_gpu 1 \
  --experiment_name macca_pp \
  --dataset_dir ./datasets
