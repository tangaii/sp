#!/bin/sh
set -eu

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

exec python -u main.py \
  --datadir ./data/ \
  --datname TADPOLE \
  --gpu -1 \
  --nepoch 300 \
  --early 300 \
  --lr 0.012 \
  --reg 0.0036 \
  --dropout 0 \
  --nlayer 2 \
  --n_hidden 10 \
  --n_head 4 \
  --nmodal 6 \
  --th 0.9 \
  --GC_mode weighted-cosine \
  --MP_mode GCN \
  --MF_mode concat \
  --alpha 1 \
  --theta_smooth 1 \
  --theta_degree 0.5 \
  --theta_sparsity 0 \
  --mode simple-2 \
  --seed 0 \
  --no-save-visualizations
