#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/JOYFUL"

python train.py \
  --dataset iemocap_4 \
  --modalities atv \
  --graph_mode hybrid_expand \
  --sim_metric cosine \
  --sim_threshold 0.7 \
  --sim_topk 5 \
  --hyper_min_size 3 \
  --hyper_max_size 8 \
  --max_hyperedges_per_dialog 30 \
  --hyper_edge_ratio_cap 1.0 \
  --from_begin \
  --epochs 50 \
  --batch_size 32 \
  --device cpu
