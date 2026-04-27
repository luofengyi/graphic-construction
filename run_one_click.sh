#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/JOYFUL"

# One-click pipeline:
# 1) smoke test with epochs=1 (seed=24)
# 2) full baseline runs with fixed seeds (24,42,77)
# Each run will auto-save best metrics, classification reports and confusion matrices.
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
  --run_smoke_and_baseline \
  --smoke_epochs 1 \
  --epochs 50 \
  --baseline_seed 24 \
  --device cuda:0

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
  --run_seeds 24,42,77 \
  --device cuda:0
