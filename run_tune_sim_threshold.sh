#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/JOYFUL"

# Quick screening setup:
# - Fixed seeds: 24,42,77
# - Sweep sim_threshold: 0.6, 0.65, 0.7, 0.75
# - Keep other baseline params fixed
# You can override EPOCHS from shell, e.g. EPOCHS=15 bash run_tune_sim_threshold.sh
EPOCHS="${EPOCHS:-20}"
SEEDS="${SEEDS:-24,42,77}"

for THR in 0.6 0.65 0.7 0.75; do
  OUT_DIR="./run_outputs/sim_threshold_${THR}"
  echo "=== Running sim_threshold=${THR}, seeds=${SEEDS}, epochs=${EPOCHS} ==="

  python train.py \
    --dataset iemocap_4 \
    --modalities atv \
    --graph_mode hybrid_expand \
    --sim_metric cosine \
    --sim_threshold "${THR}" \
    --sim_topk 5 \
    --hyper_min_size 3 \
    --hyper_max_size 8 \
    --max_hyperedges_per_dialog 30 \
    --hyper_edge_ratio_cap 1.0 \
    --from_begin \
    --epochs "${EPOCHS}" \
    --run_seeds "${SEEDS}" \
    --output_dir "${OUT_DIR}" \
    --device cuda:0

  python summarize_runs.py \
    --output_dir "${OUT_DIR}" \
    --seeds "${SEEDS}" \
    --csv_name "summary_table.csv" \
    --md_name "summary_table.md"
done

echo "All threshold sweeps finished. Check ./run_outputs/sim_threshold_*/summary_table.*"
