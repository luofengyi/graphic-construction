# MERC-main 服务器补文件与训练指南

这份文档用于在另一台服务器 `git clone` 本仓库后，快速补齐运行所需文件，并直接开始训练。

## 1. 拉取代码

```bash
git clone https://github.com/luofengyi/graphic-construction.git
cd graphic-construction/JOYFUL
```

## 2. 需要手动补充的文件

由于仓库已排除大文件（数据集和 PT），你需要把以下文件手动上传到服务器对应路径：

### 必需数据文件（训练必须）

- `JOYFUL/data/iemocap/data_iemocap.pkl`
- `JOYFUL/data/iemocap_4/data_iemocap_4.pkl`

目录结构必须是：

```text
JOYFUL/
  data/
    iemocap/
      data_iemocap.pkl
    iemocap_4/
      data_iemocap_4.pkl
```

### 可选模型文件（仅评估或断点续训需要）

- `JOYFUL/model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt`

> 从头训练（`--from_begin`）不依赖该 PT 文件。

## 3. 创建与安装环境（推荐）

```bash
conda create -n joyful python=3.9 -y
conda activate joyful

# CPU 版本（与当前仓库适配）
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4 scikit-learn tqdm sentence-transformers
pip install torch-geometric pygcl dgl==2.2.1 torchdata==0.7.1
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

## 4. 可直接训练的脚本

下面脚本可直接保存为 `run_train.sh` 后执行（或逐行执行）。

```bash
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/JOYFUL"

# 四分类 + 你当前实现的单路融合构图（binary + hyperedge expansion）
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
```

执行：

```bash
bash run_train.sh
```

## 5. 最小验证命令（先跑通）

建议先用 1 epoch 烟测：

```bash
python train.py --dataset iemocap_4 --modalities atv --graph_mode hybrid_expand --from_begin --epochs 1 --device cpu
```

看到 `Epoch`、`Dev set`、`Test set` 指标输出即表示训练流程正常。
