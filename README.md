# ResNet1D_most_mini 模型训练与推理说明

本项目使用 `ResNet1D_most_mini` 架构对 ECoG数据进行动作分类。模型具备多尺度卷积结构、残差连接、支持通道选择与滑窗增强，采用 PyTorch Lightning 进行训练与推理管理。

---

## 模型定义和训练（`motion_decoding.py`）

#### 输入参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `input_channels` | int | 输入的通道数（通常为选择后的通道数量） |
| `num_classes` | int | 分类类别数（如 4） |
| `num_blocks` | list[int] | 残差模块数量（如 `[1]` 表示一层） |
| `dropout_rate` | float | dropout 比例（如 0.6） |
| `contrastive_temperature` | float | 暂未启用（用于对比学习） |
| `max_epochs` | int | 最大训练轮数，用于调度器 |
| `lr` | float | 学习率（如 `3e-4`） |

#### 模型结构

- **conv1/conv1_2**：输入特征分别经过 kernel=32 与 kernel=64 的一维卷积
- **拼接后 BN + 激活**，再送入残差结构 `layer1`
- **残差块 `BasicBlock`**：使用多尺度卷积（3, 15, 31）作为第一层，后接标准卷积 + BN + Dropout
- **分类器 `fnn`**：`[512 → 256 → num_classes]` 的全连接网络

### ✅ 类：`DataModule(pl.LightningDataModule)`

用于封装训练、验证、测试集，简化 PyTorch Lightning 的数据接口管理。

#### 参数

| 参数名 | 说明 |
|--------|------|
| `train_dataset` | 训练集 TensorDataset |
| `val_dataset` | 验证集 TensorDataset |
| `test_dataset` | 测试集 TensorDataset |
| `batch_size` | 批大小（如 256） |
| `num_workers` | 加载线程数（默认 4） |

---

### 关键设置参数

```python
batch_size = 256
dropout = 0.6
lr = 3e-4
weight_decay = 1e-4
channel_num = 80
feature_dim = 126 # 通道选择后会被更改
num_classes = 4
num_epochs = 200
window_size = 600
step = 60
```

### 数据处理流程

1. 读取多个 session 的 ECoG 原始 trial 数据（`read_data_motion_session`）
2. 使用 `slide_window` 按窗口滑动提取片段
3. 利用 `channel_contribution` 函数选出最重要的 `channel_num` 个通道（基于 Riemann）
4. 构建 `TensorDataset`：train/val/test 分别构建
5. 封装成 `DataModule`

### 训练器配置

```python
trainer = pl.Trainer(
    max_epochs=num_epochs,
    devices=1,
    default_root_dir=output_path,
    logger=logger,
)
```
---

## 推理脚本（`run_infer.py`）

用于对训练好的模型进行测试集推理并输出评估结果与预测记录。

### 关键参数

与训练参数保持一致：

```python
window_size = 600
step = 60
chs_select = [...]  # 与训练时 channel_contribution 的结果保持一致
feature_dim = len(chs_select)
num_classes = 4
dropout = 0.6
lr = 3e-4
num_epochs = 200
```

### 推理流程

1. 加载 `.ckpt` 文件并构造模型结构（必须结构一致）
2. 加载测试集数据，并使用 `slide_window` 和通道筛选预处理
3. 构造 `DataLoader`
4. 调用 `run_inference()` 执行推理与结果保存

### 推理输出内容

- 控制台输出：Test Loss / Accuracy
- `confusion_matrix.png`：混淆矩阵图
- `test_results.json`：逐样本预测详情

```json
{
  "accuracy": 0.912,
  "samples": [
    {"index": 0, "true_label": 1, "pred_label": 1},
    {"index": 1, "true_label": 3, "pred_label": 2},
    ...
  ]
}
```

### 训练

```bash
python motion_decoding.py
```

### 推理

```bash
python run_infer.py
```

---

## RieHy：Riemannian Hypergraph / OTTA 模块

`RieHy/` 目录包含基于黎曼几何与超图的 ECoG 动作解码方法，覆盖 **源域训练**、**在线自适应（OTTA）**、**多特征超图** 和 **MDM 基线** 等流程。

### 目录脚本速览

- `train.py`：深度模型源域离线训练（输出模型权重与训练记录）。
- `adapt.py`：深度模型在线自适应（读取训练结果并对目标日期进行流式更新）。
- `multi_feature_hypergraph_train.py` / `multi_feature_hypergraph_adapt.py`：多特征超图的训练与在线适应。
- `Riemannian_MDM_train.py` / `Riemannian_MDM_adapt.py`：Riemannian MDM 基线训练与适应。
- `resemble_adapt.py`：集成深度模型与 MDM 的在线适应。
- `run_otta.py`：流水线脚本，顺序执行以上代码并记录结果。

### 数据路径与参数说明

- `data_path` 支持：
  - `daily_bdy` 目录（按日期子目录读取）
  - BCI 竞赛数据：`bcic_1` 形式（末尾为 subject id）
  - `.h5` 数据文件
- `source_dates` / `target_dates` 使用空格分隔（`nargs='+'`），多天训练时可用 `20250325_20250326_20250327` 形式。
- 所有脚本支持 `--config` 传入 JSON，加载默认参数后再用 CLI 覆盖。

### 1) 深度模型源域训练（train.py）

```bash
python RieHy/train.py \
  --source_dates 20250325_20250326_20250327 \
  --data_path /path/to/daily_bdy \
  --output_path /path/to/OTTA_results \
  --alignment Riemannian \
  --model_name MultiScale1DCNN_v2
```

输出（写入 `output_path/<timestamp>/`）：
- `best_model_<date>.pt`
- `training_history_<date>.json`
- `args.json`
- `adapt_config.json`（可直接供 `adapt.py` 使用）

### 2) 深度模型在线适应（adapt.py）

```bash
python RieHy/adapt.py \
  --checkpoint_root /path/to/OTTA_results/20250305_123456 \
  --source_dates 20250325_20250326_20250327 \
  --target_dates 20250329 20250331 20250401 \
  --data_path /path/to/daily_bdy \
  --indices_root /path/to/OTTA_results/20250305_123456 \
  --alignment Riemannian \
  --buffer_size 32
```

也可使用训练阶段生成的 `adapt_config.json`：

```bash
python RieHy/adapt.py --config /path/to/OTTA_results/20250305_123456/adapt_config.json
```

输出：
- `adaptation_results_<date>.npz`
- `adaptation_accuracy_<date>.png`
- `adapt_<timestamp>.log`

### 3) OTTA 一键流水线（run_otta.py）

```bash
python RieHy/run_otta.py \
  --output-path /path/to/OTTA_results \
  --data-path /path/to/daily_bdy \
  --source-dates 20250325_20250326_20250327 \
  --target-dates 20250329,20250331,20250401
```

该脚本会：
1. 运行 `train.py`
2. 更新最新实验目录下的 `adapt_config.json`
3. 运行 `multi_feature_hypergraph_train.py`
4. 运行 `Riemannian_MDM_train.py`
5. 生成 `resemble_adapt_config.json`

如需自定义训练/适应参数，可用 `--train-extra`、`--adapt-extra` 或 `--multi-feature-extra` 传递给各脚本。

### 4) 多特征超图与 MDM 基线

```bash
python RieHy/multi_feature_hypergraph_train.py \
  --checkpoint_root /path/to/OTTA_results/20250305_123456 \
  --source_dates 20250325_20250326_20250327 \
  --data_path /path/to/daily_bdy \
  --output_path /path/to/Multi-feature_results

python RieHy/multi_feature_hypergraph_adapt.py \
  --checkpoint_root /path/to/OTTA_results/20250305_123456 \
  --source_hypergraph_root /path/to/Multi-feature_results/20250305_123456 \
  --source_dates 20250325_20250326_20250327 \
  --target_dates 20250329 20250331 20250401 \
  --data_path /path/to/daily_bdy

python RieHy/Riemannian_MDM_train.py \
  --source_dates 20250325_20250326_20250327 \
  --data_path /path/to/daily_bdy \
  --output_path /path/to/MDM_results

python RieHy/Riemannian_MDM_adapt.py \
  --class_means_root /path/to/MDM_results/20250305_123456 \
  --source_dates 20250325_20250326_20250327 \
  --target_dates 20250329 20250331 20250401 \
  --data_path /path/to/daily_bdy
```

### 5) 模型集成

```bash
python RieHy/resemble_adapt.py \
  --checkpoint_root /path/to/OTTA_results/20250305_123456 \
  --class_means_root /path/to/MDM_results/20250305_123456 \
  --source_dates 20250325_20250326_20250327 \
  --target_dates 20250329 20250331 20250401 \
  --data_path /path/to/daily_bdy
```
