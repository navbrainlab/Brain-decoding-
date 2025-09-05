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