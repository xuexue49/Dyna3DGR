# 训练指南

本文档详细介绍如何使用 Dyna3DGR 训练脚本进行模型训练。

## 快速开始

### 基本训练

```bash
python scripts/train.py \
    --config configs/acdc.yaml \
    --data_root data/ACDC \
    --output_dir outputs/experiment_1
```

### 从检查点恢复训练

```bash
python scripts/train.py \
    --config configs/acdc.yaml \
    --data_root data/ACDC \
    --output_dir outputs/experiment_1 \
    --resume outputs/experiment_1/checkpoints/latest.pth
```

### 调试模式

```bash
python scripts/train.py \
    --config configs/acdc.yaml \
    --data_root data/ACDC \
    --output_dir outputs/debug \
    --debug
```

## 训练流程

### 1. 数据准备

确保已经完成数据预处理：

```bash
# 预处理 ACDC 数据集
python scripts/preprocess_data.py \
    --input_dir /path/to/raw/ACDC \
    --output_dir data/ACDC \
    --target_spacing 1.5 1.5 10.0 \
    --target_size 256 256 \
    --normalize \
    --extract_points
```

### 2. 配置文件

训练脚本使用 YAML 配置文件。主要配置项：

#### 模型配置

```yaml
model:
  gaussian:
    num_points: 5000        # Gaussian 数量
    feature_dim: 1          # 特征维度
    init_scale: 0.01        # 初始尺度
    init_opacity: 0.5       # 初始不透明度
  
  deformation:
    spatial_freq: 10        # 空间编码频率
    temporal_freq: 6        # 时间编码频率
    hidden_dim: 256         # 隐藏层维度
    num_layers: 8           # 网络层数
```

#### 训练配置

```yaml
training:
  num_epochs: 50
  batch_size: 2
  learning_rate: 0.0001
  weight_decay: 0.0001
  
  loss_weights:
    reconstruction: 1.0
    temporal_consistency: 0.1
    regularization: 0.01
    cyclic_consistency: 0.05
```

#### 数据配置

```yaml
data:
  data_root: "./data/ACDC"
  image_size: [256, 256]
  num_frames: 10
  normalize: true
  augmentation: true
  num_workers: 4
```

### 3. 启动训练

```bash
python scripts/train.py \
    --config configs/acdc.yaml \
    --data_root data/ACDC \
    --output_dir outputs/my_experiment
```

### 4. 监控训练

#### TensorBoard

训练过程会自动记录到 TensorBoard：

```bash
tensorboard --logdir outputs/my_experiment/logs
```

在浏览器中访问 `http://localhost:6006` 查看：
- 训练和验证损失曲线
- 学习率变化
- Gaussian 数量变化
- 各个损失分量

#### 命令行输出

训练脚本会实时显示：
- 每个 epoch 的进度条
- 当前损失值
- Epoch 总结（时间、损失、Gaussian 数量）

示例输出：

```
Epoch 1/50
------------------------------------------------------------
Epoch 1: 100%|████████| 35/35 [00:45<00:00,  1.30s/it, loss=0.234, recon=0.189]

Epoch 1 Summary:
  Time: 45.23s
  Train Loss: 0.234567
    - Reconstruction: 0.189234
    - Temporal: 0.012345
    - Regularization: 0.023456
    - Cyclic: 0.009532
  Val Loss: 0.245678
  Gaussians: 5234 points
  Learning Rate: 1.000e-04
```

## 训练特性

### 自适应 Gaussian 密度控制

训练过程中会自动调整 Gaussian 数量：

#### Densification（密度化）

在梯度较大的区域增加 Gaussians：

```yaml
densify:
  enabled: true
  start_iter: 500         # 从第 500 次迭代开始
  interval: 100           # 每 100 次迭代执行一次
  grad_threshold: 0.0002  # 梯度阈值
  max_points: 50000       # 最大点数
```

#### Pruning（剪枝）

移除不透明度低的 Gaussians：

```yaml
prune:
  enabled: true
  start_iter: 1000        # 从第 1000 次迭代开始
  interval: 100           # 每 100 次迭代执行一次
  opacity_threshold: 0.01 # 不透明度阈值
```

### 检查点管理

#### 自动保存

训练脚本会自动保存：
- **定期检查点**: 每 N 个 epoch 保存一次
- **最佳模型**: 验证损失最低时保存
- **最新模型**: 每个 epoch 结束时更新

```yaml
checkpoint:
  save_interval: 5        # 每 5 个 epoch 保存
  keep_last: 3            # 保留最近 3 个检查点
```

#### 检查点内容

每个检查点包含：
- Gaussian 模型状态
- Deformation 网络状态
- 优化器状态
- 学习率调度器状态
- 训练进度（epoch, iteration）
- 损失历史
- 配置信息

#### 恢复训练

```bash
python scripts/train.py \
    --config configs/acdc.yaml \
    --data_root data/ACDC \
    --output_dir outputs/experiment_1 \
    --resume outputs/experiment_1/checkpoints/latest.pth
```

### 学习率调度

支持三种学习率调度策略：

#### 1. Exponential Decay

```yaml
lr_scheduler:
  type: "exponential"
  gamma: 0.95             # lr = lr * gamma^epoch
```

#### 2. Step Decay

```yaml
lr_scheduler:
  type: "step"
  step_size: 10           # 每 10 个 epoch
  gamma: 0.5              # lr = lr * 0.5
```

#### 3. Cosine Annealing

```yaml
lr_scheduler:
  type: "cosine"
  # lr 按余弦函数衰减到 0
```

### 梯度裁剪

自动应用梯度裁剪防止梯度爆炸：

```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

## 输出文件

训练完成后，输出目录包含：

```
outputs/my_experiment/
├── checkpoints/
│   ├── checkpoint_epoch_5.pth
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_15.pth
│   ├── best.pth                    # 最佳模型
│   └── latest.pth                  # 最新模型
├── logs/
│   └── events.out.tfevents.*       # TensorBoard 日志
├── visualizations/                  # 可视化结果（未来）
├── config.yaml                      # 保存的配置
└── metrics.json                     # 训练指标
```

### metrics.json

包含完整的训练历史：

```json
{
  "train_losses": [
    {
      "total": 0.234567,
      "recon": 0.189234,
      "temporal": 0.012345,
      "reg": 0.023456,
      "cyclic": 0.009532
    },
    ...
  ],
  "val_losses": [...],
  "best_loss": 0.198765,
  "final_epoch": 50,
  "final_iteration": 1750
}
```

## 高级用法

### 自定义配置

创建自定义配置文件：

```yaml
# configs/my_config.yaml
model:
  gaussian:
    num_points: 10000     # 更多 Gaussians
  deformation:
    hidden_dim: 512       # 更大的网络

training:
  num_epochs: 100
  batch_size: 4
  learning_rate: 0.0002
  
  loss_weights:
    reconstruction: 1.0
    temporal_consistency: 0.2  # 增加时间一致性权重
```

使用自定义配置：

```bash
python scripts/train.py \
    --config configs/my_config.yaml \
    --data_root data/ACDC \
    --output_dir outputs/custom_experiment
```

### 调试模式

调试模式会：
- 只处理前几个 batch
- 显示详细错误信息
- 快速迭代测试

```bash
python scripts/train.py \
    --config configs/acdc.yaml \
    --data_root data/ACDC \
    --output_dir outputs/debug \
    --debug
```

### 指定设备

```bash
# 使用 GPU
python scripts/train.py \
    --config configs/acdc.yaml \
    --data_root data/ACDC \
    --output_dir outputs/experiment_1 \
    --device cuda

# 使用 CPU
python scripts/train.py \
    --config configs/acdc.yaml \
    --data_root data/ACDC \
    --output_dir outputs/experiment_1 \
    --device cpu
```

## 性能优化

### 内存优化

如果遇到内存不足：

1. **减小 batch size**:
```yaml
training:
  batch_size: 1
```

2. **减少 Gaussian 数量**:
```yaml
model:
  gaussian:
    num_points: 3000
```

3. **减少采样帧数**:
```yaml
data:
  num_frames: 5
```

4. **减小图像大小**:
```yaml
data:
  image_size: [128, 128]
```

### 速度优化

1. **增加 num_workers**:
```yaml
data:
  num_workers: 8
```

2. **启用 pin_memory**:
```yaml
data:
  pin_memory: true
```

3. **减少日志频率**:
```yaml
training:
  logging:
    log_interval: 50
```

## 常见问题

### Q1: 训练很慢

**解决方案**:
1. 检查是否使用 GPU
2. 增加 `num_workers`
3. 减小 `num_frames`
4. 使用更小的模型

### Q2: 损失不下降

**解决方案**:
1. 检查学习率（可能太大或太小）
2. 检查数据是否正确加载
3. 调整损失权重
4. 增加训练 epochs

### Q3: 显存溢出

**解决方案**:
1. 减小 `batch_size`
2. 减少 `num_points`
3. 减小 `image_size`
4. 减少 `num_frames`

### Q4: 检查点加载失败

**解决方案**:
1. 确保检查点文件存在
2. 检查模型配置是否匹配
3. 使用 `--device cpu` 在 CPU 上加载

### Q5: 数据加载错误

**解决方案**:
1. 确保数据已预处理
2. 检查 `data_root` 路径
3. 验证数据格式

## 训练技巧

### 1. 渐进式训练

先用小模型快速验证，再增加复杂度：

**阶段 1**: 小模型快速验证
```yaml
model:
  gaussian:
    num_points: 1000
training:
  num_epochs: 10
```

**阶段 2**: 中等模型
```yaml
model:
  gaussian:
    num_points: 5000
training:
  num_epochs: 50
```

**阶段 3**: 完整模型
```yaml
model:
  gaussian:
    num_points: 10000
training:
  num_epochs: 100
```

### 2. 学习率调优

使用学习率查找器找到最佳学习率：

```python
# 尝试不同的学习率
for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
    # 训练几个 epoch 观察损失
```

### 3. 损失权重调优

根据各个损失的量级调整权重：

```yaml
loss_weights:
  reconstruction: 1.0      # 基准
  temporal_consistency: 0.1  # 如果太大会过度平滑
  regularization: 0.01     # 防止过拟合
  cyclic_consistency: 0.05 # 保证周期性
```

### 4. 早停策略

如果验证损失不再下降，可以提前停止训练。

## 实验管理

### 命名规范

使用描述性的实验名称：

```bash
python scripts/train.py \
    --output_dir outputs/acdc_lr1e4_bs4_g10k_$(date +%Y%m%d_%H%M%S)
```

### 参数记录

所有配置会自动保存到 `output_dir/config.yaml`，便于复现实验。

### 结果比较

使用 TensorBoard 比较多个实验：

```bash
tensorboard --logdir outputs/
```

## 下一步

训练完成后：
1. 查看 [评估指南](evaluation.md) 评估模型性能
2. 查看 [可视化指南](visualization.md) 可视化结果
3. 查看 [推理指南](inference.md) 使用训练好的模型

## 获取帮助

如果遇到问题：
1. 查看本文档的常见问题部分
2. 使用 `--debug` 模式获取详细错误信息
3. 查看 [GitHub Issues](https://github.com/xuexue49/Dyna3DGR/issues)
4. 提交新的 Issue 并附上错误日志
