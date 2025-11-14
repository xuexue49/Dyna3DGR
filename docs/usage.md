# 使用指南

本文档介绍如何使用 Dyna3DGR 进行4D心脏运动跟踪。

## 目录

1. [数据准备](#数据准备)
2. [训练模型](#训练模型)
3. [评估模型](#评估模型)
4. [运动跟踪](#运动跟踪)
5. [可视化](#可视化)
6. [配置说明](#配置说明)

## 数据准备

### ACDC 数据集

Dyna3DGR 主要在 ACDC (Automated Cardiac Diagnosis Challenge) 数据集上进行训练和评估。

#### 下载数据集

1. 访问 [ACDC Challenge 官网](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
2. 注册并下载数据集
3. 解压到本地目录

#### 数据集结构

ACDC 数据集应按以下结构组织：

```
data/ACDC/
├── patient001/
│   ├── patient001_frame01.nii.gz
│   ├── patient001_frame01_gt.nii.gz
│   ├── patient001_frame02.nii.gz
│   ├── patient001_frame02_gt.nii.gz
│   ├── ...
│   └── Info.cfg
├── patient002/
│   └── ...
├── ...
└── patient150/
```

其中：
- `*_frameXX.nii.gz`: 心脏MRI图像
- `*_frameXX_gt.nii.gz`: 分割标注（ground truth）
- `Info.cfg`: 患者信息和元数据

#### 数据预处理

如果需要自定义数据预处理，可以使用提供的脚本：

```bash
python scripts/preprocess_data.py \
    --input_dir data/ACDC_raw \
    --output_dir data/ACDC \
    --resize 256 256 \
    --normalize
```

## 训练模型

### 基本训练

使用默认配置训练模型：

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_root data/ACDC \
    --output_dir outputs/experiment_1
```

### 自定义配置

创建自定义配置文件（例如 `configs/my_config.yaml`）：

```yaml
model:
  gaussian:
    num_points: 15000  # 增加 Gaussian 数量
  deformation:
    hidden_dim: 512    # 增加网络容量

training:
  num_epochs: 150
  batch_size: 8
  learning_rate: 0.0002
  
  loss_weights:
    reconstruction: 1.0
    temporal_consistency: 0.2
    regularization: 0.02
```

然后使用自定义配置训练：

```bash
python scripts/train.py \
    --config configs/my_config.yaml \
    --data_root data/ACDC \
    --output_dir outputs/my_experiment
```

### 从检查点恢复训练

如果训练中断，可以从检查点恢复：

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_root data/ACDC \
    --output_dir outputs/experiment_1 \
    --resume outputs/experiment_1/checkpoints/checkpoint_epoch_50.pth
```

### 多GPU训练

使用多个GPU进行分布式训练：

```bash
# 使用 torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/default.yaml \
    --data_root data/ACDC \
    --output_dir outputs/distributed_experiment
```

### 监控训练

#### TensorBoard

训练过程中会自动记录日志到 TensorBoard：

```bash
tensorboard --logdir outputs/experiment_1/logs
```

然后在浏览器中访问 `http://localhost:6006`

#### Weights & Biases (可选)

如果想使用 W&B 进行实验跟踪，在配置文件中启用：

```yaml
training:
  logging:
    wandb: true
    wandb_project: "dyna3dgr"
    wandb_entity: "your_username"
```

## 评估模型

### 在测试集上评估

```bash
python scripts/evaluate.py \
    --model_path outputs/experiment_1/checkpoints/best.pth \
    --data_root data/ACDC \
    --split test \
    --output_dir outputs/experiment_1/evaluation
```

### 评估指标

评估脚本会计算以下指标：

1. **Dice Score**: 分割重叠度
2. **Hausdorff Distance**: 表面距离误差
3. **Tracking Error**: 点跟踪误差
4. **PSNR**: 峰值信噪比
5. **SSIM**: 结构相似性

结果会保存在 `outputs/experiment_1/evaluation/metrics.json`

### 评估单个病例

```bash
python scripts/evaluate.py \
    --model_path outputs/experiment_1/checkpoints/best.pth \
    --input data/ACDC/patient001 \
    --output outputs/evaluation/patient001
```

## 运动跟踪

### 跟踪心脏运动

```bash
python scripts/track_motion.py \
    --model_path outputs/experiment_1/checkpoints/best.pth \
    --input data/ACDC/patient001 \
    --output outputs/tracking/patient001 \
    --save_trajectories \
    --save_deformation_field
```

### 输出说明

运动跟踪会生成以下输出：

- `trajectories.npy`: 点轨迹 [N, T, 3]
- `deformation_field.npy`: 变形场 [T, H, W, D, 3]
- `motion_magnitude.npy`: 运动幅度 [T, H, W, D]
- `visualizations/`: 可视化图像

### 提取特定区域的运动

```bash
python scripts/track_motion.py \
    --model_path outputs/experiment_1/checkpoints/best.pth \
    --input data/ACDC/patient001 \
    --output outputs/tracking/patient001 \
    --region left_ventricle \
    --save_strain_analysis
```

## 可视化

### 渲染重建结果

```bash
python scripts/render.py \
    --model_path outputs/experiment_1/checkpoints/best.pth \
    --input data/ACDC/patient001 \
    --output outputs/renders/patient001 \
    --num_frames 30 \
    --interpolate
```

### 生成运动可视化

```bash
python scripts/visualize.py \
    --model_path outputs/experiment_1/checkpoints/best.pth \
    --input data/ACDC/patient001 \
    --output outputs/visualizations/patient001 \
    --mode motion \
    --colormap jet
```

### 交互式可视化

使用 Jupyter Notebook 进行交互式可视化：

```bash
jupyter notebook notebooks/demo.ipynb
```

### 3D可视化

生成3D可视化（需要安装 Open3D）：

```bash
python scripts/visualize_3d.py \
    --model_path outputs/experiment_1/checkpoints/best.pth \
    --input data/ACDC/patient001 \
    --output outputs/3d_vis/patient001
```

## 配置说明

### 模型配置

```yaml
model:
  gaussian:
    num_points: 10000        # Gaussian 数量
    feature_dim: 1           # 特征维度
    init_scale: 0.01         # 初始尺度
    init_opacity: 0.5        # 初始不透明度
  
  deformation:
    spatial_freq: 10         # 空间位置编码频率
    temporal_freq: 6         # 时间位置编码频率
    hidden_dim: 256          # 隐藏层维度
    num_layers: 8            # MLP层数
```

### 训练配置

```yaml
training:
  num_epochs: 100            # 训练轮数
  batch_size: 4              # 批大小
  learning_rate: 0.0001      # 学习率
  weight_decay: 0.0001       # 权重衰减
  
  lr_scheduler:
    type: "exponential"      # 学习率调度器类型
    gamma: 0.95              # 衰减因子
  
  loss_weights:
    reconstruction: 1.0      # 重建损失权重
    temporal_consistency: 0.1  # 时间一致性损失权重
    regularization: 0.01     # 正则化损失权重
    cyclic_consistency: 0.05 # 循环一致性损失权重
```

### 数据配置

```yaml
data:
  dataset: "acdc"            # 数据集名称
  data_root: "./data/ACDC"   # 数据根目录
  image_size: [256, 256]     # 图像大小
  num_frames: 20             # 每个序列的帧数
  normalize: true            # 是否归一化
  augmentation: true         # 是否数据增强
```

## 常见使用场景

### 场景1: 快速原型验证

使用小规模数据快速验证想法：

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_root data/ACDC \
    --output_dir outputs/quick_test \
    --num_epochs 10 \
    --batch_size 2
```

### 场景2: 高质量训练

使用最佳配置进行完整训练：

```bash
python scripts/train.py \
    --config configs/high_quality.yaml \
    --data_root data/ACDC \
    --output_dir outputs/high_quality \
    --num_epochs 200
```

### 场景3: 超参数搜索

使用不同配置进行实验：

```bash
for lr in 0.0001 0.0002 0.0005; do
    for hidden_dim in 128 256 512; do
        python scripts/train.py \
            --config configs/default.yaml \
            --data_root data/ACDC \
            --output_dir outputs/exp_lr${lr}_hd${hidden_dim} \
            --learning_rate $lr \
            --hidden_dim $hidden_dim
    done
done
```

## 故障排除

### 训练不收敛

1. 降低学习率
2. 增加 batch size
3. 调整损失权重
4. 检查数据预处理

### 显存不足

1. 减小 batch size
2. 减少 Gaussian 数量
3. 启用混合精度训练
4. 使用梯度累积

### 运动跟踪不准确

1. 增加训练轮数
2. 增加时间一致性损失权重
3. 调整 Gaussian 密度
4. 检查数据质量

## 下一步

- 查看 [API 文档](api.md) 了解详细的 API 参考
- 查看 [架构说明](architecture.md) 了解系统设计
- 查看示例 notebooks 学习高级用法

## 获取帮助

如果遇到问题：
1. 查看 [FAQ](../README.md#常见问题)
2. 搜索 [GitHub Issues](https://github.com/xuexue49/Dyna3DGR/issues)
3. 提交新的 Issue
