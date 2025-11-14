# 可视化和评估工具

本文档介绍 Dyna3DGR 的可视化和评估工具，用于分析渲染结果和评估模型性能。

## 目录

- [概述](#概述)
- [交互式可视化](#交互式可视化)
- [批量评估](#批量评估)
- [评估指标](#评估指标)
- [可视化函数](#可视化函数)
- [使用示例](#使用示例)

## 概述

Dyna3DGR 提供了完整的可视化和评估工具集：

- **交互式查看器**: 实时浏览渲染结果
- **批量评估**: 自动评估多个患者
- **丰富的指标**: PSNR, SSIM, MAE, MSE 等
- **多种可视化**: 切片对比、序列动画、网格视图
- **导出功能**: 保存图像、视频、报告

## 交互式可视化

### visualize_results.py

交互式查看器允许您实时浏览和比较渲染结果。

#### 基本用法

```bash
python scripts/visualize_results.py \
    --checkpoint outputs/patient001/best.pth \
    --patient_dir data/ACDC/patient001 \
    --mode interactive
```

#### 功能特性

**交互控件**:
- **时间滑块**: 选择时间帧
- **切片滑块**: 选择切片
- **Previous/Next 按钮**: 逐帧浏览
- **Play 按钮**: 自动播放序列
- **Save 按钮**: 保存当前视图

**显示内容**:
- 左侧: Ground Truth
- 中间: 渲染结果
- 右侧: 差异图
- 底部: 中心行强度曲线
- 统计信息: MAE, MSE, PSNR

#### 所有模式

```bash
# 交互式查看器
python scripts/visualize_results.py \
    --checkpoint outputs/patient001/best.pth \
    --patient_dir data/ACDC/patient001 \
    --mode interactive

# 批量生成图像
python scripts/visualize_results.py \
    --checkpoint outputs/patient001/best.pth \
    --patient_dir data/ACDC/patient001 \
    --mode batch \
    --output_dir visualizations

# 生成视频
python scripts/visualize_results.py \
    --checkpoint outputs/patient001/best.pth \
    --patient_dir data/ACDC/patient001 \
    --mode video \
    --slice_idx 5 \
    --output_dir visualizations

# 生成对比网格
python scripts/visualize_results.py \
    --checkpoint outputs/patient001/best.pth \
    --patient_dir data/ACDC/patient001 \
    --mode grid \
    --output_dir visualizations
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型检查点路径 | 必需 |
| `--patient_dir` | 患者数据目录 | 必需 |
| `--output_dir` | 输出目录 | `visualizations` |
| `--mode` | 可视化模式 | `interactive` |
| `--device` | 设备 (cuda/cpu) | `cuda` |
| `--image_size` | 图像大小 | `256 256` |
| `--num_slices` | 切片数量 | `10` |
| `--num_frames` | 时间帧数 | `20` |
| `--slice_idx` | 特定切片索引 | `None` |

## 批量评估

### evaluate.py

批量评估脚本自动评估多个患者并生成详细报告。

#### 基本用法

```bash
python scripts/evaluate.py \
    --data_root data/ACDC \
    --checkpoint_dir outputs \
    --split test \
    --output_dir evaluation_results
```

#### 输出结构

```
evaluation_results/
├── all_metrics.csv              # 所有患者的指标
├── statistics.csv               # 统计摘要
├── summary.json                 # 评估总结
├── patient001/
│   ├── metrics.json             # 患者指标
│   ├── sequence_metrics.csv     # 序列指标
│   ├── comparison_t10_s5.png    # 切片对比
│   ├── all_slices_t10.png       # 所有切片
│   ├── comparison_grid.png      # 对比网格
│   └── metrics_over_time.png    # 时间序列指标
├── patient002/
│   └── ...
└── ...
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_root` | 数据根目录 | 必需 |
| `--checkpoint_dir` | 检查点目录 | 必需 |
| `--output_dir` | 输出目录 | `evaluation_results` |
| `--split` | 数据集划分 | `test` |
| `--device` | 设备 | `cuda` |
| `--image_size` | 图像大小 | `256 256` |
| `--num_slices` | 切片数量 | `10` |
| `--no_visualizations` | 跳过可视化 | `False` |
| `--patient_ids` | 特定患者ID | `None` |

#### 评估特定患者

```bash
python scripts/evaluate.py \
    --data_root data/ACDC \
    --checkpoint_dir outputs \
    --patient_ids patient001 patient002 patient003 \
    --output_dir evaluation_results
```

## 评估指标

### 图像质量指标

#### MAE (Mean Absolute Error)
```python
MAE = mean(|pred - target|)
```
- 范围: [0, ∞)
- 越小越好
- 衡量平均像素差异

#### MSE (Mean Squared Error)
```python
MSE = mean((pred - target)²)
```
- 范围: [0, ∞)
- 越小越好
- 对大误差更敏感

#### RMSE (Root Mean Squared Error)
```python
RMSE = sqrt(MSE)
```
- 范围: [0, ∞)
- 越小越好
- 与原始数据同单位

#### PSNR (Peak Signal-to-Noise Ratio)
```python
PSNR = 10 * log10(MAX² / MSE)
```
- 范围: [0, ∞) dB
- 越大越好
- 常用于图像质量评估
- 典型值: 30-50 dB

#### SSIM (Structural Similarity Index)
```python
SSIM = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
```
- 范围: [-1, 1]
- 越接近 1 越好
- 考虑亮度、对比度、结构
- 更符合人眼感知

#### NCC (Normalized Cross-Correlation)
```python
NCC = Σ((x - μ_x)(y - μ_y)) / sqrt(Σ(x - μ_x)² Σ(y - μ_y)²)
```
- 范围: [-1, 1]
- 越接近 1 越好
- 衡量相关性

### 分割指标

#### Dice Coefficient
```python
Dice = 2|X ∩ Y| / (|X| + |Y|)
```
- 范围: [0, 1]
- 越大越好
- 常用于分割评估

#### IoU (Intersection over Union)
```python
IoU = |X ∩ Y| / |X ∪ Y|
```
- 范围: [0, 1]
- 越大越好
- 也称 Jaccard Index

#### Hausdorff Distance
```python
HD = max(h(X, Y), h(Y, X))
```
- 范围: [0, ∞)
- 越小越好
- 衡量边界距离

### 时间一致性

#### Temporal Consistency
```python
TC = mean(|frame[t+1] - frame[t]|)
```
- 范围: [0, ∞)
- 越小越好
- 衡量序列平滑度

## 可视化函数

### 核心函数

#### compare_slices()

比较单个切片的渲染结果和真值。

```python
from dyna3dgr.utils.visualization import compare_slices

fig = compare_slices(
    rendered=rendered_slices,      # [T, num_slices, H, W]
    ground_truth=gt_slices,        # [T, num_slices, H, W]
    slice_idx=5,                   # 切片索引
    time_idx=10,                   # 时间索引
    save_path='comparison.png',    # 保存路径
    show_difference=True,          # 显示差异图
    colormap='gray',               # 颜色映射
)
```

#### visualize_sequence()

创建时间序列动画。

```python
from dyna3dgr.utils.visualization import visualize_sequence

anim = visualize_sequence(
    rendered=rendered_slices,      # [T, num_slices, H, W]
    ground_truth=gt_slices,        # [T, num_slices, H, W]
    slice_idx=5,                   # 切片索引
    save_path='sequence.gif',      # 保存为 GIF
    interval=200,                  # 帧间隔 (ms)
)
```

#### visualize_all_slices()

显示所有切片。

```python
from dyna3dgr.utils.visualization import visualize_all_slices

fig = visualize_all_slices(
    rendered=rendered_slices,      # [T, num_slices, H, W]
    ground_truth=gt_slices,        # [T, num_slices, H, W]
    time_idx=10,                   # 时间索引
    save_path='all_slices.png',    # 保存路径
    max_cols=5,                    # 最大列数
)
```

#### create_comparison_grid()

创建多时间点/切片的对比网格。

```python
from dyna3dgr.utils.visualization import create_comparison_grid

fig = create_comparison_grid(
    rendered=rendered_slices,      # [T, num_slices, H, W]
    ground_truth=gt_slices,        # [T, num_slices, H, W]
    num_samples=9,                 # 样本数量
    save_path='grid.png',          # 保存路径
)
```

#### plot_metrics_over_time()

绘制指标随时间变化。

```python
from dyna3dgr.utils.visualization import plot_metrics_over_time

fig = plot_metrics_over_time(
    metrics={
        'PSNR': psnr_values,       # [T]
        'SSIM': ssim_values,       # [T]
        'MAE': mae_values,         # [T]
    },
    save_path='metrics.png',       # 保存路径
)
```

#### create_video_comparison()

创建视频对比。

```python
from dyna3dgr.utils.visualization import create_video_comparison

create_video_comparison(
    rendered=rendered_slices,      # [T, num_slices, H, W]
    ground_truth=gt_slices,        # [T, num_slices, H, W]
    slice_idx=5,                   # 切片索引
    output_path='comparison.mp4',  # 输出路径
    fps=10,                        # 帧率
    codec='mp4v',                  # 编解码器
)
```

### 评估函数

#### compute_all_metrics()

计算所有可用指标。

```python
from dyna3dgr.utils.metrics import compute_all_metrics

metrics = compute_all_metrics(
    pred=rendered,                 # 预测结果
    target=ground_truth,           # 真值
    data_range=1.0,                # 数据范围
    include_segmentation=False,    # 是否包含分割指标
)

# 返回字典:
# {
#     'MAE': 0.0234,
#     'MSE': 0.0012,
#     'RMSE': 0.0346,
#     'PSNR': 38.45,
#     'SSIM': 0.9234,
#     'NCC': 0.9567,
# }
```

#### compute_sequence_metrics()

计算序列中每帧的指标。

```python
from dyna3dgr.utils.metrics import compute_sequence_metrics

metrics = compute_sequence_metrics(
    pred_sequence=rendered,        # [T, H, W] or [T, num_slices, H, W]
    target_sequence=ground_truth,  # [T, H, W] or [T, num_slices, H, W]
    data_range=1.0,                # 数据范围
)

# 返回字典:
# {
#     'MAE': array([...]),         # [T]
#     'PSNR': array([...]),        # [T]
#     'SSIM': array([...]),        # [T]
#     ...
# }
```

#### MetricsTracker

跟踪和累积指标。

```python
from dyna3dgr.utils.metrics import MetricsTracker

tracker = MetricsTracker()

# 更新指标
for batch in dataloader:
    metrics = compute_metrics(batch)
    tracker.update(metrics)

# 获取平均值
avg_metrics = tracker.get_average()
print(tracker)  # 打印摘要

# 重置
tracker.reset()
```

## 使用示例

### 示例 1: 快速可视化

```python
import torch
import numpy as np
from dyna3dgr.models import Gaussian3D, DeformationNetwork
from dyna3dgr.rendering import Medical2DSliceRenderer
from dyna3dgr.utils.visualization import compare_slices

# 加载模型
checkpoint = torch.load('outputs/patient001/best.pth')
gaussians = Gaussian3D(num_points=5000)
gaussians.load_state_dict(checkpoint['gaussians_state'])

# 渲染
renderer = Medical2DSliceRenderer(image_size=(256, 256), num_slices=10)
rendered = renderer(
    means=gaussians.xyz,
    scales=gaussians.scale,
    rotations=gaussians.rotation,
    opacities=gaussians.opacity,
    features=gaussians.features,
)

# 可视化
fig = compare_slices(
    rendered=rendered.cpu().numpy(),
    ground_truth=gt_data,
    slice_idx=5,
    time_idx=0,
)
plt.show()
```

### 示例 2: 批量评估

```bash
# 评估测试集所有患者
python scripts/evaluate.py \
    --data_root data/ACDC \
    --checkpoint_dir outputs \
    --split test \
    --output_dir evaluation_results

# 查看结果
cat evaluation_results/summary.json
```

### 示例 3: 创建报告

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('evaluation_results/all_metrics.csv')

# 绘制指标分布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ['PSNR', 'SSIM', 'MAE', 'MSE', 'RMSE', 'NCC']

for ax, metric in zip(axes.flat, metrics):
    df[metric].hist(ax=ax, bins=20)
    ax.set_title(f'{metric} Distribution')
    ax.set_xlabel(metric)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('metrics_distribution.png')
```

### 示例 4: 交互式探索

```python
# 启动交互式查看器
from scripts.visualize_results import InteractiveViewer

viewer = InteractiveViewer(
    rendered=rendered_data,
    ground_truth=gt_data,
    patient_id='patient001',
)
viewer.show()
```

### 示例 5: 生成视频

```bash
# 为特定切片生成对比视频
python scripts/visualize_results.py \
    --checkpoint outputs/patient001/best.pth \
    --patient_dir data/ACDC/patient001 \
    --mode video \
    --slice_idx 5 \
    --output_dir videos
```

## 最佳实践

### 1. 评估流程

```bash
# 1. 训练完成后，首先进行批量评估
python scripts/evaluate.py \
    --data_root data/ACDC \
    --checkpoint_dir outputs \
    --split test

# 2. 查看总体统计
cat evaluation_results/summary.json

# 3. 查看详细结果
cat evaluation_results/all_metrics.csv

# 4. 对特定患者进行交互式探索
python scripts/visualize_results.py \
    --checkpoint outputs/patient001/best.pth \
    --patient_dir data/ACDC/patient001 \
    --mode interactive
```

### 2. 指标选择

- **图像重建**: 使用 PSNR 和 SSIM
- **分割评估**: 使用 Dice 和 Hausdorff
- **时间一致性**: 使用 Temporal Consistency
- **综合评估**: 结合多个指标

### 3. 可视化技巧

- 使用交互式查看器快速浏览
- 生成网格图对比多个时间点
- 创建视频展示动态变化
- 绘制指标曲线分析趋势

### 4. 性能优化

```python
# 使用 no_visualizations 加速评估
python scripts/evaluate.py \
    --data_root data/ACDC \
    --checkpoint_dir outputs \
    --no_visualizations  # 跳过可视化

# 评估特定患者
python scripts/evaluate.py \
    --data_root data/ACDC \
    --checkpoint_dir outputs \
    --patient_ids patient001 patient002
```

## 故障排除

### 问题 1: 内存不足

**解决方案**:
```bash
# 减小图像大小
python scripts/visualize_results.py \
    --image_size 128 128 \
    ...

# 减少切片数量
python scripts/visualize_results.py \
    --num_slices 5 \
    ...
```

### 问题 2: 渲染速度慢

**解决方案**:
```python
# 使用 CPU
python scripts/evaluate.py \
    --device cpu \
    ...

# 跳过可视化
python scripts/evaluate.py \
    --no_visualizations \
    ...
```

### 问题 3: 形状不匹配

**解决方案**:
- 确保渲染参数与数据一致
- 检查 `num_slices` 和 `num_frames` 设置
- 查看警告信息了解实际形状

## 参考资料

- [训练指南](training.md)
- [渲染系统](rendering.md)
- [数据加载](data_loading.md)

## 下一步

- 尝试交互式查看器
- 运行批量评估
- 分析评估结果
- 创建自定义可视化
