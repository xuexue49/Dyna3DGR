# Gaussian Splatting 渲染系统

本文档介绍 Dyna3DGR 的 Gaussian Splatting 渲染系统，包括医学图像专用的 2D 切片渲染器。

## 目录

- [概述](#概述)
- [医学图像渲染](#医学图像渲染)
- [渲染器类型](#渲染器类型)
- [使用示例](#使用示例)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

## 概述

Dyna3DGR 实现了可微分的 Gaussian Splatting 渲染，专门针对医学图像进行了优化。

### 关键特性

- ✅ **2D 切片渲染**: 专为医学图像设计
- ✅ **正交投影**: 无透视畸变
- ✅ **可微分**: 支持端到端训练
- ✅ **高效**: 分块处理和距离裁剪
- ✅ **时间序列**: 支持动态场景

### 与标准渲染的区别

| 特性 | 标准 3D 渲染 | 医学 2D 切片渲染 |
|------|-------------|-----------------|
| 输出 | 3D 体积 | 2D 切片序列 |
| 投影 | 透视/正交 | 正交 |
| 抗锯齿 | 是 | 否 |
| 预过滤 | 是 | 否 |
| 用途 | 通用场景 | 医学图像 |

## 医学图像渲染

### 为什么使用 2D 切片？

医学图像（如心脏 MRI）通常以 **2D 切片序列** 的形式采集和存储：

```
患者数据:
├── slice_00.png  (t=0)
├── slice_01.png  (t=1)
├── ...
└── slice_20.png  (t=20)
```

使用 2D 切片渲染的优势：

1. **符合数据格式**: 直接匹配采集方式
2. **计算效率**: 比完整 3D 体积渲染更快
3. **更好的插值**: 适合帧间插值
4. **成功实践**: MedGS 等工作的验证

### 3D 到 2D 投影原理

给定 3D Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 和固定的切片位置 $z = z_{\text{slice}}$，我们计算 2D 边缘分布：

**条件均值**:
$$
\boldsymbol{\mu}_{2D} = \boldsymbol{\mu}_{xy} + \boldsymbol{\Sigma}_{xy,z} \boldsymbol{\Sigma}_{zz}^{-1} (z_{\text{slice}} - \mu_z)
$$

**条件协方差**:
$$
\boldsymbol{\Sigma}_{2D} = \boldsymbol{\Sigma}_{xy} - \boldsymbol{\Sigma}_{xy,z} \boldsymbol{\Sigma}_{zz}^{-1} \boldsymbol{\Sigma}_{z,xy}
$$

这保证了：
- 数学上严格
- 保持 Gaussian 性质
- 完全可微分

## 渲染器类型

### 1. Medical2DSliceRenderer

**专为医学图像设计的 2D 切片渲染器。**

```python
from dyna3dgr.rendering import Medical2DSliceRenderer

renderer = Medical2DSliceRenderer(
    image_size=(256, 256),      # 每个切片的大小
    num_slices=10,              # 切片数量
    slice_spacing=1.0,          # 切片间距 (mm)
    background_value=0.0,       # 背景值
    chunk_size=500,             # 分块大小
)
```

**参数说明**:
- `image_size`: 2D 切片的分辨率 (H, W)
- `num_slices`: 序列中的切片数量
- `slice_spacing`: 切片之间的物理间距
- `background_value`: 背景强度值（通常为 0）
- `chunk_size`: 每次处理的 Gaussian 数量

**使用示例**:

```python
# 渲染单个时间点
rendered_slices = renderer(
    means=gaussian_xyz,        # [N, 3]
    scales=gaussian_scales,    # [N, 3]
    rotations=gaussian_rots,   # [N, 4] (quaternions)
    opacities=gaussian_alpha,  # [N, 1]
    features=gaussian_features,# [N, 1] (intensity)
)
# 输出: [num_slices, H, W]
```

**时间序列渲染**:

```python
# 渲染整个时间序列
rendered_sequence = renderer.render_with_time(
    means=gaussian_xyz,
    scales=gaussian_scales,
    rotations=gaussian_rots,
    opacities=gaussian_alpha,
    features=gaussian_features,
    timestamps=torch.linspace(0, 1, 20),  # [T]
    deformation_net=deformation_network,
)
# 输出: [T, num_slices, H, W]
```

### 2. EfficientGaussianRenderer

**通用的 3D 体积渲染器。**

```python
from dyna3dgr.rendering import EfficientGaussianRenderer

renderer = EfficientGaussianRenderer(
    image_size=(256, 256, 10),  # (H, W, D)
    background_color=0.0,
    chunk_size=1000,
    distance_threshold=3.0,      # 3 sigma
)
```

适用于：
- 完整 3D 体积渲染
- 非医学图像场景
- 需要完整 3D 输出的情况

### 3. Camera System

**相机系统用于视角控制。**

```python
from dyna3dgr.rendering import Camera, VolumetricCamera

# 基础相机
camera = Camera(
    image_size=(256, 256, 10),
    projection_type='orthographic',
)

# 医学图像专用相机
vol_camera = VolumetricCamera(
    volume_size=(256, 256, 10),
    voxel_spacing=(1.0, 1.0, 5.0),  # mm
)
```

## 使用示例

### 完整训练示例

```python
import torch
from dyna3dgr.models import Gaussian3D, DeformationNetwork
from dyna3dgr.rendering import Medical2DSliceRenderer
from dyna3dgr.data import create_patient_dataloader

# 1. 初始化模型
gaussians = Gaussian3D(num_points=5000)
deformation_net = DeformationNetwork()

# 2. 初始化渲染器
renderer = Medical2DSliceRenderer(
    image_size=(256, 256),
    num_slices=10,
)

# 3. 加载患者数据
patient_loader = create_patient_dataloader(
    patient_dir='data/patient001',
    image_size=(256, 256),
)

# 4. 训练循环
optimizer = torch.optim.Adam([
    {'params': gaussians.parameters()},
    {'params': deformation_net.parameters()},
])

for epoch in range(num_epochs):
    for batch in patient_loader:
        images = batch['images']      # [1, T, H, W]
        timestamps = batch['timestamps']  # [1, T]
        
        # 渲染
        rendered = []
        for t in range(images.shape[1]):
            # 应用变形
            deformed_xyz, deformed_alpha = deformation_net.apply_deformation(
                gaussians.xyz,
                gaussians.opacity,
                timestamps[0, t],
            )
            
            # 渲染切片
            slices = renderer(
                means=deformed_xyz,
                scales=gaussians.scale,
                rotations=gaussians.rotation,
                opacities=deformed_alpha,
                features=gaussians.features,
            )
            rendered.append(slices)
        
        rendered = torch.stack(rendered, dim=0)  # [T, num_slices, H, W]
        
        # 计算损失
        loss = F.mse_loss(rendered, images[0])
        
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 渲染可视化

```python
import matplotlib.pyplot as plt

# 渲染单个时间点
rendered_slices = renderer(
    means=gaussians.xyz,
    scales=gaussians.scale,
    rotations=gaussians.rotation,
    opacities=gaussians.opacity,
    features=gaussians.features,
)

# 可视化所有切片
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i < rendered_slices.shape[0]:
        ax.imshow(rendered_slices[i].cpu().numpy(), cmap='gray')
        ax.set_title(f'Slice {i}')
        ax.axis('off')
plt.tight_layout()
plt.savefig('rendered_slices.png')
```

### 保存渲染结果

```python
import nibabel as nib

# 渲染时间序列
sequence = renderer.render_with_time(
    means=gaussians.xyz,
    scales=gaussians.scale,
    rotations=gaussians.rotation,
    opacities=gaussians.opacity,
    features=gaussians.features,
    timestamps=torch.linspace(0, 1, 20),
    deformation_net=deformation_net,
)  # [T, num_slices, H, W]

# 保存为 NIfTI
sequence_np = sequence.cpu().numpy()
sequence_np = sequence_np.transpose(2, 3, 1, 0)  # [H, W, num_slices, T]

nifti_img = nib.Nifti1Image(sequence_np, affine=np.eye(4))
nib.save(nifti_img, 'rendered_sequence.nii.gz')
```

## 性能优化

### 1. 分块处理

渲染器自动将 Gaussians 分块处理以节省内存：

```python
renderer = Medical2DSliceRenderer(
    chunk_size=500,  # 减小以节省内存
)
```

**内存使用**: `O(chunk_size * H * W)`

### 2. 距离裁剪

只处理距离切片较近的 Gaussians：

```python
# 在渲染器内部自动进行
z_distances = torch.abs(means[:, 2] - slice_z)
max_distance = 3.0 * scales[:, 2].max()
near_mask = z_distances < max_distance
```

这可以显著减少计算量。

### 3. GPU 优化

```python
# 确保所有张量在 GPU 上
gaussians = gaussians.cuda()
renderer = renderer.cuda()

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    rendered = renderer(...)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. 批量渲染

```python
# 预计算协方差矩阵
covariances = renderer._compute_covariance(
    gaussians.scale,
    gaussians.rotation,
)

# 重复使用
for t in range(T):
    rendered = renderer._render_single_slice(
        slice_idx=0,
        means=deformed_means[t],
        covariances=covariances,  # 重用
        opacities=gaussians.opacity,
        features=gaussians.features,
    )
```

## 常见问题

### Q1: 渲染结果全黑？

**可能原因**:
1. Gaussians 的位置超出切片范围
2. 不透明度太小
3. 特征值为 0

**解决方案**:
```python
# 检查 Gaussian 位置
print(f"Mean position: {gaussians.xyz.mean(dim=0)}")
print(f"Position range: {gaussians.xyz.min(dim=0)[0]} to {gaussians.xyz.max(dim=0)[0]}")

# 检查切片位置
print(f"Slice positions: {renderer.slice_positions}")

# 检查不透明度
print(f"Opacity range: {gaussians.opacity.min()} to {gaussians.opacity.max()}")
```

### Q2: 渲染速度慢？

**优化策略**:
1. 减小 `chunk_size`
2. 减少 Gaussian 数量
3. 使用 GPU
4. 启用混合精度

```python
# 自适应 chunk size
num_gaussians = gaussians.num_points
optimal_chunk_size = min(1000, num_gaussians // 10)

renderer = Medical2DSliceRenderer(
    chunk_size=optimal_chunk_size,
)
```

### Q3: 内存不足？

**解决方案**:
```python
# 1. 减小 chunk_size
renderer = Medical2DSliceRenderer(
    chunk_size=200,  # 更小的块
)

# 2. 减小图像大小
renderer = Medical2DSliceRenderer(
    image_size=(128, 128),  # 更小的分辨率
)

# 3. 减少切片数量
renderer = Medical2DSliceRenderer(
    num_slices=5,  # 更少的切片
)

# 4. 使用梯度检查点
from torch.utils.checkpoint import checkpoint

rendered = checkpoint(
    renderer,
    means, scales, rotations, opacities, features
)
```

### Q4: 如何调试渲染器？

```python
# 启用调试模式
import torch
torch.autograd.set_detect_anomaly(True)

# 检查梯度
rendered = renderer(...)
loss = rendered.sum()
loss.backward()

print(f"Means grad: {gaussians.xyz.grad is not None}")
print(f"Scales grad: {gaussians.scale.grad is not None}")

# 可视化中间结果
import matplotlib.pyplot as plt

# 单个 Gaussian 的贡献
single_gaussian_render = renderer(
    means=gaussians.xyz[:1],  # 只渲染一个
    scales=gaussians.scale[:1],
    rotations=gaussians.rotation[:1],
    opacities=torch.ones(1, 1),  # 完全不透明
    features=torch.ones(1, 1),
)

plt.imshow(single_gaussian_render[0].cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title('Single Gaussian Contribution')
plt.savefig('debug_single_gaussian.png')
```

### Q5: 如何处理不同的体素间距？

```python
# 使用 VolumetricCamera
from dyna3dgr.rendering import VolumetricCamera

camera = VolumetricCamera(
    volume_size=(256, 256, 10),
    voxel_spacing=(1.0, 1.0, 5.0),  # 不同的间距
)

# 转换坐标
world_coords = camera.voxel_to_world(voxel_indices)
voxel_coords = camera.world_to_voxel(world_coords)
```

## 参考资料

- [MedGS 论文](https://arxiv.org/abs/2509.16806)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Dyna3DGR 论文](https://arxiv.org/abs/2507.16608)

## 下一步

- [训练指南](training.md)
- [数据准备](data_loading.md)
- [评估方法](evaluation.md)
