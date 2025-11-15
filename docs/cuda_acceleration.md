# CUDA 加速渲染

## 安装

```bash
bash scripts/install_cuda_rasterizer.sh
```

## 启用

在配置文件中设置：
```yaml
use_cuda_rasterizer: true  # 使用 CUDA 加速
use_volume_renderer: false  # 暂时禁用完整体积渲染
```

## 速度对比

| 方法 | 速度 | 质量 |
|------|------|------|
| PyTorch 体积渲染 | 1x (慢) | 高 |
| PyTorch 单切片 | 10x | 中 |
| **CUDA 单切片** | **50x** | 中 |

## 临时方案

如果 CUDA 安装失败，可以：

1. **降低分辨率**
```yaml
image_size: [64, 64, 10]  # 从 [128, 128, 10]
```

2. **减少 Gaussians**
```yaml
num_gaussians: 2000  # 从 5000
```

3. **使用单切片**
```yaml
use_volume_renderer: false
```
