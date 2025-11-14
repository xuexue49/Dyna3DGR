# 数据加载指南

本文档介绍如何使用 Dyna3DGR 的数据加载器处理 ACDC 数据集。

## ACDC 数据集

ACDC (Automated Cardiac Diagnosis Challenge) 数据集包含150个患者的心脏MRI序列，涵盖5种心脏病理类型。

### 数据集特点

- **患者数量**: 150
- **病理类型**: 
  - NOR: 正常
  - MINF: 心肌梗死
  - DCM: 扩张型心肌病
  - HCM: 肥厚型心肌病
  - RV: 右心室异常
- **图像模态**: 短轴心脏MRI (cine-MRI)
- **时间帧数**: 每个患者约20-30帧（覆盖一个心动周期）
- **分割标注**: 
  - 0: 背景
  - 1: 右心室 (RV)
  - 2: 心肌 (MYO)
  - 3: 左心室 (LV)

### 下载数据集

1. 访问 [ACDC Challenge 官网](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
2. 注册账号
3. 下载训练集和测试集
4. 解压到本地目录

## 数据预处理

### 基本预处理

使用提供的预处理脚本处理原始数据：

```bash
python scripts/preprocess_data.py \
    --input_dir /path/to/raw/ACDC \
    --output_dir data/ACDC \
    --target_spacing 1.5 1.5 10.0 \
    --target_size 256 256 \
    --normalize \
    --extract_points \
    --num_points 10000
```

### 参数说明

- `--input_dir`: 原始ACDC数据目录
- `--output_dir`: 输出目录
- `--target_spacing`: 目标体素间距 (x, y, z) 单位mm
- `--target_size`: 目标图像大小 (H, W)
- `--normalize`: 是否归一化强度值
- `--extract_points`: 是否从分割中提取点云
- `--num_points`: 每帧采样的点数

### 预处理步骤

预处理脚本执行以下操作：

1. **重采样**: 将图像重采样到统一的体素间距
2. **归一化**: 将强度值归一化到 [0, 1]
3. **裁剪/填充**: 调整图像到目标大小
4. **点云提取**: 从分割掩码中采样点云（可选）

### 输出结构

预处理后的数据结构：

```
data/ACDC/
├── patient001/
│   ├── patient001_frame01.nii.gz
│   ├── patient001_frame01_gt.nii.gz
│   ├── patient001_frame02.nii.gz
│   ├── patient001_frame02_gt.nii.gz
│   ├── ...
│   ├── Info.cfg
│   └── point_clouds.json  # 如果启用了extract_points
├── patient002/
├── ...
└── preprocessing_metadata.json
```

## 使用数据加载器

### 基本使用

```python
from dyna3dgr.data import create_acdc_dataloader

# 创建训练集加载器
train_loader = create_acdc_dataloader(
    data_root='data/ACDC',
    split='train',
    batch_size=4,
    num_workers=4,
    image_size=(256, 256),
    num_frames=20,
    normalize=True,
    augmentation=True,
    load_segmentation=True,
)

# 迭代数据
for batch in train_loader:
    images = batch['images']          # [B, T, H, W, D]
    segmentations = batch['segmentations']  # [B, T, H, W, D]
    timestamps = batch['timestamps']   # [B, T]
    lengths = batch['lengths']         # [B]
    patient_ids = batch['patient_ids'] # List[str]
    
    # 训练代码...
```

### 数据加载器参数

#### ACDCDataset 参数

- `data_root` (str): ACDC数据集根目录
- `split` (str): 数据集划分 ('train', 'val', 'test')
- `image_size` (Tuple[int, int]): 目标图像大小 (H, W)
- `num_frames` (Optional[int]): 采样的帧数（None表示加载所有帧）
- `normalize` (bool): 是否归一化图像
- `augmentation` (bool): 是否应用数据增强
- `load_segmentation` (bool): 是否加载分割标注
- `cache_data` (bool): 是否缓存数据到内存

#### DataLoader 参数

- `batch_size` (int): 批大小
- `num_workers` (int): 数据加载进程数
- `shuffle` (bool): 是否打乱数据（训练集自动启用）
- `pin_memory` (bool): 是否固定内存（加速GPU传输）

### 数据格式

#### 批数据结构

```python
batch = {
    'images': Tensor[B, T, H, W, D],        # 图像序列
    'segmentations': Tensor[B, T, H, W, D], # 分割序列（可选）
    'timestamps': Tensor[B, T],             # 时间戳（归一化到[0,1]）
    'lengths': Tensor[B],                   # 每个样本的实际帧数
    'patient_ids': List[str],               # 患者ID列表
    'metadata': List[Dict],                 # 元数据列表
}
```

#### 元数据结构

```python
metadata = {
    'info': {
        'ED': '1',              # 舒张末期帧号
        'ES': '10',             # 收缩末期帧号
        'Group': 'NOR',         # 病理组
        'Height': '170',        # 身高 (cm)
        'Weight': '70',         # 体重 (kg)
        # ...
    },
    'num_frames': 20,           # 帧数
    'voxel_spacing': [1.5, 1.5, 10.0],  # 体素间距
}
```

### 数据增强

训练时自动应用的数据增强：

1. **水平翻转**: 50%概率
2. **旋转**: ±10度范围内随机旋转
3. **强度偏移**: ±0.1范围内随机偏移
4. **强度缩放**: 0.9-1.1范围内随机缩放

可以通过 `augmentation=False` 禁用数据增强。

### 处理变长序列

由于不同患者的帧数可能不同，数据加载器会自动填充到批次中的最大长度：

```python
for batch in train_loader:
    images = batch['images']      # [B, T_max, H, W, D]
    lengths = batch['lengths']    # [B]
    
    # 处理每个样本
    for i in range(batch_size):
        actual_length = lengths[i].item()
        valid_images = images[i, :actual_length]  # [T_actual, H, W, D]
        
        # 处理有效帧...
```

## 测试数据加载器

使用测试脚本验证数据加载器：

```bash
python scripts/test_dataloader.py \
    --data_root data/ACDC \
    --split train \
    --batch_size 2 \
    --num_samples 5 \
    --output_dir outputs/dataloader_test
```

这将：
1. 加载数据集
2. 打印批次信息
3. 可视化样本数据
4. 保存可视化结果到输出目录

## 高级用法

### 自定义数据集划分

默认划分为 70% 训练、15% 验证、15% 测试。如需自定义：

```python
from dyna3dgr.data import ACDCDataset

class CustomACDCDataset(ACDCDataset):
    def _split_dataset(self):
        # 自定义划分逻辑
        if self.split == 'train':
            return self.patient_dirs[:100]
        elif self.split == 'val':
            return self.patient_dirs[100:125]
        else:
            return self.patient_dirs[125:]
```

### 只加载特定病理类型

```python
from dyna3dgr.data import ACDCDataset

class FilteredACDCDataset(ACDCDataset):
    def _find_patients(self):
        patient_dirs = super()._find_patients()
        
        # 只保留正常患者
        filtered = []
        for patient_dir in patient_dirs:
            info = self._load_info(patient_dir)
            if info.get('Group') == 'NOR':
                filtered.append(patient_dir)
        
        return filtered
```

### 自定义数据增强

```python
from dyna3dgr.data import ACDCDataset

class AugmentedACDCDataset(ACDCDataset):
    def _augment(self, image, segmentation):
        # 调用父类增强
        image, segmentation = super()._augment(image, segmentation)
        
        # 添加自定义增强
        if np.random.rand() > 0.5:
            # 添加高斯噪声
            noise = np.random.normal(0, 0.01, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image, segmentation
```

## 性能优化

### 内存优化

如果内存充足，可以启用数据缓存：

```python
train_loader = create_acdc_dataloader(
    data_root='data/ACDC',
    split='train',
    cache_data=True,  # 缓存到内存
    # ...
)
```

### 加载速度优化

1. **增加工作进程数**:
```python
train_loader = create_acdc_dataloader(
    num_workers=8,  # 根据CPU核心数调整
    pin_memory=True,
    # ...
)
```

2. **预处理数据**: 使用预处理脚本提前处理数据

3. **减少帧数**: 如果不需要所有帧，可以采样：
```python
train_loader = create_acdc_dataloader(
    num_frames=10,  # 只加载10帧
    # ...
)
```

## 常见问题

### Q1: 数据加载很慢

**解决方案**:
1. 增加 `num_workers`
2. 使用预处理脚本
3. 启用 `cache_data`（如果内存充足）
4. 使用SSD存储数据

### Q2: 内存不足

**解决方案**:
1. 减小 `batch_size`
2. 减少 `num_frames`
3. 禁用 `cache_data`
4. 减小 `image_size`

### Q3: 分割标注缺失

**解决方案**:
```python
train_loader = create_acdc_dataloader(
    load_segmentation=False,  # 不加载分割
    # ...
)
```

### Q4: 数据增强导致标注错位

确保图像和分割使用相同的变换：

```python
def _augment(self, image, segmentation):
    # 使用相同的随机种子
    state = np.random.get_state()
    
    # 变换图像
    image = transform(image)
    
    # 使用相同的随机状态变换分割
    np.random.set_state(state)
    segmentation = transform(segmentation)
    
    return image, segmentation
```

## 示例代码

### 完整训练循环示例

```python
from dyna3dgr.data import create_acdc_dataloader
import torch

# 创建数据加载器
train_loader = create_acdc_dataloader(
    data_root='data/ACDC',
    split='train',
    batch_size=4,
    num_workers=4,
)

val_loader = create_acdc_dataloader(
    data_root='data/ACDC',
    split='val',
    batch_size=4,
    num_workers=4,
    augmentation=False,
)

# 训练循环
for epoch in range(num_epochs):
    # 训练
    model.train()
    for batch in train_loader:
        images = batch['images'].to(device)
        timestamps = batch['timestamps'].to(device)
        lengths = batch['lengths']
        
        # 前向传播
        outputs = model(images, timestamps)
        
        # 计算损失
        loss = criterion(outputs, images)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            # 验证代码...
```

## 下一步

- 查看 [训练指南](usage.md#训练模型) 了解如何使用数据加载器进行训练
- 查看 [API 文档](api.md) 了解详细的API参考
- 查看示例 notebooks 学习更多用法
