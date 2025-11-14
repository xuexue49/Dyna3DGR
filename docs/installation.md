# 安装指南

本文档提供 Dyna3DGR 的详细安装说明。

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+
  - 推荐: RTX 3090, RTX 4090, A100, V100
  - 最低: GTX 1080 Ti, RTX 2080
- **显存**: 建议 12GB+ VRAM
  - 训练: 12-24GB
  - 推理: 8GB+
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
  - Windows 和 macOS 理论上可行，但未经充分测试
- **Python**: 3.8, 3.9, 或 3.10
- **CUDA**: 11.8 或 12.1+
- **GCC**: 7.5+ (用于编译CUDA扩展)

## 安装步骤

### 方法 1: 使用 Conda (推荐)

这是最简单和推荐的安装方法。

#### 1. 安装 Conda

如果您还没有安装 Conda，请从 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/products/distribution) 下载并安装。

#### 2. 克隆仓库

```bash
git clone https://github.com/xuexue49/Dyna3DGR.git
cd Dyna3DGR
```

#### 3. 初始化子模块

```bash
git submodule update --init --recursive
```

#### 4. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate dyna3dgr
```

这将自动安装所有依赖项，包括 PyTorch 和 CUDA。

#### 5. 安装 CUDA 扩展

```bash
# 安装 diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization

# 安装 simple-knn
pip install submodules/simple-knn
```

#### 6. 验证安装

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# 检查 Dyna3DGR 是否正确安装
python -c "import dyna3dgr; print('Dyna3DGR installed successfully!')"
```

### 方法 2: 使用 pip 和虚拟环境

如果您不想使用 Conda，可以使用 Python 的虚拟环境。

#### 1. 克隆仓库

```bash
git clone https://github.com/xuexue49/Dyna3DGR.git
cd Dyna3DGR
git submodule update --init --recursive
```

#### 2. 创建虚拟环境

```bash
python3.8 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows
```

#### 3. 安装 PyTorch

根据您的 CUDA 版本安装 PyTorch。访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合您系统的安装命令。

例如，对于 CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 4. 安装依赖

```bash
pip install -r requirements.txt
```

#### 5. 安装 CUDA 扩展

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

#### 6. 验证安装

同方法 1 的步骤 6。

## 常见问题

### Q1: CUDA 扩展编译失败

**问题**: 安装 `diff-gaussian-rasterization` 或 `simple-knn` 时出现编译错误。

**解决方案**:
1. 确保安装了正确版本的 CUDA Toolkit
2. 确保 GCC 版本兼容 (推荐 GCC 7.5-9.x)
3. 检查 PyTorch 的 CUDA 版本与系统 CUDA 版本是否匹配

```bash
# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 检查系统 CUDA 版本
nvcc --version
```

### Q2: 显存不足 (Out of Memory)

**问题**: 训练时出现 CUDA out of memory 错误。

**解决方案**:
1. 减小 batch size
2. 减少 Gaussian 数量
3. 使用混合精度训练
4. 使用梯度累积

在配置文件中调整:

```yaml
training:
  batch_size: 2  # 减小 batch size
model:
  gaussian:
    num_points: 5000  # 减少 Gaussian 数量
hardware:
  mixed_precision: true  # 启用混合精度
```

### Q3: 子模块未找到

**问题**: 运行时提示找不到 `diff_gaussian_rasterization` 或 `simple_knn`。

**解决方案**:

```bash
# 确保子模块已初始化
git submodule update --init --recursive

# 重新安装 CUDA 扩展
pip install --force-reinstall submodules/diff-gaussian-rasterization
pip install --force-reinstall submodules/simple-knn
```

### Q4: ImportError: libcudart.so.XX not found

**问题**: 运行时找不到 CUDA 库文件。

**解决方案**:

```bash
# 添加 CUDA 库路径到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 或者在 ~/.bashrc 中添加
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Q5: Windows 上的安装问题

**问题**: 在 Windows 上安装 CUDA 扩展失败。

**解决方案**:
1. 安装 Visual Studio 2019 或更新版本
2. 确保安装了 "Desktop development with C++" 工作负载
3. 使用 x64 Native Tools Command Prompt for VS 2019 运行安装命令

## 开发者安装

如果您想修改代码并贡献到项目，建议使用开发者模式安装:

```bash
# 克隆仓库
git clone https://github.com/xuexue49/Dyna3DGR.git
cd Dyna3DGR

# 创建环境
conda env create -f environment.yml
conda activate dyna3dgr

# 以可编辑模式安装
pip install -e .

# 安装开发依赖
pip install -r requirements-dev.txt
```

## 卸载

如果需要卸载 Dyna3DGR:

```bash
# 如果使用 Conda
conda env remove -n dyna3dgr

# 如果使用 pip
pip uninstall dyna3dgr

# 删除仓库
rm -rf Dyna3DGR
```

## 下一步

安装完成后，请参考以下文档:
- [使用指南](usage.md) - 学习如何使用 Dyna3DGR
- [快速开始](../README.md#快速开始) - 运行第一个示例
- [API 文档](api.md) - 详细的 API 参考

## 获取帮助

如果遇到安装问题:
1. 查看本文档的常见问题部分
2. 搜索 [GitHub Issues](https://github.com/xuexue49/Dyna3DGR/issues)
3. 提交新的 Issue 并提供详细的错误信息
