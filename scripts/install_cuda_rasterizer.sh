#!/bin/bash
# Install diff-gaussian-rasterization for fast CUDA rendering

set -e

echo "Installing diff-gaussian-rasterization..."
echo ""

# Detect CUDA from conda environment
if [ -n "$CONDA_PREFIX" ]; then
    echo "Detected conda environment: $CONDA_PREFIX"
    export CUDA_HOME=$CONDA_PREFIX
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
    echo "Using conda CUDA: $CUDA_HOME"
else
    echo "Warning: Not in conda environment, using system CUDA"
fi

echo ""

# Clone repository
if [ ! -d "submodules/diff-gaussian-rasterization" ]; then
    echo "Cloning diff-gaussian-rasterization..."
    mkdir -p submodules
    cd submodules
    git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
    cd ..
fi

# Install (use --no-build-isolation to use current environment's torch)
echo "Installing..."
cd submodules/diff-gaussian-rasterization
pip install --no-build-isolation .
cd ../..

echo ""
echo "✓ Installation complete!"
echo ""
echo "To enable CUDA rendering, set in config:"
echo "  use_cuda_rasterizer: true"
echo "  use_volume_renderer: false"
