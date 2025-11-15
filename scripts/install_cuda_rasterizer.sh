#!/bin/bash
# Install diff-gaussian-rasterization for fast CUDA rendering

set -e

echo "Installing diff-gaussian-rasterization..."

# Clone repository
if [ ! -d "submodules/diff-gaussian-rasterization" ]; then
    mkdir -p submodules
    cd submodules
    git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
    cd ..
fi

# Install
cd submodules/diff-gaussian-rasterization
pip install .
cd ../..

echo "✓ Installation complete!"
echo ""
echo "To enable CUDA rendering, set in config:"
echo "  use_cuda_rasterizer: true"
