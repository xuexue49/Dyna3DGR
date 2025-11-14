#!/bin/bash

# Two-Step Training Script for Dyna3DGR
#
# This script automates the complete two-step training pipeline:
#   Step 1: Train Gaussian representation only (static reconstruction)
#   Step 2: Train complete dynamic model (with deformation)
#
# Usage:
#   ./scripts/train_two_step.sh <patient_dir> <output_dir> [config] [device]
#
# Example:
#   ./scripts/train_two_step.sh data/ACDC/training/patient001 outputs/patient001
#   ./scripts/train_two_step.sh data/ACDC/training/patient001 outputs/patient001 configs/two_step_training.yaml cuda

set -e  # Exit on error

# ============================================================
# Parse Arguments
# ============================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <patient_dir> <output_dir> [config] [device]"
    echo ""
    echo "Arguments:"
    echo "  patient_dir   Path to patient directory (e.g., data/ACDC/training/patient001)"
    echo "  output_dir    Output directory for checkpoints and logs"
    echo "  config        (Optional) Path to config file (default: configs/two_step_training.yaml)"
    echo "  device        (Optional) Device to use: cuda or cpu (default: cuda)"
    echo ""
    echo "Example:"
    echo "  $0 data/ACDC/training/patient001 outputs/patient001"
    echo "  $0 data/ACDC/training/patient001 outputs/patient001 configs/two_step_training.yaml cuda"
    exit 1
fi

PATIENT_DIR=$1
OUTPUT_DIR=$2
CONFIG=${3:-configs/two_step_training.yaml}
DEVICE=${4:-cuda}

# ============================================================
# Validate Inputs
# ============================================================

echo "============================================================"
echo "Two-Step Training Pipeline"
echo "============================================================"
echo "Patient directory: $PATIENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG"
echo "Device: $DEVICE"
echo "============================================================"
echo ""

# Check if patient directory exists
if [ ! -d "$PATIENT_DIR" ]; then
    echo "Error: Patient directory not found: $PATIENT_DIR"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Step 1: Train Gaussian Representation Only
# ============================================================

echo ""
echo "============================================================"
echo "STEP 1: Training Gaussian Representation Only"
echo "============================================================"
echo ""

STEP1_OUTPUT="$OUTPUT_DIR/step1_gaussians"

python scripts/train_step1_gaussians.py \
    --config "$CONFIG" \
    --patient_dir "$PATIENT_DIR" \
    --output_dir "$STEP1_OUTPUT" \
    --device "$DEVICE"

# Check if Step 1 completed successfully
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Step 1 training failed"
    exit 1
fi

echo ""
echo "✓ Step 1 completed successfully"
echo "  Checkpoint saved to: $STEP1_OUTPUT/checkpoints/best.pth"
echo ""

# ============================================================
# Step 2: Train Complete Dynamic Model
# ============================================================

echo ""
echo "============================================================"
echo "STEP 2: Training Complete Dynamic Model"
echo "============================================================"
echo ""

STEP2_OUTPUT="$OUTPUT_DIR/step2_dynamic"
STEP1_CHECKPOINT="$STEP1_OUTPUT/checkpoints/best.pth"

# Check if Step 1 checkpoint exists
if [ ! -f "$STEP1_CHECKPOINT" ]; then
    echo "Error: Step 1 checkpoint not found: $STEP1_CHECKPOINT"
    exit 1
fi

python scripts/train_step2_dynamic.py \
    --config "$CONFIG" \
    --patient_dir "$PATIENT_DIR" \
    --step1_checkpoint "$STEP1_CHECKPOINT" \
    --output_dir "$STEP2_OUTPUT" \
    --device "$DEVICE"

# Check if Step 2 completed successfully
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Step 2 training failed"
    exit 1
fi

echo ""
echo "✓ Step 2 completed successfully"
echo "  Checkpoint saved to: $STEP2_OUTPUT/checkpoints/best.pth"
echo ""

# ============================================================
# Training Complete
# ============================================================

echo ""
echo "============================================================"
echo "TWO-STEP TRAINING COMPLETED SUCCESSFULLY!"
echo "============================================================"
echo ""
echo "Output structure:"
echo "  $OUTPUT_DIR/"
echo "  ├── step1_gaussians/"
echo "  │   ├── checkpoints/"
echo "  │   │   ├── best.pth          # Best Step 1 model"
echo "  │   │   └── ..."
echo "  │   ├── logs/                 # TensorBoard logs"
echo "  │   └── config_step1.yaml     # Step 1 config"
echo "  └── step2_dynamic/"
echo "      ├── checkpoints/"
echo "      │   ├── best.pth          # Best Step 2 model (FINAL)"
echo "      │   └── ..."
echo "      ├── logs/                 # TensorBoard logs"
echo "      └── config_step2.yaml     # Step 2 config"
echo ""
echo "Final model: $STEP2_OUTPUT/checkpoints/best.pth"
echo ""
echo "Next steps:"
echo "  1. Visualize results:"
echo "     python scripts/visualize_results.py \\"
echo "       --checkpoint $STEP2_OUTPUT/checkpoints/best.pth \\"
echo "       --patient_dir $PATIENT_DIR \\"
echo "       --mode interactive"
echo ""
echo "  2. Evaluate model:"
echo "     python scripts/evaluate.py \\"
echo "       --checkpoint $STEP2_OUTPUT/checkpoints/best.pth \\"
echo "       --patient_dir $PATIENT_DIR"
echo ""
echo "  3. View training logs:"
echo "     tensorboard --logdir $OUTPUT_DIR"
echo ""
echo "============================================================"
