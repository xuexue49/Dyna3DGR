#!/bin/bash
# Training script for single patient (per-case optimization)
#
# Usage:
#   bash scripts/train_patient.sh <patient_dir> <output_dir>
#
# Example:
#   bash scripts/train_patient.sh data/ACDC/patient001 outputs/patient001

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <patient_dir> <output_dir>"
    echo "Example: $0 data/ACDC/patient001 outputs/patient001"
    exit 1
fi

PATIENT_DIR=$1
OUTPUT_DIR=$2

# Configuration
CONFIG="configs/acdc_paper.yaml"
DEVICE="cuda"  # or "cpu"

# Print info
echo "============================================================"
echo "Dyna3DGR Training (Per-Case Optimization)"
echo "============================================================"
echo "Patient directory: $PATIENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Configuration: $CONFIG"
echo "Device: $DEVICE"
echo "============================================================"
echo ""

# Run training
python scripts/train.py \
    --config "$CONFIG" \
    --patient_dir "$PATIENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

# Print completion
echo ""
echo "============================================================"
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints"
echo "Logs saved to: $OUTPUT_DIR/logs"
echo "============================================================"
echo ""
echo "To visualize results:"
echo "  python scripts/visualize_results.py \\"
echo "    --checkpoint $OUTPUT_DIR/checkpoints/best.pth \\"
echo "    --patient_dir $PATIENT_DIR \\"
echo "    --mode interactive"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
