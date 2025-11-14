#!/bin/bash
# Quick training script for Dyna3DGR
# This script provides a simple way to start training with default settings

# Default values
DATA_ROOT="data/ACDC"
OUTPUT_DIR="outputs/experiment_$(date +%Y%m%d_%H%M%S)"
CONFIG="configs/acdc.yaml"
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--data_root PATH] [--output_dir PATH] [--config PATH] [--device cuda|cpu] [--debug] [--resume PATH]"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=================================="
echo "Dyna3DGR Training"
echo "=================================="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Config: $CONFIG"
echo "Device: $DEVICE"
echo "=================================="
echo ""

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data directory not found: $DATA_ROOT"
    echo "Please run data preprocessing first:"
    echo "  python scripts/preprocess_data.py --input_dir /path/to/raw/ACDC --output_dir $DATA_ROOT"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Run training
python scripts/train.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    $DEBUG \
    $RESUME

echo ""
echo "=================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================="
