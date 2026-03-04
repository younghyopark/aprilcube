#!/usr/bin/env bash
# Generate all cube models under models/ directory.
# Each model is named {grid}_{tagsize}_cube.

set -euo pipefail
cd "$(dirname "$0")"

DICT="4x4_100"
MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

# Define models: grid tag_size
MODELS=(
    "1x1x1 24"
    "1x1x1 30"
    "2x1x1 24"
    "2x1x1 30"
    "2x2x1 24"
    "2x2x1 30"
    "2x2x2 24"
    "2x2x2 30"
    "3x1x1 24"
    "3x1x1 30"
    "3x2x1 24"
    "3x2x1 30"
    "3x3x1 24"
    "3x3x1 30"
    "4x3x1 24"
    "4x3x1 30"
)

for entry in "${MODELS[@]}"; do
    read -r grid tag_size <<< "$entry"
    name="${grid}_${tag_size}_cube"
    out="$MODELS_DIR/$name"
    echo "=== $name ==="
    python generate_cube.py \
        --grid "$grid" \
        --dict "$DICT" \
        --tag-size "$tag_size" \
        -o "$out"
    echo ""
done

echo "All models generated in $MODELS_DIR/"
