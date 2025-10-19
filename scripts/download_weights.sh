#!/bin/bash
# Download pretrained StyleGAN2-FFHQ weights
#
# This script attempts to download StyleGAN2-FFHQ PyTorch weights
# from common sources. If automatic download fails, manual fallback
# instructions are provided.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CHECKPOINT_DIR="checkpoints"
WEIGHT_FILE="stylegan2-ffhq-1024x1024.pt"
WEIGHT_PATH="${CHECKPOINT_DIR}/${WEIGHT_FILE}"

# Create checkpoint directory
echo "Creating checkpoint directory..."
mkdir -p "${CHECKPOINT_DIR}"

# Check if weights already exist
if [ -f "${WEIGHT_PATH}" ]; then
    echo -e "${GREEN}✓ Weights already exist at ${WEIGHT_PATH}${NC}"
    echo "File size: $(du -h "${WEIGHT_PATH}" | cut -f1)"
    read -p "Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

echo "Attempting to download StyleGAN2-FFHQ weights..."
echo

# Method 1: Try downloading from rosinality's repository (converted weights)
echo "Trying Method 1: rosinality StyleGAN2-PyTorch (converted weights)..."
ROSINALITY_URL="https://github.com/rosinality/stylegan2-pytorch/releases/download/weights/stylegan2-ffhq-config-f.pt"

if command -v wget &> /dev/null; then
    echo "Using wget..."
    if wget -O "${WEIGHT_PATH}" "${ROSINALITY_URL}" 2>/dev/null; then
        echo -e "${GREEN}✓ Successfully downloaded weights!${NC}"
        echo "File size: $(du -h "${WEIGHT_PATH}" | cut -f1)"
        exit 0
    else
        echo -e "${YELLOW}⚠ Method 1 failed${NC}"
        rm -f "${WEIGHT_PATH}"
    fi
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    if curl -L -o "${WEIGHT_PATH}" "${ROSINALITY_URL}" 2>/dev/null; then
        echo -e "${GREEN}✓ Successfully downloaded weights!${NC}"
        echo "File size: $(du -h "${WEIGHT_PATH}" | cut -f1)"
        exit 0
    else
        echo -e "${YELLOW}⚠ Method 1 failed${NC}"
        rm -f "${WEIGHT_PATH}"
    fi
else
    echo -e "${YELLOW}⚠ Neither wget nor curl found${NC}"
fi

echo

# Method 2: Try alternative source (NVIDIA Drive)
echo "Trying Method 2: NVIDIA Drive (official weights)..."
NVIDIA_URL="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

if command -v wget &> /dev/null; then
    if wget -O "${CHECKPOINT_DIR}/stylegan2-ffhq-config-f.pkl" "${NVIDIA_URL}" 2>/dev/null; then
        echo -e "${GREEN}✓ Downloaded official .pkl format${NC}"
        echo -e "${YELLOW}Note: You may need to convert .pkl to .pt format${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ Method 2 failed${NC}"
    fi
elif command -v curl &> /dev/null; then
    if curl -L -o "${CHECKPOINT_DIR}/stylegan2-ffhq-config-f.pkl" "${NVIDIA_URL}" 2>/dev/null; then
        echo -e "${GREEN}✓ Downloaded official .pkl format${NC}"
        echo -e "${YELLOW}Note: You may need to convert .pkl to .pt format${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ Method 2 failed${NC}"
    fi
fi

echo
echo -e "${RED}✗ Automatic download failed${NC}"
echo
echo "═══════════════════════════════════════════════════════════════"
echo "MANUAL DOWNLOAD INSTRUCTIONS"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "Option 1: Rosinality StyleGAN2-PyTorch (Recommended)"
echo "  1. Visit: https://github.com/rosinality/stylegan2-pytorch"
echo "  2. Download: stylegan2-ffhq-config-f.pt from releases"
echo "  3. Place in: ${CHECKPOINT_DIR}/"
echo "  4. Rename to: ${WEIGHT_FILE}"
echo
echo "Option 2: Official NVIDIA StyleGAN2-ADA"
echo "  1. Visit: https://github.com/NVlabs/stylegan2-ada-pytorch"
echo "  2. Download: ffhq.pkl from pretrained models"
echo "  3. Convert: Use provided conversion script (if needed)"
echo "  4. Place in: ${CHECKPOINT_DIR}/"
echo
echo "Option 3: Hugging Face Hub"
echo "  1. Visit: https://huggingface.co/spaces/hysts/StyleGAN2"
echo "  2. Look for downloadable model files"
echo "  3. Place in: ${CHECKPOINT_DIR}/"
echo
echo "Direct download URLs:"
echo "  • ${ROSINALITY_URL}"
echo "  • ${NVIDIA_URL}"
echo
echo "After manual download, verify with:"
echo "  ls -lh ${WEIGHT_PATH}"
echo "  python -c 'import torch; print(torch.load(\"${WEIGHT_PATH}\", map_location=\"cpu\").keys())'"
echo
echo "═══════════════════════════════════════════════════════════════"

exit 1

