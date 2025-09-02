#!/bin/bash
set -e  # Exit immediately if a command fails

# Step 1: Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Step 2 & 3: Download Miniconda installer
if [[ "$ARCH" == "aarch64" ]]; then
    echo "Downloading Miniconda for ARM64 (aarch64)..."
    wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "Downloading Miniconda for x86_64..."
    curl -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Step 4: Install Miniconda
bash Miniconda3.sh -b -p $HOME/miniconda
rm Miniconda3.sh

# Step 5: Initialize conda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc || true

# Step 6 & 7: Accept terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Step 8: Create and activate environment
echo "Creating conda environment: guardian-loop"
conda create -n guardian-loop python=3.10 -y
conda activate guardian-loop

# Step 9: Install required packages
echo "Installing Python packages..."
conda install -c conda-forge numpy pandas scikit-learn scipy matplotlib ipykernel pyyaml -y

# Step 10: Detect CUDA version for PyTorch
TORCH_INDEX_URL="https://download.pytorch.org/whl"

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -n1)
    if [[ -n "$CUDA_VERSION" ]]; then
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        CU_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"

        echo "Detected CUDA $CUDA_VERSION → Installing PyTorch for $CU_TAG"
        TORCH_URL="$TORCH_INDEX_URL/$CU_TAG"
    else
        TORCH_URL="$TORCH_INDEX_URL/cpu"
        echo "CUDA not detected → Installing CPU-only PyTorch"
    fi
else
    TORCH_URL="$TORCH_INDEX_URL/cpu"
    echo "nvidia-smi not found → Installing CPU-only PyTorch"
fi

# Step 11: Install torch & torchvision
pip install --upgrade --force-reinstall torch torchvision --index-url $TORCH_URL

# Step 12: Install other libraries
pip install openai langchain langgraph transformers sentence-transformers datasets

echo "✅ Setup complete. Run: conda activate guardian-loop"
