#!/bin/bash

# Update conda (optional but recommended)
echo "Updating conda..."
conda update -n base -c defaults conda -y

# Create environment with Python 3.10 (better PyTorch support)
echo "Creating conda environment with Python 3.10..."
conda create -n virt_solo_learn python=3.10 -y

# Use conda run to execute commands in the environment
echo "Installing PyTorch with CUDA support..."
conda run -n virt_solo_learn conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining packages using conda run
echo "Installing other requirements..."
conda run -n virt_solo_learn pip install einops
conda run -n virt_solo_learn pip install lightning==2.1.2
conda run -n virt_solo_learn pip install "torchmetrics>=0.6.0,<0.12.0"
conda run -n virt_solo_learn pip install tqdm
conda run -n virt_solo_learn pip install wandb
conda run -n virt_solo_learn pip install scipy
conda run -n virt_solo_learn pip install timm
conda run -n virt_solo_learn pip install scikit-learn
conda run -n virt_solo_learn pip install hydra-core

# Optional: Install additional useful packages
conda run -n virt_solo_learn pip install jupyter ipykernel matplotlib pandas

# Verify installation
echo "Verifying installation..."
conda run -n virt_solo_learn python -c "import torch; print(f'PyTorch: {torch.__version__}')"
conda run -n virt_solo_learn python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
conda run -n virt_solo_learn python -c "import lightning; print(f'Lightning: {lightning.__version__}')"

echo "Installation complete!"
echo "To activate the environment, run: conda activate virt_solo_learn"