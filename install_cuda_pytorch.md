# Installing CUDA-enabled PyTorch for GPU Training

## Current Status

Your PyTorch installation is **CPU-only**. To train on GPU, you need to install CUDA-enabled PyTorch.

## Check Your System

1. **Check if you have an NVIDIA GPU:**

   ```powershell
   nvidia-smi
   ```

   If this command works, you have an NVIDIA GPU and drivers installed.

2. **Check CUDA version:**
   The output of `nvidia-smi` will show your CUDA version (e.g., CUDA 11.8, 12.1, etc.)

## Install CUDA-enabled PyTorch

### Option 1: CUDA 11.8 (Most Compatible)

```powershell
# Uninstall CPU-only PyTorch
uv pip uninstall torch torchvision

# Install CUDA 11.8 version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: CUDA 12.1 (If your GPU supports it)

```powershell
# Uninstall CPU-only PyTorch
uv pip uninstall torch torchvision

# Install CUDA 12.1 version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Option 3: Let PyTorch Auto-detect

```powershell
# Uninstall CPU-only PyTorch
uv pip uninstall torch torchvision

# Install with auto-detection
uv pip install torch torchvision
```

## Verify Installation

After installing, verify GPU is available:

```powershell
python check_gpu.py
```

Or:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Start Training on GPU

Once CUDA-enabled PyTorch is installed:

```powershell
python train_model.py --device cuda
```

## If You Don't Have an NVIDIA GPU

If you don't have an NVIDIA GPU, you can still train on CPU (it will be slower):

```powershell
python train_model.py --device cpu
```

Or just let it auto-detect:

```powershell
python train_model.py
```

## Training Time Estimates

- **GPU (NVIDIA)**: ~30-60 minutes for 100 epochs
- **CPU**: ~4-8 hours for 100 epochs

## Troubleshooting

### "CUDA out of memory" error

Reduce batch size:

```powershell
python train_model.py --device cuda --batch 8
```

### "No CUDA GPUs are available"

1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA-enabled PyTorch is installed
3. Restart your terminal/IDE after installation
