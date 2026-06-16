# Hardware Profile

## System

OS:
CPU:
RAM:

## GPU

GPU:
Driver version:
CUDA version:
Compute capability:
Global memory:
SM count:
Warp size:
Max threads per block:

## PyTorch

PyTorch version:
CUDA available:
PyTorch CUDA version:
GPU name from torch:

## Commands Ran

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
