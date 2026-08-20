# Troubleshooting & Common Issues Guide

## 1. CUDA Out-Of-Memory (OOM)

### Symptoms
`torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...`

### Solution
1. **Reduce Micro-Batch Size**: Ensure `batch_size <= 4` and compensate with `accum_steps = 8`.
2. **Cap Sequence Length**: Set `block_size <= 512`.
3. **Enable AMP**: Ensure `USE_AMP=true` in `.env` to train in 16-bit precision.

---

## 2. PyTorch CUDA Not Available / CPU Fallback

### Symptoms
`[GenkitCore] Model allocated on CPU (CUDA not detected)`

### Solution
Install PyTorch with CUDA matching your GPU driver:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
Verify with:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

---

## 3. Database Connection Issues

### Behavior
If MySQL is not running or credentials fail, Genkit AI automatically switches to local SQLite (`backend/genkit.db`). No configuration or manual intervention is required.

---

## 4. Port Conflicts

If port 8000 or 5173 is already in use:
```bash
# Override backend port
python app/main.py --port 8001
```
