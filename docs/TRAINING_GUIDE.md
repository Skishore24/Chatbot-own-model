# 🏋️ GENKIT AI — Model Training & Fine-Tuning Guide

## PyTorch Training Parameters

- **Learning Rate**: $3 \times 10^{-4}$ (AdamW Optimizer, Weight Decay $\lambda = 0.1$).
- **Scheduler**: Cosine Annealing with Linear Warmup (500 warmup steps, min LR $10^{-5}$).
- **Automatic Mixed Precision (AMP)**: `torch.cuda.amp.autocast(dtype=torch.bfloat16)`.
- **Gradient Accumulation**: Step factor $N=4$ for effective batch size of 128.
- **Gradient Clipping**: $\|g\|_2 \le 1.0$.

## Execution Command

```bash
# Execute training script
python scripts/train.py
```
Checkpoints will be saved automatically to `backend/checkpoints/model_v5.pt`.
