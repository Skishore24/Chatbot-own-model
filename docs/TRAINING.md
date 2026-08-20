# Model Training & GPU Acceleration Guide

## 1. Hardware Target Specifications

This training pipeline is optimized for:
- **GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (`CUDA 12.x / 13.x`)
- **System Memory**: 16 GB RAM
- **Processor**: Intel Core i5 13th Gen HX
- **Operating System**: Windows / Linux

---

## 2. Fast Launching

To launch the full training pipeline using the GPU-configured virtual environment:

```bash
# 1. Navigate to backend directory
cd backend

# 2. Run consolidated training pipeline
python train.py --epochs 60 --batch-size 4 --accum-steps 8 --block-size 512 --vocab-size 10000
```

---

## 3. Training Architecture & Hyperparameter Design

To prevent Out-Of-Memory (OOM) on a 6GB VRAM GPU while maintaining high learning stability:

1. **Micro-Batching + Gradient Accumulation**:
   - `micro_batch_size = 4` (occupies ~2.1 GB VRAM)
   - `gradient_accumulation_steps = 8`
   - **Effective Batch Size**: $4 \times 8 = 32$
2. **Automatic Mixed Precision (AMP)**:
   - Uses `torch.amp.autocast('cuda', dtype=torch.bfloat16)` if supported or `torch.float16` with `GradScaler`.
3. **Cosine Annealing with Warmup**:
   - Linear warmup for the first 200 steps to prevent transformer divergence.
   - Cosine decay down to minimum learning rate ($1\times 10^{-5}$).
4. **Gradient Clipping**:
   - Maximum gradient norm clipped to $1.0$ using `torch.nn.utils.clip_grad_norm_`.

---

## 4. Checkpoints & Outputs

Upon training completion, the pipeline outputs:
- `backend/genkit-model/model_v6.pt` (PyTorch state dict & optimizer state)
- `backend/genkit-model/bpe_tokenizer_v6.json` (Byte-Fallback BPE merge table & vocabulary)
- `backend/genkit-model/config_v6.json` (Architecture configuration parameters)
