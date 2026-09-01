# Genkit Custom LLM Specification & Training Guide

## Model Architecture Details

| Parameter | Value | Description |
|---|---|---|
| Architecture | Decoder-Only Transformer | Enterprise GPT with RoPE & GQA |
| Parameters | ~75M – 85M | Optimized for high-throughput enterprise edge / GPU serving |
| Vocabulary Size | 2,084 tokens | Deterministic Byte-Fallback BPE Tokenizer (`bpe_tokenizer_v5.json`) |
| Sequence Length | 512 tokens | Block context window with Rotary Positional Embeddings |
| Hidden Dimension (`n_embd`) | 384 | Hidden dimension size ($d_{model}$) |
| Transformer Layers (`n_layer`) | 6 | Decoder blocks |
| Attention Heads (`n_head`) | 6 | Query Attention Heads ($H_Q$) |
| KV Attention Heads (`n_kv_head`) | 2 | Key-Value Attention Heads for Grouped Query Attention (GQA) |
| Normalization | RMSNorm | Root Mean Square Layer Normalization ($\epsilon = 10^{-5}$) |
| Attention Mechanism | Flash / Causal Masked | KV-Cache accelerated with padding token invariance |

---

## Tokenizer Architecture

The tokenizer implements deterministic **Byte-Fallback BPE (Byte-Pair Encoding)**:
- 12 reserved control tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`, `<context_start>`, `<context_end>`, `<query_start>`, `<query_end>`, `<thought_start>`, `<thought_end>`, `<ans_start>`, `<ans_end>`).
- 256 individual byte fallback tokens (`<0x00>` to `<0xFF>`) ensuring 100% loss-free representation of unknown Unicode characters, accents, emojis, and symbols.
- Learned subword merges across verified domain corpus.

---

## Training Pipeline

To train or fine-tune the model:

```powershell
cd backend
python train.py --epochs 30 --batch-size 4 --accum-steps 8
```

### Options:
- `--epochs`: Number of training epochs (default: 60)
- `--batch-size`: Micro-batch size (default: 4)
- `--accum-steps`: Gradient accumulation steps (default: 8, effective batch size = 32)
- `--lr`: Peak learning rate (default: `3e-4`)
- `--device`: Compute device (`cuda` or `cpu`)
- `--retrain-tokenizer`: Force retraining BPE tokenizer from scratch

---

## Checkpoint Verification

To verify the integrity and compatibility of any checkpoint file:

```powershell
python scripts/verify_checkpoint.py --path genkit-model/model_v6.pt
```
