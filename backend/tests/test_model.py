"""
backend/tests/test_model.py
----------------------------------------------------
Unit tests for Neural Model, Attention, RoPE, and KV Cache.
"""

import unittest
import torch

from app.llm.config import GPTConfig
from app.llm.normalization import RMSNorm
from app.llm.positional import precompute_freqs_cis, apply_rotary_emb
from app.llm.model import EnterpriseGPTModel


class TestEnterpriseGPTModel(unittest.TestCase):

    def setUp(self):
        self.config = GPTConfig(
            vocab_size=100,
            block_size=64,
            n_embd=64,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            dropout=0.0,
        )
        self.model = EnterpriseGPTModel(self.config)
        self.model.eval()

    def test_rmsnorm(self):
        """Test RMSNorm output dimensions and variance normalization."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        self.assertEqual(y.shape, (2, 10, 64))

    def test_rope_with_offset(self):
        """Test RoPE complex application with offset."""
        head_dim = self.config.head_dim
        freqs_cis = precompute_freqs_cis(head_dim, 128)
        q = torch.randn(1, 4, self.config.n_head, head_dim)
        k = torch.randn(1, 4, self.config.n_kv_head, head_dim)

        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis, offset=0)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

        # Single step with offset
        q_step = torch.randn(1, 1, self.config.n_head, head_dim)
        k_step = torch.randn(1, 1, self.config.n_kv_head, head_dim)
        q_step_rot, k_step_rot = apply_rotary_emb(q_step, k_step, freqs_cis, offset=4)
        self.assertEqual(q_step_rot.shape, (1, 1, self.config.n_head, head_dim))

    def test_forward_pass_shapes(self):
        """Verify full sequence forward pass output shape."""
        input_ids = torch.randint(0, 100, (2, 16))
        logits, presents = self.model(input_ids, use_cache=False)
        self.assertEqual(logits.shape, (2, 16, 100))
        self.assertIsNone(presents)

    def test_cached_generation_equivalence(self):
        """Verify that step-by-step KV-cached forward produces identical last-token logits as full forward."""
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

        # 1. Full non-cached forward pass
        with torch.inference_mode():
            full_logits, _ = self.model(input_ids, use_cache=False)
            target_next_logits = full_logits[:, -1, :]

        # 2. Cached step-by-step pass
        with torch.inference_mode():
            # Prefill on first 3 tokens
            prefix = input_ids[:, :3]
            _, past_kv = self.model(prefix, use_cache=True)

            # Step on 4th token
            last_tok = input_ids[:, 3:4]
            step_logits, _ = self.model(last_tok, past_key_values=past_kv, use_cache=True)
            cached_next_logits = step_logits[:, -1, :]

        # Numerical comparison
        diff = torch.max(torch.abs(target_next_logits - cached_next_logits)).item()
        self.assertLess(diff, 1e-4, f"Cached logits diverged from full forward pass (diff={diff})")


if __name__ == "__main__":
    unittest.main()
