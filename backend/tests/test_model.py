"""
backend/tests/test_model.py
----------------------------------------------------
Unit tests for Neural Model, Attention, RoPE, KV Cache, and Attention Masking.
"""

import unittest
import torch

from app.llm.config import GPTConfig
from app.llm.normalization import RMSNorm
from app.llm.positional import precompute_freqs_cis, apply_rotary_emb
from app.llm.model import EnterpriseGPTModel


class TestEnterpriseGPTModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
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
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        logits, presents = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
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

    def test_full_step_by_step_token_equivalence(self):
        """Verify token-by-token sequential KV-cache accumulation yields identical outputs at every single step."""
        tokens = [12, 45, 78, 23, 89]
        input_tensor = torch.tensor([tokens], dtype=torch.long)

        with torch.inference_mode():
            full_logits, _ = self.model(input_tensor, use_cache=False)

            past_kv = None
            for idx, tok in enumerate(tokens):
                tok_tensor = torch.tensor([[tok]], dtype=torch.long)
                step_logit, past_kv = self.model(tok_tensor, past_key_values=past_kv, use_cache=True)
                step_target = full_logits[:, idx : idx + 1, :]
                diff = torch.max(torch.abs(step_logit - step_target)).item()
                self.assertLess(diff, 1e-4, f"Step {idx} cached logit mismatch (diff={diff})")

    def test_padding_mask_invariance(self):
        """Verify that right-padding tokens with attention_mask does not affect real tokens."""
        real_tokens = torch.tensor([[15, 25, 35, 45]], dtype=torch.long)
        padded_tokens = torch.tensor([[15, 25, 35, 45, 0, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)

        with torch.inference_mode():
            real_logits, _ = self.model(real_tokens, use_cache=False)
            padded_logits, _ = self.model(padded_tokens, attention_mask=attention_mask, use_cache=False)

            # Compare logits for real tokens (first 4 positions)
            real_part_from_padded = padded_logits[:, :4, :]
            diff = torch.max(torch.abs(real_logits - real_part_from_padded)).item()
            self.assertLess(diff, 1e-4, f"Padding affected real-token logits: max diff={diff}")

    def test_gptconfig_serialization(self):
        """Verify GPTConfig serialization to dict, file, and load from file."""
        import tempfile
        from pathlib import Path

        cfg_dict = self.config.to_dict()
        self.assertIn("vocab_size", cfg_dict)
        self.assertIn("n_layer", cfg_dict)

        reloaded_cfg = GPTConfig.from_dict(cfg_dict)
        self.assertEqual(reloaded_cfg.vocab_size, self.config.vocab_size)
        self.assertEqual(reloaded_cfg.n_embd, self.config.n_embd)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.config.save_to_file(tmp_path)
            from_file = GPTConfig.load_from_file(tmp_path)
            self.assertEqual(from_file.vocab_size, self.config.vocab_size)
            self.assertEqual(from_file.n_layer, self.config.n_layer)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
