"""
backend/tests/test_kv_cache.py
----------------------------------------------------
Mandatory KV Cache Numerical Precision Tests.
Validates that token-by-token cached generation exactly matches
full-sequence non-cached forward pass across arbitrary sequences and lengths.
"""

import unittest
import torch

from app.llm.config import GPTConfig
from app.llm.model import EnterpriseGPTModel


class TestKVCacheNumericalEquivalence(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = GPTConfig(
            vocab_size=128,
            block_size=128,
            n_embd=64,
            n_layer=3,
            n_head=4,
            n_kv_head=2,
            dropout=0.0,
        )
        self.model = EnterpriseGPTModel(self.config)
        self.model.eval()

    def test_single_token_prefill_and_step(self):
        """Test [A, B, C, D] where prefix [A, B, C] is prefilled and D is stepped."""
        seq = torch.tensor([[10, 25, 42, 73]], dtype=torch.long)

        with torch.inference_mode():
            full_out, _ = self.model(seq, use_cache=False)
            expected_d_logits = full_out[:, 3, :]

            # Prefill on [A, B, C]
            _, past_kv = self.model(seq[:, :3], use_cache=True)

            # Step on [D]
            step_out, _ = self.model(seq[:, 3:4], past_key_values=past_kv, use_cache=True)
            actual_d_logits = step_out[:, 0, :]

        diff = torch.max(torch.abs(expected_d_logits - actual_d_logits)).item()
        self.assertLess(diff, 1e-4, f"KV cache stepped token diverged: diff={diff}")

    def test_progressive_step_by_step_cache(self):
        """Test A -> B -> C -> D -> E incrementally and verify all token logits."""
        tokens = [5, 18, 33, 62, 89, 104, 115]
        full_seq = torch.tensor([tokens], dtype=torch.long)

        with torch.inference_mode():
            full_logits, _ = self.model(full_seq, use_cache=False)

            past_kv = None
            for idx, tok in enumerate(tokens):
                step_tok = torch.tensor([[tok]], dtype=torch.long)
                step_logits, past_kv = self.model(step_tok, past_key_values=past_kv, use_cache=True)

                target_logit = full_logits[:, idx : idx + 1, :]
                diff = torch.max(torch.abs(step_logits - target_logit)).item()
                self.assertLess(diff, 1e-4, f"Token index {idx} mismatch: diff={diff}")


if __name__ == "__main__":
    unittest.main()
