"""
backend/tests/test_config_validation.py
----------------------------------------------------
Unit tests for GPTConfig boundary conditions, parameter validation, and invalid configuration rejection.
"""

import unittest
from app.llm.config import GPTConfig


class TestConfigValidation(unittest.TestCase):

    def test_valid_default_config(self):
        """Test default configuration is valid."""
        cfg = GPTConfig()
        cfg.validate()  # Should not raise
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.intermediate_dim, 1024)

    def test_invalid_embed_dim_not_divisible_by_heads(self):
        """Test that n_embd not divisible by n_head raises ValueError."""
        cfg = GPTConfig(n_embd=385, n_head=6)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_heads_not_divisible_by_kv_heads(self):
        """Test that n_head not divisible by n_kv_head raises ValueError for GQA."""
        cfg = GPTConfig(n_head=7, n_kv_head=2)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_negative_parameters(self):
        """Test that negative or zero parameters raise ValueError."""
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=0).validate()

        with self.assertRaises(ValueError):
            GPTConfig(block_size=-10).validate()

        with self.assertRaises(ValueError):
            GPTConfig(n_layer=0).validate()

    def test_invalid_special_token_ids(self):
        """Test that token IDs outside vocabulary range raise ValueError."""
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=100, pad_token_id=105).validate()

        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=100, eos_token_id=-1).validate()


if __name__ == "__main__":
    unittest.main()
