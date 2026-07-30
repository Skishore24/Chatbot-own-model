"""
backend/tests/test_gqa_model.py
----------------------------------------------------
Unit tests for EnterpriseGPTModel, GenerationEngine, and PromptBuilder.
"""

import sys
import unittest
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.llm.inference import GenerationEngine
from app.ai.llm.prompt_builder import prompt_builder
from app.ai.tokenizer.tokenizer import ByteFallbackBPETokenizer


class TestEnterpriseGPTModel(unittest.TestCase):

    def setUp(self):
        self.config = GPTConfig(
            vocab_size=1000,
            block_size=128,
            n_embd=128,
            n_head=4,
            n_kv_head=2,  # GQA ratio 2:1
            n_layer=2,
        )
        self.model = EnterpriseGPTModel(self.config)
        self.tokenizer = ByteFallbackBPETokenizer()

    def test_forward_pass(self):
        """Verify model forward pass shape."""
        x = torch.randint(0, 1000, (2, 16))  # Batch=2, SeqLen=16
        logits, kv_caches = self.model(x, use_cache=True)

        self.assertEqual(logits.shape, (2, 16, 1000))
        self.assertIsNotNone(kv_caches)
        self.assertEqual(len(kv_caches), 2)  # 2 layers

    def test_generation_engine(self):
        """Verify text generation engine output."""
        engine = GenerationEngine(self.model, self.tokenizer, device="cpu")
        prompt = "Hello Genkit AI"
        out_text = engine.generate_text(prompt, max_new_tokens=10)
        self.assertIsInstance(out_text, str)

    def test_prompt_builder(self):
        """Verify prompt builder control tokens."""
        prompt = prompt_builder.build_prompt(
            query="What are Genkit AI services?",
            context_passages=["Genkit provides AI and web development services."],
            history=[{"role": "user", "text": "Hi"}, {"role": "assistant", "text": "Hello!"}],
        )
        self.assertIn("<context_start>", prompt)
        self.assertIn("<query_start>", prompt)
        self.assertIn("<ans_start>", prompt)


if __name__ == "__main__":
    unittest.main()
