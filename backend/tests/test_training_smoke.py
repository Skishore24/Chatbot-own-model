"""
backend/tests/test_training_smoke.py
----------------------------------------------------
Real-data pre-training smoke test:
1 real sample -> tokenizer -> model -> forward -> loss -> backward -> optimizer step.
"""

import unittest
import torch
import torch.nn as nn

from app.core.config import settings
from app.llm.config import GPTConfig
from app.llm.model import EnterpriseGPTModel
from app.llm.tokenizer import ByteFallbackBPETokenizer
from training.prepare import build_instruction_corpus


class TestTrainingSmokeTest(unittest.TestCase):

    def test_real_data_training_step(self):
        """Validates real dataset sample through forward, loss, backward, and optimizer step."""
        corpus = build_instruction_corpus()
        self.assertTrue(len(corpus) > 0, "Instruction corpus should not be empty")

        real_sample = corpus[0]
        self.assertTrue(isinstance(real_sample, str) and len(real_sample) > 5)

        # Build tokenizer with small sample
        tokenizer = ByteFallbackBPETokenizer(vocab_size=256)
        tokenizer.train_on_corpus([real_sample], target_vocab_size=256)

        encoded = tokenizer.encode(real_sample, add_special_tokens=True)
        self.assertGreater(len(encoded), 1)

        block_size = 64
        if len(encoded) > block_size + 1:
            seq = encoded[: block_size + 1]
        else:
            seq = encoded + [tokenizer.pad_id] * (block_size + 1 - len(encoded))

        x = torch.tensor([seq[:-1]], dtype=torch.long)
        y = torch.tensor([seq[1:]], dtype=torch.long)
        mask = (x != tokenizer.pad_id).long()

        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            n_embd=64,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            dropout=0.0,
            pad_token_id=tokenizer.pad_id,
        )

        model = EnterpriseGPTModel(config)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        optimizer.zero_grad()
        logits, _ = model(x, attention_mask=mask)

        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        self.assertFalse(torch.isnan(loss), "Loss computed was NaN")
        self.assertFalse(torch.isinf(loss), "Loss computed was Inf")
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
