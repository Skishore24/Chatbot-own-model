"""
backend/app/ai/llm/inference.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Generation Engine
Handles sampling, Top-K, Top-P, Temperature, Repetition Penalty & Streaming Token Generation.
"""

from typing import Generator, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from app.core.logger import logger
from app.core.config import settings
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.tokenizer.tokenizer import ByteFallbackBPETokenizer


class GenerationEngine:
    """Enterprise Sampling & Token Generation Engine."""

    def __init__(self, model: EnterpriseGPTModel, tokenizer: ByteFallbackBPETokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def sample_next_token(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.05,
    ) -> int:
        """
        Applies temperature scaling, repetition penalty, top-K, and top-P sampling to select the next token.
        """
        # Take logits for the last token position: (1, VocabSize)
        logits = logits[0, -1, :] / max(temperature, 1e-5)

        # Apply Repetition Penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty

        # Apply Top-K Truncation
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        # Apply Top-P (Nucleus) Truncation
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("Inf")

        # Sample from Multinomial Distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token

    @torch.no_grad()
    def generate_text(
        self,
        prompt_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.05,
    ) -> str:
        """
        Generates full completed text response synchronously.
        """
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        generated_tokens = list(input_ids)
        eos_id = self.tokenizer.encoder.get("<eos>", 2)

        curr_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kv_caches = None

        for _ in range(max_new_tokens):
            logits, kv_caches = self.model(curr_tensor, kv_caches=kv_caches, use_cache=True)
            next_token = self.sample_next_token(
                logits,
                generated_tokens=generated_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            generated_tokens.append(next_token)
            if next_token == eos_id:
                break

            curr_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        return self.tokenizer.decode(generated_tokens[len(input_ids):], skip_special_tokens=True)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.05,
    ) -> Generator[str, None, None]:
        """
        Generates tokens iteratively, yielding decoded string chunks for real-time SSE streaming.
        """
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        generated_tokens = list(input_ids)
        eos_id = self.tokenizer.encoder.get("<eos>", 2)

        curr_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kv_caches = None

        for _ in range(max_new_tokens):
            logits, kv_caches = self.model(curr_tensor, kv_caches=kv_caches, use_cache=True)
            next_token = self.sample_next_token(
                logits,
                generated_tokens=generated_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            generated_tokens.append(next_token)
            if next_token == eos_id:
                break

            token_chunk = self.tokenizer.decode([next_token], skip_special_tokens=True)
            if token_chunk:
                yield token_chunk

            curr_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
