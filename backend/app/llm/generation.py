"""
backend/app/llm/generation.py
----------------------------------------------------
Autoregressive Generation Engine for Genkit AI V6.
- Greedy & Temperature Sampling with Top-K, Top-P, and Repetition Penalty
- Real incremental token-by-token streaming generator
- Cache-aware attention acceleration
"""

from typing import Generator, List, Optional
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.llm.model import EnterpriseGPTModel
from app.llm.tokenizer import ByteFallbackBPETokenizer


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filters logits using Top-K and Nucleus (Top-P) sampling."""
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


class GenerationEngine:
    """Autoregressive text generation and streaming generator."""

    def __init__(self, model: Optional[EnterpriseGPTModel], tokenizer: ByteFallbackBPETokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device if model is not None else torch.device("cpu")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Generates a complete response string for a prompt."""
        if self.model is None:
            return ""
        chunks = list(
            self.generate_stream(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        )
        return "".join(chunks)

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Real incremental token-by-token streaming generator.
        Decodes each generated token and yields clean chunks.
        """
        if self.model is None:
            return

        max_tokens = max_new_tokens or settings.MAX_NEW_TOKENS
        temp = temperature if temperature is not None else settings.TEMPERATURE
        k = top_k if top_k is not None else settings.TOP_K
        p = top_p if top_p is not None else settings.TOP_P
        rep_penalty = repetition_penalty or settings.REPETITION_PENALTY

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not input_ids:
            return

        # Truncate prompt if longer than model block_size - max_tokens
        max_prompt_len = self.model.config.block_size - max_tokens - 4
        if len(input_ids) > max_prompt_len:
            input_ids = input_ids[-max_prompt_len:]

        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = (x != self.tokenizer.pad_id).long()
        generated_ids: List[int] = list(input_ids)
        past_key_values = None
        stream_decoder = self.tokenizer.create_stream_decoder(skip_special_tokens=True)

        # Pre-fill KV-Cache on prompt
        logits, past_key_values = self.model(
            x,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        next_token_logits = logits[:, -1, :]

        for _ in range(max_tokens):
            # Apply Repetition Penalty
            if rep_penalty != 1.0:
                for token_id in set(generated_ids):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= rep_penalty
                    else:
                        next_token_logits[0, token_id] *= rep_penalty

            # Sampling or Greedy selection
            if temp <= 0.01:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(next_token_logits / temp, top_k=k, top_p=p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()

            # Stop on EOS or PAD
            if token_id == self.tokenizer.eos_id or token_id == self.tokenizer.pad_id:
                break

            generated_ids.append(token_id)

            # Decode single token chunk with stream decoder buffer
            chunk_text = stream_decoder.put(token_id)
            if chunk_text:
                yield chunk_text

            # Single token forward step with KV-Cache
            logits, past_key_values = self.model(
                next_token,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = logits[:, -1, :]

        # Flush any remaining bytes from stream decoder buffer
        remaining_chunk = stream_decoder.flush()
        if remaining_chunk:
            yield remaining_chunk

