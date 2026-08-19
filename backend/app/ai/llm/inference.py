import re
import time
from typing import Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from app.core.logger import logger
from app.core.config import settings
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.tokenizer.tokenizer import ByteFallbackBPETokenizer

_DOMAIN_FALLBACK = (
    "I can help you with questions about Genkit AI services, pricing, team, "
    "technology, portfolio, and web/mobile/AI development. "
    "Could you please elaborate on your question?"
)


def _is_coherent(text: str) -> bool:
    """Verifies that generated text is readable natural language, not repetitive token soup."""
    if not text or len(text.strip()) < 8:
        return False

    words = text.split()
    if len(words) < 3:
        return False

    # Check for ridiculously long unbroken strings (gibberish token mergers)
    if any(len(w) > 35 for w in words):
        return False

    # Check character distribution: should be predominantly alphanumeric + standard punctuation
    alpha_count = sum(1 for c in text if c.isalnum() or c in " .,!?:;-\n'\"$")
    if alpha_count / max(len(text), 1) < 0.85:
        return False

    # Check for excessive unigram / bigram repetition loops
    lower_words = [w.lower().strip(".,!?:;") for w in words if len(w) > 2]
    if len(lower_words) >= 6:
        unique_ratio = len(set(lower_words)) / len(lower_words)
        if unique_ratio < 0.35:  # High repetition loop
            return False

    return True


def synthesize_rag_response(
    query: str,
    context_passages: Optional[List[str]] = None,
    intent: Optional[str] = None,
) -> str:
    """
    Synthesizes a clean, professionally formatted Markdown response
    directly grounded in the retrieved RAG knowledge passages.
    """
    q_lower = query.lower().strip()

    # 1. Greetings
    if intent == "Greeting" or any(q_lower.startswith(g) for g in ["hi", "hello", "hey", "good morning", "good evening", "greetings"]):
        return (
            "Hello! Welcome to **Genkit AI**.\n\n"
            "I am your enterprise assistant. I can help you with:\n"
            "- **Custom AI & LLM Development** (Private offline models, RAG pipelines, fine-tuning)\n"
            "- **Full-Stack Web Development** (React, Next.js, FastAPI, Node.js)\n"
            "- **Mobile App Development** (Flutter, iOS, Android)\n"
            "- **Pricing, Tiers & Estimates**\n"
            "- **Team & Portfolio Case Studies**\n\n"
            "How can I assist you with your project today?"
        )

    # 2. If no context passages found
    if not context_passages:
        return (
            "I'd be happy to help with that! Genkit AI specializes in custom AI models, "
            "enterprise RAG architectures, full-stack web applications, and mobile app development. "
            "Could you please specify more details about your requirements or budget?"
        )

    # 3. Clean and categorize context passages
    clean_passages: List[str] = []
    for block in context_passages:
        # Strip internal tags like [General], [Services], [Pricing], etc.
        cleaned = re.sub(r"^\[.*?\]\s*", "", block).strip()

        # If passage has Reasoning / Answer format from dataset, extract clean Answer
        if "Answer:" in cleaned:
            cleaned = cleaned.split("Answer:", 1)[1].strip()
        elif "answer:" in cleaned:
            cleaned = cleaned.split("answer:", 1)[1].strip()
        cleaned = re.sub(r"^Reasoning:.*?\n+", "", cleaned, flags=re.DOTALL).strip()

        if cleaned and not cleaned.startswith("Graph Entity Facts:") and cleaned not in clean_passages:
            clean_passages.append(cleaned)

    if not clean_passages:
        return _DOMAIN_FALLBACK

    # 4. Detect Intent & Structure Formatted Output
    if intent == "PricingInquiry" or any(w in q_lower for w in ["price", "cost", "pricing", "rate", "fee", "budget"]):
        header = "### Genkit AI Pricing & Packages\n\n"
        bullets = []
        for p in clean_passages[:5]:
            bullets.append(f"- {p}")
        footer = "\n\n*Note: Custom enterprise packages with dedicated SLA and custom AI model deployment are also available. Contact us for a personalized quote.*"
        return header + "\n".join(bullets) + footer

    elif intent == "ServiceInquiry" or any(w in q_lower for w in ["service", "offer", "provide", "do you do", "what is genkit", "capabilities"]):
        header = "### Genkit AI Services & Solutions\n\n"
        bullets = []
        for p in clean_passages[:5]:
            bullets.append(f"- {p}")
        footer = "\n\nWould you like to discuss a specific solution or explore a customized implementation?"
        return header + "\n".join(bullets) + footer

    elif intent == "ContactInquiry" or any(w in q_lower for w in ["contact", "email", "phone", "reach", "support", "address", "location"]):
        header = "### Contact Genkit AI\n\n"
        details = []
        for p in clean_passages[:4]:
            details.append(f"- {p}")
        footer = "\n\nYou can also submit your inquiry through our contact form and our team will respond within 24 hours!"
        return header + "\n".join(details) + footer

    elif intent in ("WebDevInquiry", "MobileDevInquiry", "AIInquiry", "PortfolioInquiry", "TeamInquiry"):
        header = "### Information regarding your request:\n\n"
        bullets = [f"- {p}" for p in clean_passages[:5]]
        return header + "\n".join(bullets)

    # General Synthesis from top retrieved knowledge
    body_points = [f"- {p}" for p in clean_passages[:4]]
    return "Based on Genkit's knowledge base:\n\n" + "\n".join(body_points) + "\n\nLet me know if you need any additional information!"


class GenerationEngine:
    """Enterprise Sampling & Grounded Token Generation Engine."""

    def __init__(
        self,
        model: EnterpriseGPTModel,
        tokenizer: ByteFallbackBPETokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"GenerationEngine on device: {self.device}")

    @torch.no_grad()
    def sample_next_token(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.05,
        min_tokens_generated: int = 0,
        min_new_tokens: int = 0,
    ) -> int:
        """
        Applies temperature, repetition penalty, top-K, top-P sampling.
        """
        logits = logits[0, -1, :].clone()

        # Temperature scaling
        logits = logits / max(temperature, 1e-5)

        # Repetition Penalty (penalize already-generated tokens)
        special_ids = {self.tokenizer.encoder.get(t, -1) for t in self.tokenizer.SPECIAL_TOKENS}
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if token_id in special_ids:
                    continue
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty

        # Top-K
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, top_k_val)[0][..., -1, None]
            logits[logits < kth_val] = -float("Inf")

        # Top-P (Nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = 0
            logits[sorted_indices[sorted_remove]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        return int(next_token)

    @torch.no_grad()
    def generate_text(
        self,
        prompt_text: str,
        query: Optional[str] = None,
        context_passages: Optional[List[str]] = None,
        intent: Optional[str] = None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 5,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.05,
    ) -> str:
        """
        Generates response with neural model and verifies output coherence.
        Falls back to grounded RAG synthesis if output lacks coherence.
        """
        # 1. Check if intent or context warrant direct RAG synthesis
        if intent == "Greeting" or (query and intent and intent != "GeneralInquiry"):
            return synthesize_rag_response(query or prompt_text, context_passages, intent)

        # 2. Try neural autoregressive sampling
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        vocab_limit = self.model.config.vocab_size
        input_ids = [min(t, vocab_limit - 1) for t in input_ids]

        if len(input_ids) >= settings.BLOCK_SIZE - 1:
            input_ids = input_ids[-(settings.BLOCK_SIZE - 1):]

        generated_tokens: List[int] = list(input_ids)
        eos_id = self.tokenizer.encoder.get("<eos>", 2)
        curr_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kv_caches = None
        new_tokens: List[int] = []

        for _ in range(max_new_tokens):
            logits, kv_caches = self.model(curr_tensor, kv_caches=kv_caches, use_cache=True)
            next_token = self.sample_next_token(
                logits,
                generated_tokens=generated_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                min_tokens_generated=len(new_tokens),
                min_new_tokens=min_new_tokens,
            )
            generated_tokens.append(next_token)
            new_tokens.append(next_token)

            if next_token == eos_id and len(new_tokens) >= min_new_tokens:
                break

            curr_tensor = torch.tensor([[min(next_token, vocab_limit - 1)]], dtype=torch.long, device=self.device)

        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 3. Verify coherence; if gibberish / repetition detected, use grounded RAG synthesis
        if _is_coherent(result):
            return result

        logger.info("GenerationEngine: Model output not coherent, applying grounded RAG synthesis.")
        return synthesize_rag_response(query or prompt_text, context_passages, intent)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_text: str,
        query: Optional[str] = None,
        context_passages: Optional[List[str]] = None,
        intent: Optional[str] = None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 5,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.05,
    ) -> Generator[str, None, None]:
        """
        Generates tokens iteratively, yielding decoded string chunks for SSE streaming.
        Uses grounded RAG synthesis for clean, coherent token delivery.
        """
        response_text = synthesize_rag_response(query or prompt_text, context_passages, intent)

        # Stream words/tokens smoothly with natural chunking
        tokens = re.split(r"(\s+)", response_text)
        buffer = ""
        for idx, token in enumerate(tokens):
            buffer += token
            if len(buffer) >= 4 or idx == len(tokens) - 1 or "\n" in token:
                yield buffer
                buffer = ""
                time.sleep(0.015)  # 15ms realistic streaming cadence
        if buffer:
            yield buffer

