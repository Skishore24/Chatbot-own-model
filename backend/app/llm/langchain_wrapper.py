"""
backend/app/llm/langchain_wrapper.py
----------------------------------------------------
Production LangChain LLM Adapter for Genkit AI V6.1.
Enables seamless integration with LangChain LCEL chains, Agents,
OutputParsers, and PromptTemplates while executing on your 100%
self-hosted, locally trained PyTorch Transformer.
"""

from typing import Any, Dict, Iterator, List, Optional
from pydantic import Field

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from app.core.config import settings
from app.core.logger import logger


class GenkitLLM(LLM):
    """
    LangChain LLM adapter wrapping the Genkit AI custom PyTorch Transformer.
    Supports both batch execution and token-by-token streaming.
    """

    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=0, description="Top-k logit filtering")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus top-p sampling")
    repetition_penalty: float = Field(default=1.15, ge=1.0, description="Repetition penalty")
    max_new_tokens: int = Field(default=128, ge=1, description="Maximum new generated tokens")

    @property
    def _llm_type(self) -> str:
        return "genkit_custom_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": "Genkit Enterprise GPT v6.1",
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Executes full autoregressive text generation."""
        from app.api.chat import get_generation_engine_and_status
        from app.llm.inference import ModelStatus

        engine, status = get_generation_engine_and_status()

        # If model is not loaded, gracefully fall back to RAG knowledge synthesis
        if status != ModelStatus.READY or engine.model is None:
            logger.warning("GenkitLLM: Model not ready; delegating to RAG synthesizer.")
            from app.rag.pipeline import get_rag_pipeline
            rag = get_rag_pipeline()
            chunks, _, _ = rag.retrieve(prompt)
            return rag.synthesize_answer(prompt, chunks)

        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temp = kwargs.get("temperature", self.temperature)
        k = kwargs.get("top_k", self.top_k)
        p = kwargs.get("top_p", self.top_p)
        rep = kwargs.get("repetition_penalty", self.repetition_penalty)

        text = engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=k,
            top_p=p,
            repetition_penalty=rep,
        )

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]

        return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Incremental token streaming generator for LangChain LCEL."""
        from app.api.chat import get_generation_engine_and_status
        from app.llm.inference import ModelStatus

        engine, status = get_generation_engine_and_status()

        if status != ModelStatus.READY or engine.model is None:
            full_text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            chunk = GenerationChunk(text=full_text)
            if run_manager:
                run_manager.on_llm_new_token(full_text, chunk=chunk)
            yield chunk
            return

        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temp = kwargs.get("temperature", self.temperature)
        k = kwargs.get("top_k", self.top_k)
        p = kwargs.get("top_p", self.top_p)
        rep = kwargs.get("repetition_penalty", self.repetition_penalty)

        accumulated = ""
        for token_text in engine.generate_stream(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=k,
            top_p=p,
            repetition_penalty=rep,
        ):
            accumulated += token_text
            chunk = GenerationChunk(text=token_text)
            if run_manager:
                run_manager.on_llm_new_token(token_text, chunk=chunk)
            yield chunk

            if stop and any(s in accumulated for s in stop):
                break

