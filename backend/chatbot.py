"""
chatbot.py
----------------------------------------------------
Genkit AI - Production Chat Pipeline (v3.0)

Pipeline
--------
User Input
    ↓
Text Cleaning + Normalization + Spell Correction
    ↓
Domain Guard (Genkit-specific? Yes/No)
    ↓ (if No) → Polite refusal (no hallucination, no LLM call)
    ↓ (if Yes)
Intent Classification + Entity Extraction
    ↓
Quick Response (for greetings/farewells/thanks — no LLM needed)
    ↓ OR
Pronoun/Coreference Resolution
    ↓
Conversation Context (last N turns from memory)
    ↓
Knowledge Retrieval (TF-IDF vector search with enriched query)
    ↓
Structured Prompt Construction
    ↓
Custom GPT Transformer → Generate Response
    ↓
Post-Processing (artifact removal, cleanup)
    ↓
Save to Memory + Database
    ↓
Stream Response

Key Differences from v2:
- exact_dataset_match() REMOVED — no more FAQ retrieval
- fuzzy_dataset_match() REMOVED — no more FAQ retrieval
- LLM is ALWAYS called for Genkit questions
- Knowledge retrieval feeds CONTEXT into LLM, not the final answer
- Domain Guard is the first gate (cheap, fast)
- NLP pipeline (intent + entity) enriches retrieval query
- Structured prompt ensures model generates properly

Author : Genkit AI
"""

import os
import sys
import re
import logging
from typing import Iterator, List, Dict, Optional

# ============================================================
# PATH SETUP
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATASET_DIR,
    MAX_HISTORY_MESSAGES,
    logger,
)

# ============================================================
# AI PIPELINE IMPORTS
# ============================================================
from database import (
    update_user_profile,
    get_user_profile,
    increment_chat_count,
    save_chat_to_db,
    save_lead_to_db,
)

from ai.embeddings.embedding import (
    _store,
    save_memory,
    search_memory,
)

from ai.rag.retriever import Retriever
from ai.rag.reranker import Reranker
from ai.rag.context_builder import ContextBuilder
from ai.rag.knowledge_graph import knowledge_graph

from ai.llm.inference import generate, generate_with_confidence, is_model_loaded
from ai.llm.prompt_builder import prompt_builder

from ai.nlp.domain_guard import domain_guard
from ai.nlp.intent_classifier import intent_classifier
from ai.nlp.entity_extractor import entity_extractor
from ai.nlp.confidence_engine import confidence_engine
from ai.nlp.response_validator import response_validator

from ai.memory.conversation import conversation_memory

from ai.preprocessing.cleaner import cleaner
from ai.preprocessing.spell import spell_checker

from utils.helper import (
    clean_query,
    detect_lead,
    is_valid_query,
    format_response,
    is_greeting,
    is_goodbye,
    is_thanks,
    DEFAULT_RESPONSES,
)

# ============================================================
# COMPONENTS
# ============================================================
retriever = Retriever()
reranker = Reranker()
context_builder = ContextBuilder()

# ============================================================
# CONSTANTS
# ============================================================
CHATBOT_VERSION = "3.0.0"

_INVALID_QUERY_MSG = (
    "Please enter a valid question about Genkit. "
    "I am here to help with our services, pricing, contact, and more."
)

_MODEL_NOT_READY_MSG = (
    "Genkit AI is still warming up. The model is not trained yet. "
    "Please run the training pipeline first: python train.py"
)

_NO_CONTEXT_MSG = (
    "I am Genkit AI. I can answer questions about Genkit's services, "
    "team, pricing, contact, and more. "
    "Could you please rephrase your question?"
)

# ============================================================
# RESPONSE POST-PROCESSOR
# ============================================================

def _post_process(text: str) -> str:
    """
    Clean up raw model output and format into clear, grammatical sentences.
    """
    if not text:
        return ""

    # Remove any Reasoning prefix or leaked prompt blocks
    if "Reasoning:" in text:
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
        else:
            lines = [l for l in text.split("\n") if not l.strip().startswith("Reasoning:")]
            text = " ".join(lines).strip()

    # Remove any leaked prompt tokens
    for tag in ["[ASSISTANT]", "[USER]", "[INSTRUCTION]", "[CONTEXT]", "[HISTORY]",
                "assistant:", "user:", "instruction:", "[/INST]", "[INST]", "Fact:", "Reasoning:"]:
        text = re.sub(re.escape(tag), "", text, flags=re.IGNORECASE)

    # Remove leading punctuation/whitespace artifacts
    text = re.sub(r"^[\s\.\,\:\;\-\*]+", "", text)

    # Collapse multiple spaces and newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Split into lines/sentences and format each sentence with proper capitalization and trailing punctuation
    raw_lines = text.split("\n")
    processed_lines = []

    for line in raw_lines:
        line_str = line.strip()
        if not line_str:
            continue
        
        # Deduplicate sentences within line
        sentences = re.split(r"(?<=[.!?])\s+", line_str)
        seen = set()
        clean_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_lower = s_clean.lower()
            if s_lower not in seen:
                seen.add(s_lower)
                # Capitalize initial character
                s_formatted = s_clean[0].upper() + s_clean[1:]
                # Ensure ending punctuation if complete sentence
                # Ensure single ending punctuation if complete sentence
                s_formatted = re.sub(r"\.\.+", ".", s_formatted)
                if s_formatted[-1] not in ".!?:" and not s_formatted.startswith("-"):
                    s_formatted += "."
                clean_sentences.append(s_formatted)
        
        if clean_sentences:
            processed_lines.append(" ".join(clean_sentences))

    result = "\n\n".join(processed_lines)
    return re.sub(r"\.\.+", ".", result)


def _is_good_response(text: str) -> bool:
    """
    Check if the generated response is coherent, grammatical, and usable.
    Flags:
    - Too short (< 20 chars)
    - High ratio of <unk> tokens (> 25%)
    - Repetitive 3-gram loops (e.g., repeating phrases 3+ times)
    """
    if not text or len(text.strip()) < 20:
        return False

    unk_count = text.count("<unk>")
    words = [w.lower() for w in text.split()]
    word_count = len(words)

    if word_count > 0 and unk_count / word_count > 0.25:
        return False

    # Detect 3-gram repetition loops
    if word_count >= 12:
        from collections import Counter
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        most_common = trigram_counts.most_common(1)
        if most_common and most_common[0][1] >= 3:
            return False

    return True


# ============================================================
# KNOWLEDGE RETRIEVAL
# ============================================================

def _retrieve_knowledge(
    query: str,
    enriched_query: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Retrieve relevant documents from the knowledge base.
    Uses both the original query and the NLP-enriched query.
    """
    try:
        # Primary retrieval with enriched query (better NLP signal)
        documents = retriever.retrieve(enriched_query, top_k=top_k)

        # Re-rank with original query for relevance precision
        documents = reranker.rerank(query, documents, top_k=min(top_k, 3))

        return documents
    except Exception:
        logger.exception("Knowledge retrieval failed.")
        return []


# ============================================================
# MAIN CHAT PIPELINE
# ============================================================

def get_answer(
    query: str,
    session_id: str,
) -> Iterator[str]:
    """
    Main Genkit AI generation pipeline.

    Yields response tokens as a stream.
    """

    # --------------------------------------------------------
    # Step 1: Input Validation
    # --------------------------------------------------------
    if not query or not query.strip():
        yield _INVALID_QUERY_MSG
        return

    # --------------------------------------------------------
    # Step 2: Text Cleaning + Normalization + Spell Correction
    # --------------------------------------------------------
    try:
        cleaned = cleaner.clean(query)
        corrected = spell_checker.correct(cleaned)
        query_for_pipeline = corrected if corrected.strip() else cleaned
    except Exception:
        logger.exception("Preprocessing failed.")
        query_for_pipeline = query

    if not is_valid_query(query_for_pipeline):
        yield _INVALID_QUERY_MSG
        return

    logger.info("[Pipeline] Query: %s", query_for_pipeline[:80])

    # --------------------------------------------------------
    # Step 3: Domain Guard
    # --------------------------------------------------------
    domain_result = domain_guard.classify(query_for_pipeline)
    logger.info(
        "[DomainGuard] in_domain=%s confidence=%.2f reason=%s",
        domain_result["in_domain"],
        domain_result["confidence"],
        domain_result["reason"],
    )

    if not domain_result["in_domain"]:
        refusal = domain_guard.get_refusal_message()
        _save_interaction(
            session_id=session_id,
            question=query_for_pipeline,
            answer=refusal,
            intent="out_of_scope",
            source="domain_guard",
            confidence=domain_result["confidence"],
        )
        yield refusal
        return

    # --------------------------------------------------------
    # Step 4: Intent Classification + Entity Extraction
    # --------------------------------------------------------
    intent_result = intent_classifier.classify(query_for_pipeline)
    entities = entity_extractor.extract(query_for_pipeline)
    
    intent_label = intent_result["intent"]
    intent_confidence = intent_result["confidence"]

    # --------------------------------------------------------
    # Step 5: Knowledge Graph Expansion
    # --------------------------------------------------------
    extracted_concepts = []
    if entities.get("services"):
        extracted_concepts.extend(entities["services"])
    if entities.get("technologies"):
        extracted_concepts.extend(entities["technologies"])
        
    expanded_nodes = knowledge_graph.expand_context(extracted_concepts)
    enriched_query = entities.get("enriched_query", query_for_pipeline)
    if expanded_nodes:
        enriched_query = f"{enriched_query} " + " ".join(expanded_nodes)
        logger.info("[KnowledgeGraph] Expanded context with nodes: %s", expanded_nodes)

    logger.info(
        "[Intent] %s (%.2f) | Services: %s | Tech: %s",
        intent_label,
        intent_confidence,
        entities.get("services", [])[:2],
        entities.get("technologies", [])[:2],
    )

    # --------------------------------------------------------
    # Step 6: Quick Response (Greeting / Farewell / Thanks)
    # --------------------------------------------------------
    if intent_result["is_quick"]:
        quick = intent_result["quick_response"]
        _save_interaction(
            session_id=session_id,
            question=query_for_pipeline,
            answer=quick,
            intent=intent_label,
            source="quick_response",
            confidence=intent_confidence,
        )
        conversation_memory.add_message(session_id, "user", query_for_pipeline)
        conversation_memory.add_message(session_id, "assistant", quick)
        yield quick
        return

    # --------------------------------------------------------
    # Step 7: Coreference Resolution
    # --------------------------------------------------------
    if entities.get("has_pronouns"):
        resolved_query = entity_extractor.resolve_pronouns(
            query_for_pipeline, "Genkit"
        )
        enriched_query = entity_extractor.resolve_pronouns(
            enriched_query, "Genkit"
        )
        logger.debug("[Coreference] %s → %s", query_for_pipeline, resolved_query)
    else:
        resolved_query = query_for_pipeline

    # --------------------------------------------------------
    # Step 8: Conversation Memory & Summary
    # --------------------------------------------------------
    history: List[Dict] = []
    try:
        history = conversation_memory.recent_messages(
            session_id,
            count=MAX_HISTORY_MESSAGES,
        )
    except Exception:
        logger.exception("Conversation memory retrieval failed.")

    # --------------------------------------------------------
    # Step 9: Update User Profile & Save Long-Term Memory
    # --------------------------------------------------------
    try:
        update_user_profile(session_id=session_id, last_query=query_for_pipeline)
        increment_chat_count(session_id)
        
        # User name parsing
        if "my name is" in query_for_pipeline.lower():
            m = re.search(r"my name is (\w+)", query_for_pipeline, re.IGNORECASE)
            if m:
                conversation_memory.update_profile(session_id, {"name": m.group(1).title()})
        
        # Tech interests parsing
        if entities.get("technologies"):
            conversation_memory.update_profile(session_id, {"interests": entities["technologies"]})
            for tech in entities["technologies"]:
                conversation_memory.add_long_term_fact(session_id, f"User asked about technology: {tech}")
                
        profile = conversation_memory.get_profile(session_id)
    except Exception:
        logger.exception("User profile update failed.")
        profile = {}

    # --------------------------------------------------------
    # Step 10: Knowledge Retrieval (Hybrid BM25 + Vector + Intent filter)
    # --------------------------------------------------------
    documents = retriever.retrieve(
        query=resolved_query,
        intent=intent_label,
        top_k=5,
    )

    logger.info("[Retrieval] Found %d documents using hybrid search.", len(documents))

    # --------------------------------------------------------
    # Step 11: Build Structured Prompt
    # --------------------------------------------------------
    if documents:
        prompt = prompt_builder.build(
            query=resolved_query,
            documents=documents,
            history=history,
            intent=intent_label,
            entities=entities,
        )
    else:
        prompt = prompt_builder.build_minimal(resolved_query)

    # --------------------------------------------------------
    # Step 12: LLM Generation (with token probabilities)
    # --------------------------------------------------------
    if not is_model_loaded():
        logger.warning("[LLM] Model not loaded.")
        if documents:
            fallback = documents[0].get("text") or ""
            if "Answer:" in fallback:
                fallback = fallback.split("Answer:")[-1].strip()
            fallback = _post_process(fallback)
            if fallback:
                _save_and_yield(
                    session_id, query_for_pipeline, fallback,
                    intent_label, "retrieval_fallback", intent_confidence,
                    history,
                )
                yield fallback
                return
        yield _MODEL_NOT_READY_MSG
        return

    try:
        raw_answer, gen_confidence = generate_with_confidence(
            prompt=prompt,
            max_new_tokens=250,
        )
    except Exception:
        logger.exception("[LLM] Generation failed.")
        raw_answer = ""
        gen_confidence = 0.0

    answer = _post_process(raw_answer)

    top_doc_score = documents[0].get("score", 0.0) if documents else 0.0
    is_factual_query = any(k in query_for_pipeline.lower() for k in ["founder", "founders", "pricing", "cost", "price", "rate", "hourly", "who is", "who are", "llm", "rag", "model", "service", "services", "package", "contact", "email", "phone"])

    if not _is_good_response(answer) or (top_doc_score >= 0.60 and is_factual_query):
        logger.info("[LLM] Utilizing high-precision RAG factual document answer.")
        if documents:
            fallback_fact = documents[0].get("answer") or documents[0].get("text") or ""
            answer = _post_process(fallback_fact)
        else:
            answer = _NO_CONTEXT_MSG

    # --------------------------------------------------------
    # Step 14: Confidence Engine Validation
    # --------------------------------------------------------
    conf_scores = confidence_engine.compute(
        intent_score=intent_confidence,
        retrieved_docs=documents,
        generation_score=gen_confidence,
    )

    if conf_scores["low_confidence"]:
        logger.warning("[Confidence] Low confidence score. Triggering fallback refusal.")
        answer = conf_scores["fallback_message"]
    else:
        # ----------------------------------------------------
        # Step 15: Output Response Validator (Anti-Hallucination)
        # ----------------------------------------------------
        is_valid, validation_reason = response_validator.validate(
            generated_text=answer,
            retrieved_docs=documents,
            query=resolved_query,
        )
        if not is_valid:
            logger.warning(
                "[Validator] Output validation failed: %s. Reverting to facts.",
                validation_reason
            )
            # Revert response to verified retriever facts
            if documents:
                fallback_fact = documents[0].get("answer") or documents[0].get("text") or ""
                answer = _post_process(fallback_fact)
            else:
                answer = _NO_CONTEXT_MSG

    answer = format_response(answer, max_lines=12)

    # Save summary to conversation session memory
    try:
        summary_text = f"User asked: '{query_for_pipeline}'. Answered: '{answer[:60]}...'"
        conversation_memory.set_context_summary(session_id, summary_text)
    except Exception:
        pass

    # --------------------------------------------------------
    # Step 16: Save Memory + Database
    # --------------------------------------------------------
    _save_and_yield(
        session_id, query_for_pipeline, answer,
        intent_label, "llm_generated", intent_confidence,
        history,
    )

    # --------------------------------------------------------
    # Step 17: Lead Detection
    # --------------------------------------------------------
    try:
        if detect_lead(query_for_pipeline):
            save_lead_to_db(
                name=profile.get("name", "Visitor"),
                email=profile.get("email", "visitor@genkit.in"),
            )
    except Exception:
        logger.exception("Lead save failed.")

    # --------------------------------------------------------
    # Step 18: Stream Response
    # --------------------------------------------------------
    yield answer


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _save_interaction(
    session_id: str,
    question: str,
    answer: str,
    intent: str,
    source: str,
    confidence: float,
) -> None:
    """Save chat interaction to database (non-fatal)."""
    try:
        save_chat_to_db(
            session_id=session_id,
            question=question,
            answer=answer,
            intent=intent,
            source=source,
            confidence=confidence,
        )
    except Exception:
        logger.exception("Chat DB save failed.")


def _save_and_yield(
    session_id: str,
    question: str,
    answer: str,
    intent: str,
    source: str,
    confidence: float,
    history: List[Dict],
) -> None:
    """Save to memory and database."""
    try:
        conversation_memory.add_message(session_id, "user", question)
        conversation_memory.add_message(session_id, "assistant", answer)
    except Exception:
        logger.exception("Memory save failed.")

    try:
        save_memory(session_id, f"{question}\n{answer}")
    except Exception:
        logger.exception("Vector memory save failed.")

    _save_interaction(session_id, question, answer, intent, source, confidence)


# ============================================================
# STARTUP
# ============================================================

def initialize_chatbot() -> None:
    """
    Initialize all chatbot components.
    Safe to call multiple times (idempotent).
    """
    try:
        logger.info("=" * 60)
        logger.info("Initializing Genkit AI Chatbot v%s...", CHATBOT_VERSION)
        _store.init()
        logger.info("Knowledge Base : Ready.")
        logger.info("Domain Guard   : Ready.")
        logger.info("Intent Engine  : Ready.")
        logger.info("Entity Engine  : Ready.")
        logger.info("Prompt Builder : Ready.")
        logger.info("=" * 60)
    except Exception:
        logger.exception("Chatbot initialization failed.")


# ============================================================
# HEALTH CHECK
# ============================================================

def chatbot_health() -> Dict:
    """Return health status of all pipeline components."""
    from ai.llm.inference import MODEL_READY
    return {
        "version": CHATBOT_VERSION,
        "model_loaded": MODEL_READY,
        "vector_store": _store._initialized,
        "retriever": retriever is not None,
        "reranker": reranker is not None,
        "domain_guard": domain_guard is not None,
        "intent_classifier": intent_classifier is not None,
        "entity_extractor": entity_extractor is not None,
        "prompt_builder": prompt_builder is not None,
        "memory_sessions": len(conversation_memory),
    }


# ============================================================
# CLEAR MEMORY
# ============================================================

def clear_memory(session_id: str) -> None:
    """Clear conversation memory for a session."""
    try:
        conversation_memory.clear_session(session_id)
        _store.clear_memory(session_id)
    except Exception:
        logger.exception("Memory clear failed.")


# ============================================================
# AUTO INITIALIZATION
# ============================================================

try:
    initialize_chatbot()
except Exception:
    logger.exception("Automatic chatbot initialization failed.")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "get_answer",
    "initialize_chatbot",
    "chatbot_health",
    "clear_memory",
    "CHATBOT_VERSION",
]
