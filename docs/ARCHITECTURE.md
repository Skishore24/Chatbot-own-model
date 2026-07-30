# 🏛️ GENKIT AI — Clean Architecture Specification

## SOLID Architectural Principles

GENKIT AI v5.0 Enterprise is designed using **Clean Architecture** principles separating domain, presentation, AI computation, and infrastructure layers.

### 1. Single Responsibility Principle (SRP)
Every module has a single, well-defined purpose:
- `app/security/sanitizer.py`: Performs input XSS and SQLi escaping.
- `app/security/injection.py`: Detects prompt injection attack patterns.
- `app/ai/llm/attention.py`: Implements Grouped-Query Attention.

### 2. Dependency Inversion Principle (DIP)
High-level domain services depend on abstract interfaces, not concrete implementations. Repository patterns decouple business logic from MySQL database drivers.

### 3. Open/Closed Principle (OCP)
The retrieval pipeline is structured as an extensible multi-stage chain where new rerankers or graph search passes can be plugged in without modifying existing retrieval classes.

---

## Layer Definitions

```text
[ Presentation Layer ]  -> FastAPI Routers, SSE Streaming Formatters, Pydantic DTOs
[ Application Layer ]   -> Domain Guard, Coreference Resolver, Prompt Builder
[ AI Subsystems ]       -> PyTorch GPT Engine, Hybrid RAG, Byte-Fallback Tokenizer
[ Infrastructure Layer] -> Async MySQL Pool, Pydantic Config, Structured Logger
```
