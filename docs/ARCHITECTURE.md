# Genkit AI Architecture Specification (v6.1)

## High-Level System Diagram

```
User / Client Application
        │ (HTTPS / SSE Stream)
        ▼
   Frontend (React + Vite SPA / Nginx)
        │ (REST API / SSE Events)
        ▼
   FastAPI Gateway (Uvicorn / Middleware)
   ├── Rate Limiting & Input Sanitization Guard
   ├── Prompt Injection Scanner
   └── Session Management
        │
        ▼
   Chat Orchestrator & Execution Router
        ├── 1. Hybrid RAG Pipeline (Deterministic)
        │     ├── Inverted Index Lookup
        │     ├── BM25 Keyword Scoring
        │     ├── TF-IDF Vector Scoring
        │     ├── Score Normalization & Fusion Reranking
        │     └── Grounding Confidence Evaluator
        │
        ├── 2. Decision Gate
        │     ├── [Low Confidence / Off-Topic] ──► Out-of-Domain Refusal (`response_mode="system"`)
        │     ├── [Model Not Loaded / Degradation] ──► Verified Knowledge Direct (`response_mode="rag_direct"`)
        │     └── [High Confidence & Model Ready] ──► Custom PyTorch LLM (`response_mode="llm_rag"`)
        │
        └── 3. Generation Engine & Inference Runtime
              ├── Byte-Fallback BPE Tokenizer (Vocab: 2,084)
              ├── EnterpriseGPTModel (RoPE, GQA, RMSNorm, KV-Cache)
              ├── Autoregressive Sampling (Top-K, Nucleus Top-P, Repetition Penalty)
              └── StreamDecoder (UTF-8 Multi-byte Safe Buffer)
        │
        ▼
   Persistent Storage (MySQL 8.0 / SQLite Dual Fallback)
   ├── `chat_sessions` (Session metadata)
   ├── `chat_messages` (Audit log, intent, confidence, latency)
   ├── `leads` (Business inquiries & contact form)
   └── `feedback` (User satisfaction ratings)
```

---

## Model Lifecycle & Checkpoint Safety

```
Dataset Preparation (datasets/)
        │
        ▼
Byte-Fallback BPE Tokenizer (Vocab: 2,084)
        │
        ▼
GPU-Accelerated Trainer (CUDA + AMP + Cosine Warmup)
        │
        ├── Periodic Validation on Held-Out Split (Loss & Perplexity)
        │
        ▼
CheckpointManager (`app/llm/checkpoint.py`)
        │
        ├── 1. Write payload to `model_v6.pt.tmp_<timestamp>`
        ├── 2. Immediate `torch.load()` inspection & state_dict NaN/Inf scan
        ├── 3. Architecture & vocab configuration verification
        ├── 4. Backup existing checkpoint to `model_v6.pt.bak`
        └── 5. Atomic OS file replacement to `model_v6.pt`
```

---

## Execution Modes

| Response Mode | Condition | Behavior |
|---|---|---|
| `system` | Query is out-of-domain (low confidence score < 0.25) | Emits formal domain scope refusal without hallucination. |
| `llm_rag` | Query in-domain AND model state is `MODEL_READY` | Custom LLM generates autoregressive response conditioned on RAG context. |
| `rag_direct` | Query in-domain BUT model state is `MODEL_NOT_FOUND` / `MODEL_INVALID` | Dynamically synthesizes authoritative answer directly from knowledge chunks. |

---

## Directory Layout

```
Chatbot-own-model/
├── README.md                     # Master project documentation
├── ARCHITECTURE.md               # Architectural overview
├── docker-compose.yml            # Standard multi-container orchestration
├── docker-compose.dev.yml        # Development configuration with live reload
├── docker-compose.prod.yml       # Production hardened deployment configuration
│
├── backend/
│   ├── app/
│   │   ├── api/                  # FastAPI routers (chat, streaming, health, leads, feedback)
│   │   ├── core/                 # App configuration, security guard, structured logging
│   │   ├── database/             # Dual MySQL/SQLite connection manager & repositories
│   │   ├── llm/                  # PyTorch model, BPE tokenizer, CheckpointManager, inference engine
│   │   ├── rag/                  # BM25, TF-IDF, Fusion reranker, grounding validator, chunk loader
│   │   └── schemas/              # Pydantic request & response models
│   ├── datasets/
│   │   ├── raw/                  # Source domain knowledge JSON files
│   │   ├── processed/            # Compiled instruction datasets
│   │   └── evaluation/           # 52-question benchmark suite (test_questions.json)
│   ├── genkit-model/             # Checkpoints (model_v6.pt, bpe_tokenizer_v5.json, config_v6.json)
│   ├── scripts/
│   │   ├── verify_checkpoint.py  # Standalone checkpoint validator
│   │   └── evaluate.py           # Automated benchmark evaluation runner
│   ├── tests/                    # 55 automated unit & integration tests
│   ├── training/                 # Model trainer & dataset preparation pipeline
│   ├── train.py                  # CLI training entrypoint
│   └── main.py                   # FastAPI backend server entrypoint
│
├── frontend/
│   ├── src/                      # React UI components, SSE stream client, CSS theme
│   ├── Dockerfile                # Multi-stage production build (Node -> Nginx)
│   └── package.json              # Frontend dependencies
│
├── nginx/                        # Nginx reverse proxy configuration
└── docs/                         # Extended operational documentation
```
