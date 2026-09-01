# Genkit AI — Master System Overview

> **A Complete Developer Guide to the 100% Self-Hosted Neural LLM & Hybrid Deterministic RAG Architecture**

---

## 1. Purpose & Product Definition

**Genkit AI** is a company knowledge assistant tailored for **Genkit.in**. It delivers instant, grounded answers about company services, technology stacks, pricing, past portfolio projects, team members, and processes.

### Core Principles
1. **100% Self-Hosted & Local**: Zero dependence on OpenAI, Google Gemini, Anthropic Claude, Groq, OpenRouter, HuggingFace Inference API, or external cloud embedding models.
2. **Custom Neural LLM**: Proprietary decoder-only Transformer built in PyTorch with Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE) with cache offsets, RMSNorm, and SwiGLU FFN.
3. **Deterministic Lexical RAG**: Inverted Index, BM25 Okapi lexical scoring, TF-IDF Vector Space Cosine Similarity, and Reciprocal Rank Fusion (RRF).
4. **Strict Grounding & Scope Refusal**: Prefer verified refusals on out-of-domain questions rather than ungrounded hallucinations.

---

## 2. Directory Structure

```
Chatbot-own-model/
├── .env.example                     # Environment template
├── README.md                        # Primary project documentation
├── backend/
│   ├── app/
│   │   ├── api/                     # FastAPI route handlers
│   │   │   ├── auth.py              # API key & token verification
│   │   │   ├── chat.py              # Synchronous chat endpoint
│   │   │   ├── feedback.py          # User feedback endpoint
│   │   │   ├── health.py            # System health & diagnostics
│   │   │   ├── history.py           # Session history endpoints
│   │   │   ├── leads.py             # Business lead capture
│   │   │   └── streaming.py         # Server-Sent Events (SSE) streaming
│   │   ├── core/                    # Core configuration & services
│   │   │   ├── config.py            # AppSettings & path resolution
│   │   │   ├── logger.py            # Structured logging & trace IDs
│   │   │   └── security.py          # Sanitization, rate limit, token auth
│   │   ├── database/                # Persistence layer
│   │   │   ├── connection.py        # MySQL pool + SQLite fallback manager
│   │   │   ├── models.py            # SQL schemas & DDL
│   │   │   └── repository.py        # Parameterized data access objects
│   │   ├── llm/                     # Neural Model & Tokenizer
│   │   │   ├── attention.py         # Cache-aware GQA with RoPE
│   │   │   ├── config.py            # GPTConfig & parameter validation
│   │   │   ├── generation.py        # Autoregressive generation & StreamDecoder
│   │   │   ├── inference.py         # Checkpoint loader & ModelStatus lifecycle
│   │   │   ├── model.py             # EnterpriseGPTModel architecture
│   │   │   ├── normalization.py     # RMSNorm layer
│   │   │   ├── positional.py        # Rotary Positional Embeddings (RoPE)
│   │   │   └── tokenizer.py         # Byte-Fallback BPE Tokenizer
│   │   ├── rag/                     # Local Deterministic RAG Engine
│   │   │   ├── bm25.py              # BM25 Okapi ranking
│   │   │   ├── chunker.py           # DocumentChunk dataclass
│   │   │   ├── grounding.py         # Grounding validator & domain refusal guard
│   │   │   ├── index.py             # Inverted index & text normalization
│   │   │   ├── loader.py            # JSON knowledge ingestion
│   │   │   ├── pipeline.py          # Unified HybridRAGPipeline
│   │   │   ├── reranker.py          # Reciprocal Rank Fusion & coverage reranking
│   │   │   └── tfidf.py             # TF-IDF Cosine Similarity retriever
│   │   ├── schemas/                 # Pydantic request/response schemas
│   │   └── main.py                  # Master FastAPI application
│   ├── datasets/                    # Curated JSON knowledge files
│   ├── genkit-model/                # Model checkpoints & tokenizer files
│   ├── tests/                       # Unit, integration & security test suite
│   ├── training/                    # Model & tokenizer training pipelines
│   ├── evaluate.py                  # Golden evaluation benchmark runner
│   ├── predict.py                   # Interactive CLI predictor
│   ├── requirements.txt             # Python backend dependencies
│   └── train.py                     # Master model training entrypoint
├── docs/                            # Comprehensive engineering guides
└── frontend/                        # React + Vite client application
    ├── src/
    │   ├── components/              # UI components (ChatWidget, Messages, etc.)
    │   ├── hooks/                   # useChat, useAutoGrow custom hooks
    │   ├── services/                # API client & SSE reader
    │   ├── utils/                   # Markdown renderer & helper utilities
    │   ├── App.jsx                  # Main application container
    │   └── index.css                # Polished design system
    └── package.json                 # Frontend dependencies
```

---

## 3. End-to-End Execution Flows

### A. Chat Request Flow

```
[User Query]
     │
     ▼
[Frontend: React / SSE Client]
     │  POST /api/v1/chat/stream
     ▼
[Security Middleware]
     ├── Rate Limiting (Sliding Window Per-IP)
     ├── Input Sanitization
     └── Prompt Injection Scanner
     │
     ▼
[Hybrid RAG Engine]
     ├── Lexical Inverted Index
     ├── BM25 Okapi Scoring
     ├── TF-IDF Cosine Similarity
     └── RRF Hybrid Reranker -> Top-K Chunks
     │
     ▼
[Grounding & Intent Validator]
     ├── Out-of-Domain / Unsupported Check
     │       └── If OOD: Return Verified Refusal Message
     └── If Grounded: Formulate Structured System Prompt
     │
     ▼
[Custom Neural LLM Generation Engine]
     ├── Prefill Prompt into KV-Cache
     ├── Step-by-Step Autoregressive Token Decoding
     ├── Byte-Fallback StreamDecoder Buffer (UTF-8 Safe)
     └── Stop Token / Repetition Penalty Guards
     │
     ▼
[Answer Grounding Verification]
     ├── Check Generated Content against Retrieved Chunks
     └── Fallback to Verified Context Chunk if Ungrounded
     │
     ▼
[Database Persistence & Response]
     ├── Record Session & Message into MySQL (or SQLite Fallback)
     └── Stream SSE Chunks + Source Citations to Frontend
```

---

## 4. Key Subsystem Details

### 1. Custom Neural LLM (`app/llm/`)
- **Decoder-Only Transformer**: Pre-LayerNorm architecture with RMSNorm.
- **Grouped-Query Attention (GQA)**: $H_Q=6$ query heads, $H_{KV}=2$ key-value heads with KV-cache for single-token decoding speedups.
- **Rotary Position Embeddings (RoPE)**: Applied to queries and keys with position offset tracking for cached continuation.
- **SwiGLU Activation**: Gated feed-forward network using $8/3$ intermediate dimension scaling.
- **Byte-Fallback BPE Tokenizer**: 256 raw byte tokens `<0x00>..<0xFF>` ensure zero out-of-vocabulary loss for arbitrary Unicode text and emojis.

### 2. Deterministic RAG Pipeline (`app/rag/`)
- **Inverted Index**: Memory-efficient term-to-document posting list.
- **BM25 Okapi**: Lexical scoring parameterized by $k_1=1.5$ and $b=0.75$.
- **TF-IDF**: Sublinear TF scaling with L2-normalized cosine similarity.
- **Hybrid Reranker**: Combines normalized BM25, TF-IDF, RRF rank scoring ($k=60$), term coverage, title matching, and keyword priority weighting.
- **Grounding Guard**: Computes continuous confidence score $[0.0, 1.0]$. Refuses general queries deterministically.

### 3. Dual Database Persistence (`app/database/`)
- **Primary Engine**: MySQL connection pooling with parameterized SQL queries.
- **Automatic Fallback**: Local SQLite database (`genkit.db`) initialized seamlessly when MySQL is offline or unconfigured.

### 4. Frontend UI (`frontend/`)
- **React 19 + Vite**: Modern, lightweight, high-performance interface.
- **Streaming SSE Consumer**: Smooth incremental token rendering with automatic reconnection and streaming lock guards.
- **Verified Source Badges**: Interactive citations displaying source document title and category.
- **Lead Capture & Feedback Modals**: Clean user feedback and sales lead capture.

---

## 5. Development & Testing Commands

### Run Backend Tests
```bash
cd backend
.\venv\Scripts\pytest backend\tests -v
```

### Run Benchmark Evaluation
```bash
cd backend
python evaluate.py
```

### Build Frontend
```bash
cd frontend
npm run build
```

### Start Local Development Servers
- **Backend**: `python backend/app/main.py` (Port 8000)
- **Frontend**: `cd frontend && npm run dev` (Port 5173)
