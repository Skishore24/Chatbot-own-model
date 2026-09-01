# Genkit AI — 100% Self-Hosted Custom LLM + Hybrid RAG Assistant

<p align="center">
  <img src="frontend/public/vite.svg" width="80" height="80" alt="Genkit Logo" />
</p>

<p align="center">
  <b>Enterprise-Grade, 100% Custom Decoder Transformer & Deterministic Lexical RAG Engine</b><br>
  <i>Zero External AI APIs • Zero Cloud Embedding Dependencies • 100% Locally Trained & Self-Hosted</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Custom_GPT_v6.1-blue.svg" alt="Architecture">
  <img src="https://img.shields.io/badge/Hardware-CUDA_AMP_Accelerated-green.svg" alt="Hardware">
  <img src="https://img.shields.io/badge/RAG-BM25_+_TF--IDF_+_Fusion-orange.svg" alt="RAG">
  <img src="https://img.shields.io/badge/Tests-55%2F55_Passing-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License">
</p>

---

## 🌟 Key Highlights

- **100% Self-Hosted Custom LLM**: No OpenAI, Gemini, Claude, Ollama, HuggingFace inference, or external embedding APIs.
- **Custom Transformer Architecture**: Implemented from scratch in PyTorch with **Grouped-Query Attention (GQA)**, **Rotary Positional Embeddings (RoPE)** with KV-Cache offsets, **RMSNorm**, and **SwiGLU FFN**.
- **Byte-Fallback BPE Tokenizer**: Deterministic learned merge table (`bpe_tokenizer_v5.json`, Vocab Size: 2,084) + 256 raw byte tokens (`<0x00>`..`<0xFF>`) guaranteeing 100% UTF-8/emoji safety without out-of-vocabulary `<unk>` replacements.
- **Atomic & Verified Checkpoint Manager**: Safe atomic checkpoint saves (`.tmp` -> verification -> state dict validation -> `.bak` rotation -> atomic replace). Never corrupts production checkpoints.
- **Deterministic Hybrid RAG Engine**: Algorithmic lexical retrieval utilizing in-memory **Inverted Index**, **BM25 Okapi**, **TF-IDF Cosine Similarity**, and **Hybrid Fusion Reranking**.
- **Grounding Validator & Out-of-Domain Refusal**: Rejects general and unsupported non-Genkit questions deterministically to eliminate hallucinations.
- **Clean Execution Path Separation**: Explicit routing between `llm_rag`, `rag_direct`, and `system` refusal modes.
- **Real SSE Token Streaming**: Server-Sent Events streaming token-by-token with KV-Cache acceleration.
- **Dual Database Persistence**: Native MySQL connection pool with automatic local SQLite (`genkit.db`) fallback.
- **Modern React + Vite Frontend**: Responsive UI, lead capture modal, markdown formatting, typing indicators, and session persistence.
- **Production Docker**: Multi-stage Nginx frontend build and production hardened backend.

---

## 🏗️ Architecture Overview

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

## 📁 Repository Structure

```
Chatbot-own-model/
├── README.md                     # Master project documentation
├── ARCHITECTURE.md               # Architectural specification
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
    ├── ARCHITECTURE.md
    ├── DEPLOYMENT.md
    ├── MODEL.md
    └── DEVELOPMENT.md
```

---

## 🚀 Quickstart & Installation

### 1. Backend Setup

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify Model Checkpoint

```powershell
python scripts/verify_checkpoint.py --path genkit-model/model_v6.pt
```

### 3. Run Automated Tests

```powershell
python -m pytest tests/ -v
```

### 4. Train the Model

```powershell
python train.py --epochs 30 --batch-size 4 --accum-steps 8
```

### 5. Launch Backend API Server

```powershell
python main.py
```
*(Server launches on `http://0.0.0.0:8000` with Swagger UI at `http://localhost:8000/docs`)*

### 6. Launch React Frontend

```powershell
cd frontend
npm install
npm run dev
```

---

## 🐳 Docker Deployment

### Production Mode (Multi-stage Nginx + FastAPI + MySQL)
```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

### Development Mode (Hot Reload)
```bash
docker-compose -f docker-compose.dev.yml up
```

---

## 📊 Benchmark & Evaluation

Run the automated evaluation benchmark across 52 company & out-of-domain questions:

```powershell
python scripts/evaluate.py
```

Generated reports are stored in `backend/reports/`:
- `training_report.json`
- `evaluation_report.json`

---

## 🔒 Security & Guardrails

1. **Rate Limiting**: In-memory token-bucket rate limiter per IP address.
2. **Prompt Injection Scanner**: Regex heuristics guarding against instruction override attacks.
3. **SQL Injection Prevention**: 100% parameterized SQL queries via dual MySQL/SQLite repositories.
4. **Out-of-Domain Refusal**: Rejects out-of-scope questions without hallucination.

---

## 📄 License

MIT License — Copyright (c) 2026 Genkit.in
