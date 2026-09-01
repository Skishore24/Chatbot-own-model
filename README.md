# Genkit AI — 100% Self-Hosted Custom LLM + Hybrid RAG Assistant

<p align="center">
  <img src="frontend/public/vite.svg" width="80" height="80" alt="Genkit Logo" />
</p>

<p align="center">
  <b>Enterprise-Grade, 100% Custom Decoder Transformer & Deterministic Lexical RAG Engine</b><br>
  <i>Zero External AI APIs • Zero Cloud Embedding Dependencies • 100% Locally Trained & Self-Hosted</i>
</p>

<p align="center">
  <a href="#-key-features"><img src="https://img.shields.io/badge/Architecture-Custom_GPT_v6.0-blue.svg" alt="Architecture"></a>
  <a href="#-hardware-optimization"><img src="https://img.shields.io/badge/Hardware-RTX_3050_6GB_Optimized-green.svg" alt="Hardware"></a>
  <a href="#-hybrid-rag-engine"><img src="https://img.shields.io/badge/RAG-BM25_+_TF--IDF_+_RRF-orange.svg" alt="RAG"></a>
  <a href="#-testing--benchmarks"><img src="https://img.shields.io/badge/Tests-100%25_Passing-brightgreen.svg" alt="Tests"></a>
  <a href="#-license"><img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License"></a>
</p>

---

## 🌟 Key Highlights

- **100% Own Neural Model**: No OpenAI, Gemini, Claude, Llama, Mistral, HuggingFace inference, or external embedding APIs.
- **Custom Transformer Architecture**: Implemented from scratch in PyTorch with **Grouped-Query Attention (GQA)**, **Rotary Positional Embeddings (RoPE)** with cache offsets, **RMSNorm**, and **SwiGLU FFN**.
- **Byte-Fallback BPE Tokenizer**: Deterministic learned merge table + 256 raw byte tokens (<0x00>..<0xFF>) guaranteeing 100% unicode/emoji safety without out-of-vocabulary `<unk>` replacements.
- **Deterministic Hybrid RAG Engine**: Algorithmic lexical retrieval utilizing in-memory **Inverted Index**, **BM25 Okapi**, **TF-IDF Cosine Similarity**, and **Reciprocal Rank Fusion (RRF)** reranking.
- **Grounding Validator & Out-of-Domain Refusal**: Rejects general and unsupported non-Genkit questions deterministically to eliminate hallucinations.
- **Real SSE Token Streaming**: Server-Sent Events streaming token-by-token with KV-cache acceleration.
- **Dual Database Persistence**: Native MySQL connection pool with automatic local SQLite (`genkit.db`) fallback.
- **Modern React + Vite Chatbot**: Responsive dark/light theme, lead capture modal, markdown formatting, typing indicators, and session persistence.

---

## 🏗️ Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │    React + Vite Frontend (Port 5173)   │
                      └───────────────────┬────────────────────┘
                                          │  SSE Stream / REST
                                          ▼
                      ┌────────────────────────────────────────┐
                      │      FastAPI Backend (Port 8000)       │
                      │  Rate Limiting • Prompt Injection Guard │
                      └───────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
       ┌────────────────────────┐                  ┌────────────────────────┐
       │   Hybrid RAG Engine    │                  │  Enterprise GPT Model  │
       │  • Inverted Index      │                  │  • 6 Layers / 384 Dim  │
       │  • BM25 Okapi          │                  │  • GQA (6 Heads, 2 KV) │
       │  • TF-IDF Cosine Sim   │                  │  • RoPE + SwiGLU FFN   │
       │  • RRF Hybrid Reranker │                  │  • RMSNorm + KV Cache  │
       └────────────┬───────────┘                  └────────────┬───────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │        Database Persistence            │
                      │   MySQL Pool + Local SQLite Fallback   │
                      └────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Chatbot-own-model/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI REST & SSE Streaming Routers
│   │   ├── core/            # Configuration, Logging, Security Guard
│   │   ├── database/        # Thread-Safe Dual DB Manager (MySQL + SQLite)
│   │   ├── llm/             # Custom GPT Architecture, Tokenizer, Generation
│   │   ├── rag/             # Inverted Index, BM25, TF-IDF, RRF, Grounding
│   │   ├── schemas/         # Pydantic Request/Response Schemas
│   │   └── main.py          # Master FastAPI Application
│   ├── datasets/            # Curated Genkit Enterprise Knowledge JSONs
│   ├── genkit-model/        # Trained PyTorch Checkpoints & Tokenizers
│   ├── logs/                # Structured Application Logs & Traces
│   ├── reports/             # Benchmark Evaluation Reports (JSON/MD)
│   ├── tests/               # Unit & Integration Pytest Suite
│   ├── training/            # Dataset Generator, BPE Trainer, LLM Training Pipeline
│   ├── evaluate.py          # Benchmark Evaluation CLI Launcher
│   ├── main.py              # Backend Server CLI Launcher
│   ├── predict.py           # Interactive CLI Prediction Utility
│   ├── requirements.txt     # Python Dependencies
│   └── train.py             # Master Model Training Launcher
├── frontend/
│   ├── src/
│   │   ├── assets/          # Static Assets & Branding
│   │   ├── components/      # React UI Components (ChatWidget, LeadForm, etc.)
│   │   ├── hooks/           # Custom React Hooks (useChat, useChatStream)
│   │   ├── services/        # API Client & SSE Client
│   │   └── utils/           # Markdown Parser & Utility Helpers
│   ├── package.json         # Node Dependencies
│   └── vite.config.js       # Vite Build Configuration
├── docker/
│   └── Dockerfile           # Backend Container Definition
├── docs/                    # Technical Architecture & Deployment Docs
├── docker-compose.yml       # Full-Stack Orchestration (MySQL + Backend + Frontend)
├── .env.example             # Environment Configuration Template
└── README.md                # Project Overview & Quick Start
```

---

## 🚀 Quick Start

### Option A: Docker Compose (Recommended)

```bash
# Build and run full-stack services (MySQL, FastAPI Backend, React Frontend)
docker compose up --build
```
- Frontend: `http://localhost:5173`
- Backend API Docs: `http://localhost:8000/docs`

### Option B: Local Setup

#### 1. Backend Setup & Startup

```bash
# Clone repository
git clone https://github.com/Skishore24/Chatbot-own-model.git
cd Chatbot-own-model/backend

# Create virtual environment & install requirements
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Start the backend server
python main.py
```
> Server runs at `http://localhost:8000`. Interactive API Docs: `http://localhost:8000/docs`.

#### 2. Frontend Setup & Startup

```bash
cd ../frontend
npm install
npm run dev
```
> Frontend runs at `http://localhost:5173`.

---

## 🏋️ Training the Custom LLM

The training pipeline is optimized for an **NVIDIA GeForce RTX 3050 6GB Laptop GPU**:

```bash
cd backend

# Execute master training script
python train.py --epochs 60 --batch-size 4 --accum-steps 8 --block-size 512 --vocab-size 10000
```

### Hardware Optimization Features:
- **Micro-Batching + Gradient Accumulation**: Batch size of 4 with 8 accumulation steps (Effective batch size = 32).
- **Automatic Mixed Precision (AMP)**: `bfloat16`/`float16` training with `GradScaler`.
- **Pre-flight Smoke Test**: Automatic forward/backward stability verification before training.
- **Cosine Warmup LR Scheduler**: Warmup for first 200 steps decaying to $1\times 10^{-5}$.

---

## 🧪 Testing & Verification

Run the comprehensive unit and integration test suite:

```bash
cd backend
python -m unittest discover -s tests
```

### Benchmark Evaluation Suite

```bash
python evaluate.py
```
Outputs `evaluation_report.json` and `evaluation_report.md` tracking in-domain retrieval accuracy, out-of-domain refusal accuracy, and latency.

---

## 📚 Technical Documentation

Detailed technical documents are available in the [`docs/`](docs/) directory:
- [Production Deployment Guide](docs/DEPLOYMENT.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Neural Model & Tokenizer Design](docs/MODEL.md)
- [Hybrid RAG & Grounding Subsystem](docs/RAG.md)
- [FastAPI & SSE Streaming API Reference](docs/API.md)
- [Model Training & GPU Acceleration](docs/TRAINING.md)
- [Local Development Guide](docs/DEVELOPMENT.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
