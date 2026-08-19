# 🤖 GENKIT AI v5.0 Enterprise — 100% Self-Hosted AI Assistant

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.0-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.x-646CFF.svg?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GENKIT AI v5.0 Enterprise** is a high-throughput, production-grade, domain-restricted AI chatbot system built natively in **Python**, **PyTorch**, **FastAPI**, and **React**. 

The system guarantees **100% data sovereignty & offline execution**:
- 🔒 **Zero External AI Cloud APIs** — No OpenAI, Gemini, Claude, or Hugging Face cloud endpoints.
- ⚡ **Zero External Vector DBs** — Native PyTorch BLAS GEMM dense vector index + Lexical BM25.
- 🛡️ **Zero Data Leakage** — All inference, retrieval, tokenization, and database persistence run entirely locally.

---

## 🌟 Core Architecture & Capabilities

### 1. Custom PyTorch GPT Architecture
- **Attention Mechanism**: Grouped-Query Attention (**GQA**, 3:1 Q:KV head ratio) with Paged KV-Cache for ultra-fast autoregressive decoding.
- **Positional Encoding**: NTK-aware Rotary Position Embeddings (**RoPE**).
- **Normalisation & Activation**: Root Mean Square Normalization (**RMSNorm**) and **SwiGLU** Feed-Forward Networks.

### 2. Byte-Fallback BPE Tokenizer
- Trained on **15,848 domain sentences** ([`backend/genkit-model/bpe_tokenizer_v5.json`](backend/genkit-model/bpe_tokenizer_v5.json)).
- Trie-based subword encoder with byte fallbacks ($0\text{x}00$ to $0\text{xFF}$), guaranteeing **0% `<unk>` token emissions**.

### 3. Dual-Path Hybrid RAG + Knowledge Graph
- **Dataset Ingestion**: Automatically indexes **7,964 domain knowledge passages** across all 12 JSON files (`company.json`, `services.json`, `pricing.json`, `faq.json`, `portfolio.json`, `contact.json`, `technologies.json`, etc.).
- **Sparse + Dense Search**: Lexical BM25 search fused with PyTorch neural vector embeddings ($d=384$) via **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \sum_{m \in \{BM25, Dense\}} \frac{1}{k + r_m(d)}$$
- **Knowledge Graph Engine**: BFS entity-relationship graph traversal with candidate reranking and Maximal Marginal Relevance (MMR) diversification.

### 4. Grounded RAG Coherence Engine
- **Repetition Loop & Gibberish Protection**: Detects and intercepts token loops and out-of-distribution neural outputs.
- **Grounded Synthesizer**: Produces clean, structured, formatted Markdown responses directly grounded in the retrieved domain knowledge.

### 5. Server-Sent Events (SSE) Streaming
- Real-time token/chunk streaming endpoint (`/api/v5/chat/stream`) with sub-180ms Time-to-First-Token (TTFB).

---

## 📁 Repository Structure

```text
Chatbot-own-model/
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   │   ├── llm/              # PyTorch GPT, Inference, Trainer, Model Loader
│   │   │   ├── rag/              # Hybrid Retriever, Dense Embedder, Graph Engine
│   │   │   └── tokenizer/        # Byte-Fallback BPE Tokenizer
│   │   ├── api/                  # FastAPI Endpoints & SSE Streaming Engine
│   │   ├── core/                 # Config, Logging, Zero-Trust Security Service
│   │   └── database/             # Connection pooling & chat session persistence
│   ├── datasets/                 # 12 Domain Knowledge Datasets (7,964 entries)
│   ├── genkit-model/             # Trained Tokenizer & Model Checkpoints
│   ├── tests/                    # Comprehensive Unit & Integration Test Suite
│   ├── main.py                   # FastAPI Application Entrypoint
│   ├── train.py                  # Model & Tokenizer Training CLI Pipeline
│   ├── predict.py                # Interactive CLI Predictor
│   ├── evaluate.py               # RAG & Latency Evaluation Suite
│   └── requirements.txt          # Python Dependencies
├── frontend/
│   ├── src/
│   │   ├── components/           # React Chat Widget & UI Components
│   │   ├── services/             # SSE Streaming Client & API Service
│   │   └── hooks/                # useChat Hook
│   ├── vite.config.js            # Vite Dev Server & API Reverse Proxy
│   └── package.json              # Node Dependencies
├── scripts/                      # Helper & Execution Scripts
├── docs/                         # Extended System Documentation
├── .gitignore                    # Master Git Ignore Configuration
└── README.md                     # Project Master Guide
```

---

## 🚀 Quick-Start Guide

### Prerequisites
- **Python 3.10+** (Recommended: Python 3.11 or 3.12)
- **Node.js 18+** & **npm**

---

### Step 1: Set Up Backend Environment

```powershell
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2: Launch the Backend API Server

```powershell
# From project root with venv active
& ".\backend\venv\Scripts\python.exe" backend/main.py
```
- **Backend API**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Docs**: `http://127.0.0.1:8000/docs`
- **Health Endpoint**: `http://127.0.0.1:8000/health`

---

### Step 3: Launch the Frontend Web Interface

```powershell
# In a new terminal window
cd frontend

# Install node dependencies
npm install

# Start Vite development server
npm run dev
```
- **Web Chat Interface**: `http://localhost:5173`

---

### Step 4: Terminal CLI Chat (Optional)

To interact with the chatbot directly in the terminal:
```powershell
& ".\backend\venv\Scripts\python.exe" backend/predict.py
```

---

## 🧪 Testing & Evaluation

### Run Master Unit & Integration Tests (14 Tests)
```powershell
& ".\backend\venv\Scripts\python.exe" -m unittest discover -s backend/tests
```

### Run RAG & Latency Evaluation Suite
```powershell
& ".\backend\venv\Scripts\python.exe" backend/evaluate.py
```

### Train Custom Tokenizer and Model Checkpoint
```powershell
& ".\backend\venv\Scripts\python.exe" backend/train.py --epochs 2 --batch-size 8 --vocab-size 2000
```

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v5/chat/stream` | Server-Sent Events (SSE) real-time token streaming |
| `POST` | `/api/v5/chat/query` | Synchronous JSON chat completion |
| `POST` | `/api/v5/lead` | Lead capture & contact submission |
| `GET` | `/api/v5/history` | Retrieve conversation history by session ID |
| `GET` | `/api/v5/model-info` | Model configuration and parameter statistics |
| `GET` | `/health` | Server health & readiness check |

---

## 🛡️ Security & Privacy

- **Zero Outbound Calls**: Complete network isolation during inference and retrieval.
- **Input Sanitization**: Built-in SQLi, XSS, and Prompt Injection regex filters.
- **Rate Limiting**: Sliding-window IP rate limiter preventing denial-of-service attempts.

---

## 📄 License
This project is licensed under the MIT License — see the `LICENSE` file for details.
