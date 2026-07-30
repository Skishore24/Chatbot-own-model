# 🤖 GENKIT AI v5.0 Enterprise

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.0-61dafb.svg)](https://react.dev/)
[![MySQL 8.0](https://img.shields.io/badge/MySQL-8.0-4479A1.svg)](https://www.mysql.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GENKIT AI v5.0 Enterprise** is a high-throughput, production-ready, domain-restricted AI assistant built natively in **Python** and **PyTorch**. The system enforces 100% technological sovereignty: **0% external AI APIs** (No OpenAI, Gemini, Claude, Llama, HuggingFace), **0% third-party AI frameworks** (No LangChain, LlamaIndex), and **0% cloud vector databases** (No Pinecone, ChromaDB, FAISS).

---

## 🌟 Key Architectural Features

- ⚡ **Sub-20ms Token Latency**: Custom PyTorch Causal Decoder with **Grouped-Query Attention (GQA)**, **Paged KV-Cache**, **NTK-aware RoPE**, **RMSNorm**, and **SwiGLU FFN**.
- 🔤 **Byte-Fallback BPE Tokenizer**: 16,000 vocabulary size Trie-based tokenizer with byte fallback (`<0x00>` to `<0xFF>`), guaranteeing **0% `<unk>` token emissions**.
- 🔍 **GraphRAG + PyTorch Dense Search**: 11-stage search engine combining Lexical BM25, PyTorch INT8 Vector HNSW, Reciprocal Rank Fusion (RRF), and BFS Entity-Relation Graph Traversal.
- 📡 **Real-time SSE Token Streaming**: Asynchronous FastAPI endpoints serving token-by-token HTTP Server-Sent Events (TTFB < 180ms).
- 🛡️ **Zero-Trust Security**: Input XSS/SQLi sanitization, prompt injection attack scanning, and hallucination output verification.
- 💾 **Async Persistence**: `aiomysql` database connection pool for non-blocking MySQL state management.

---

## 🏗️ Architectural Overview

```text
[ React 18 UI ] ──► [ Security Gating ] ──► [ Domain Guard Gateway ] ──► [ Coreference Resolver ]
                                                                                │
                                                                                ▼
[ SSE Streaming ] ◄── [ PyTorch GQA GPT ] ◄── [ Tokenizer ] ◄── [ Hybrid RAG + GraphRAG ]
```

---

## 🛠️ Quickstart Guide

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/Genkit/Chatbot-own-model.git
cd Chatbot-own-model/backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Launch Production Server
```bash
# Launch FastAPI Backend Server
python main.py
```
*Server will start at `http://localhost:8000`. API documentation available at `http://localhost:8000/docs`.*

### Step 3: Run Master Test Suite
```bash
# Run all unit and integration tests
python -m unittest discover -s tests -p "test_*.py"
```

---

## 🐳 Docker Deployment

```bash
# Launch full multi-container stack (FastAPI Backend + MySQL)
docker-compose up --build -d
```

---

## 📚 Complete Documentation Suite

All system documentation is available in the **[`docs/`](docs/)** directory:
- 📘 **[GENKIT_AI_MASTER_DOCUMENTATION.md](docs/GENKIT_AI_MASTER_DOCUMENTATION.md)** — Master System Manual
- 🏛️ **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Software Architecture Specification
- 📡 **[API_REFERENCE.md](docs/API_REFERENCE.md)** — REST & Streaming API Documentation

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
