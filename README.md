# Genkit AI - Custom GPT Model with Hybrid RAG & React Frontend

An end-to-end, production-grade AI Chatbot built **100% from scratch** without any external AI models, APIs (such as OpenAI, Anthropic, or Hugging Face Hub), or cloud services. 

It features a custom-trained **PyTorch Causal Decoder-Only Transformer (GPT)**, an advanced **Hybrid RAG Engine** (BM25 + Vector Cosine + Knowledge Graph Expansion), **NLP & Safety Guardrails**, an automatic **MySQL** database persistence bootstrap, and a modern **React + Vite** frontend.

---

## 🌟 Architectural Highlights

- 🧠 **100% In-House PyTorch Causal GPT**: Built line-by-line using PyTorch with Multi-Head Causal Self-Attention, positional embeddings, GELU MLP blocks, and custom decoding strategy.
- 🔤 **Custom Tokenizer**: In-house BPE / WordPiece tokenizer built from scratch, mapping domain vocabulary without third-party tokenization libraries.
- 🔍 **Hybrid Retrieval-Augmented Generation (RAG)**: Multi-stage retrieval combining BM25 keyword search, custom TF-IDF dense vector cosine similarity, Knowledge Graph expansion, and context reranking.
- 🛡️ **NLP & Domain Guardrails**: Domain Guard filtering out off-topic queries, Intent Classification, Entity Extraction, and Response Validation to ensure strict factuality.
- ⚡ **Auto-Bootstrapping MySQL Storage**: Automatically creates the `genkit_ai` database if missing, initialising tables for chat history, user session facts, leads, and feedback.
- ⚛️ **Modern Glassmorphic React Frontend**: Floating chat widget built with React + Vite, featuring streaming responses, auto-scrolling, quick suggestions, and lead capture.

---

## 🔬 How the Custom AI Model & Pipeline Works

```text
User Question
     │
     ▼
[ 1. Text Preprocessing & Cleaning ] ── (Cleaner + Spell Checker)
     │
     ▼
[ 2. Domain Guard ] ────────────────── (Filters out-of-scope questions cleanly)
     │ (In-Domain)
     ▼
[ 3. NLP Intent & Entity Engine ] ───── (Intent Classifier + Entity Extractor)
     │
     ▼
[ 4. Hybrid RAG Retrieval Engine ] ─── (BM25 + Vector Cosine + Knowledge Graph)
     │
     ▼
[ 5. Context & Prompt Builder ] ──────── (Enriches RAG chunks into structured prompt)
     │
     ▼
[ 6. Custom PyTorch GPT Model ] ──────── (Decoder-Only Transformer Inference)
     │
     ▼
[ 7. Response Validator & Clean-up ] ── (Strips leaks, formats output, saves to MySQL)
     │
     ▼
Streamed Answer to User
```

### 1. Custom Decoder-Only Transformer Architecture (`backend/ai/llm/ml_model.py`)
The language model is a custom PyTorch Causal GPT model implemented from ground up:
- **Embedding Layer**: Learns token embeddings combined with learned positional embeddings:
  $$\text{Input Input Embedding} = W_e(x) + W_p(pos)$$
- **Causal Self-Attention (`CausalSelfAttention`)**: Uses multi-head self-attention with lower-triangular causal masking so tokens only attend to past tokens:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$
- **Transformer Block**: Composes Layer Normalization (`ln1`, `ln2`), Multi-Head Attention, residual skip connections, and a 4x expansion MLP with GELU activations.
- **Model Parameters**:
  - **Vocabulary Size**: 739 tokens
  - **Embedding Dimension ($n_{embd}$)**: 384
  - **Layers ($n_{layer}$)**: 10
  - **Attention Heads ($n_{head}$)**: 8
  - **Context Window ($block\_size$)**: 1024 tokens
  - **Total Parameters**: ~18.4 Million trainable parameters

### 2. Custom Tokenizer (`backend/ai/tokenizer/tokenizer.py`)
- Engineered from scratch to tokenize, encode, and decode raw text into numerical token IDs without Hugging Face dependencies.
- Handles special tokens (`<pad>`, `<unk>`, `<start>`, `<end>`, `<context>`, `<question>`, `<answer>`).

### 3. Sampling & Inference Engine (`backend/ai/llm/inference.py`)
Generates token sequences using custom sampling routines:
- **Temperature Control**: Scales logits to adjust entropy.
- **Top-K & Top-P (Nucleus) Filtering**: Truncates low-probability tokens.
- **Repetition Penalty**: Penalizes recently generated tokens to eliminate loop degeneration.

### 4. Hybrid RAG Retrieval System (`backend/ai/rag/`)
- **BM25 Search**: Captures exact keyword matches across domain documents.
- **Vector Cosine Similarity (`backend/ai/embeddings/embedding.py`)**: Computes dense vector embeddings from domain knowledge bases.
- **Knowledge Graph Expansion (`knowledge_graph.py`)**: Expands intent nodes and entities to retrieve related domain concepts.
- **Context Builder (`context_builder.py`)**: Reranks and assembles retrieved chunks into concise prompt context.

### 5. Automated Database Bootstrapping (`backend/database/mysql.py`)
- On startup, `ensure_database_exists()` connects to MySQL server without database context and creates `genkit_ai` (`utf8mb4_unicode_ci`) if deleted or missing.
- Automatically creates tables (`chats`, `leads`, `feedback`, `user_profiles`), indices, and analytics views.

---

## 📂 Project Structure

```text
Chatbot-own-model/
├── backend/
│   ├── ai/
│   │   ├── embeddings/     # Custom TF-IDF vector store & dataset indexer
│   │   ├── llm/            # PyTorch GPT model architecture, inference & trainer
│   │   ├── memory/         # Session memory & facts storage
│   │   ├── nlp/            # Domain Guard, Intent Classifier & Entity Extractor
│   │   ├── preprocessing/  # Text normalizers, cleaner & spell checker
│   │   ├── rag/            # Hybrid retriever, Knowledge Graph & context builder
│   │   └── tokenizer/      # Custom Word/Subword tokenizer implementation
│   ├── api/                # FastAPI routers (chat, lead, feedback, health, version)
│   ├── database/           # MySQL connection pool, auto-bootstrap & schemas
│   ├── dataset/            # Domain knowledge JSON files
│   ├── genkit-model/       # Trained model parameters (config.json & vocab.json)
│   ├── utils/              # Helper utilities & logging configuration
│   ├── chatbot.py          # Master RAG + LLM execution pipeline
│   ├── config.py           # Application settings & environment loader
│   ├── evaluate.py         # Model & RAG evaluation suite
│   ├── init_database.py    # Database auto-recreation & setup script
│   ├── main.py             # FastAPI entrypoint server
│   ├── predict.py          # Terminal CLI chatbot interface
│   ├── train.py            # Custom GPT training script
│   ├── .env.example        # Environment variable template
│   └── requirements.txt    # Python dependencies
│
├── src/                    # React + Vite Frontend Application
│   ├── components/
│   │   └── ChatWidget/     # Glassmorphic Chat Widget components
│   ├── services/           # API integration service (streaming chat, lead API)
│   ├── App.jsx             # Root React component
│   └── index.css           # Styling tokens & animations
│
├── .gitignore              # Repository exclusion rules
├── index.html              # HTML Shell
├── package.json            # Node.js dependencies & scripts
├── vite.config.js          # Vite config & API proxy setup
└── README.md               # Documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.9+** (PyTorch, Scikit-learn, FastAPI, MySQL-connector)
- **Node.js 18+** & **npm**
- **MySQL Server 8.0+** running locally (e.g., `localhost:3306`)

---

### 1. Backend Setup & Database Initialization

1. Open terminal and navigate to `backend`:
   ```bash
   cd backend
   ```

2. Create & activate Python virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   *Update `MYSQL_USER` and `MYSQL_PASSWORD` in `.env` if necessary.*

5. Initialize / Re-create the MySQL database:
   ```bash
   python init_database.py
   ```
   *This automatically creates `genkit_ai` database and all required tables.*

6. Start FastAPI Backend:
   ```bash
   python main.py
   ```
   - Server runs on `http://localhost:8000`
   - Interactive Docs: `http://localhost:8000/docs`

---

### 2. Frontend Setup

1. Open a new terminal in the project root:
   ```bash
   npm install
   ```

2. Start Vite development server:
   ```bash
   npm run dev
   ```
   - Application runs on `http://localhost:5173`
   - Proxies `/chat`, `/lead`, `/feedback`, `/health` to backend on port 8000.

---

## 🛠️ Model Training & CLI Testing

### Interactive Terminal Testing
Test the chatbot model directly in the command line:
```bash
cd backend
python predict.py
```

### Train / Fine-tune Custom GPT Model
Train the PyTorch decoder transformer on your dataset:
```bash
cd backend
python train.py
```

### Run RAG & Model Evaluation
Evaluate precision, recall, and response factual scores:
```bash
cd backend
python evaluate.py
```

---

## 🛡️ License & Git Repository Rules
- Large weight checkpoints (`*.pt`, `checkpoint.pt`), virtual environment folders (`venv/`), logs (`logs/`), node modules (`node_modules/`), and sensitive `.env` files are excluded via `.gitignore`.
- Baseline architecture configs (`config.json` and `vocab.json`) are tracked to allow immediate model initialization.
