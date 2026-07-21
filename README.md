# Genkit AI Chatbot - Custom GPT Model with Hybrid RAG & React Frontend

An end-to-end, production-grade AI Chatbot application featuring a custom-trained **PyTorch Causal Transformer (GPT)** model, an advanced **Hybrid Retrieval-Augmented Generation (RAG)** pipeline (BM25 + Vector Cosine + Knowledge Graph Expansion + Cross-Encoder Candidate Reranking), **MySQL** session & lead persistence, and a modern **React + Vite** floating chat widget.

---

## 🌟 Key Features

- 🧠 **Custom Causal GPT Engine**: PyTorch-based Causal Transformer with multi-head self-attention, trained on domain-specific datasets with a custom Byte-Pair / WordPiece tokenizer.
- 🔍 **Advanced Hybrid RAG System**: Multi-stage retrieval using BM25 keyword search, TF-IDF / dense vector cosine similarity, Knowledge Graph expansion, and candidate cross-encoder reranking.
- 🛡️ **NLP & Safety Guardrails**: Built-in Domain Guard, Intent Classifier, Sentiment Analyzer, Entity Extractor, and Response Validation to prevent off-domain queries and hallucinations.
- 💾 **Session & Memory Persistence**: MySQL connection pool handling real-time chat logging, session context, lead capture, and long-term user facts extraction.
- 🚀 **FastAPI High-Performance Backend**: Async REST API endpoints for `/api/chat`, `/api/lead`, `/api/feedback`, `/api/health`, `/api/auth`, and `/api/model`.
- 📊 **CLI & Evaluation Suite**: Includes `predict.py` for interactive terminal testing and `evaluate.py` for RAG precision, recall, and response score evaluation.
- ⚛️ **Modern React + Vite Frontend**: High-performance floating chat widget featuring glassmorphism design, theme support, auto-scrolling, lead capture modal, quick suggestions, and markdown rendering.

---

## 📂 Production Project Structure

```text
Chatbot-own-model/
├── backend/
│   ├── ai/
│   │   ├── agents/         # Autonomous agent orchestration & task dispatchers
│   │   ├── embeddings/     # Custom TF-IDF vector store & dataset indexer
│   │   ├── evaluation/     # RAG & model evaluation metrics engine
│   │   ├── llm/            # PyTorch GPT model, inference, prompt builder & trainer
│   │   ├── memory/         # Session memory & long-term fact storage
│   │   ├── nlp/            # Domain Guard, Intent Classifier, Entity Extractor & Validator
│   │   ├── preprocessing/  # Text normalizers, cleaning & spell check utilities
│   │   ├── rag/            # Hybrid retriever, Knowledge Graph, reranker & context builder
│   │   ├── tokenizer/      # Custom Byte-Pair / WordPiece tokenizer
│   │   └── training/       # Model training loop, loss metrics & checkpoints
│   ├── api/                # FastAPI router (auth, chat, feedback, health, lead, model)
│   ├── database/           # MySQL connection pool, schemas, models & persistence
│   ├── dataset/            # Domain knowledge JSONs (services, pricing, FAQ, team, etc.)
│   ├── genkit-model/       # Model configs, vocabulary & trainable weights
│   ├── utils/              # Dataset generator, helper utilities & production loggers
│   ├── chatbot.py          # Master RAG & LLM orchestration engine
│   ├── config.py           # Application configurations & environment setup
│   ├── evaluate.py         # Pipeline evaluation script
│   ├── main.py             # FastAPI entrypoint server
│   ├── predict.py          # Interactive terminal CLI chat interface
│   ├── train.py            # Model training entrypoint script
│   ├── .env.example        # Environment variable template
│   └── requirements.txt    # Production Python dependencies
│
└── frontend/               # React + Vite Chat Application
    ├── public/             # Static public assets
    ├── src/
    │   ├── assets/         # Branding icons & vector images
    │   ├── components/
    │   │   └── ChatWidget/ # Modular React Chat components (Header, Input, Messages, etc.)
    │   ├── hooks/          # Custom React hooks (useChat, useAutoGrow)
    │   ├── services/       # Axios API client integrations
    │   ├── utils/          # Markdown rendering & formatting helpers
    │   ├── App.jsx         # Root application component
    │   ├── index.css       # Design tokens, CSS variables & animations
    │   └── main.jsx        # React application entrypoint
    ├── index.html          # Application HTML shell
    ├── package.json        # Node.js dependencies & scripts
    └── vite.config.js      # Vite dev server configuration & API proxy rules
```

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.9+** (PyTorch, FastAPI, NLTK, Scikit-learn, etc.)
- **Node.js 18+** & **npm**
- **MySQL Server 8.0+** (Required for database persistence; can be configured in `.env`)

---

## ⚡ Quick Start Guide

### 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to update MySQL credentials, host, and port.*

5. Launch the FastAPI backend server:
   ```bash
   python main.py
   ```
   - Server running at: `http://localhost:8000`
   - Interactive Swagger API Docs: `http://localhost:8000/docs`

---

### 2. Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   - Application running at: `http://localhost:5173`
   - Vite automatically proxies API requests (`/api/*`) to `http://localhost:8000`.

---

## 🛠️ CLI Predictor & Evaluation

### Interactive CLI Chat
To test the full RAG & Model pipeline directly in your terminal:
```bash
cd backend
python predict.py
```

### Dataset Generation & Training
To generate synthetic training datasets and train/fine-tune the custom Causal GPT model:
```bash
cd backend
python utils/dataset_generator.py
python train.py
```

### RAG Evaluation Suite
To run pipeline evaluation metrics across accuracy, RAG context precision, and response quality:
```bash
cd backend
python evaluate.py
```

---

## ⚙️ Environment Variables

The backend uses a `.env` file (copied from `.env.example`). Key configuration parameters:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DEBUG` | `True` | Enables debug logging and automatic reloads |
| `HOST` | `0.0.0.0` | API server host address |
| `PORT` | `8000` | API server port |
| `MYSQL_HOST` | `localhost` | MySQL database host address |
| `MYSQL_PORT` | `3306` | MySQL server port |
| `MYSQL_DATABASE` | `genkit_ai` | Database name |
| `MYSQL_USER` | `root` | Database username |
| `MYSQL_PASSWORD` | `Admin@123` | Database password |

---

## 📡 API Endpoints Summary

- `POST /api/chat`: Primary endpoint processing chat queries through the RAG & GPT model pipeline.
- `POST /api/lead`: Submits contact/lead details captured from the floating widget.
- `POST /api/feedback`: Stores user feedback (thumbs up/down) for specific response IDs.
- `GET /api/health`: Health status endpoint returning GPU detection, system memory, and database status.
- `GET /api/model`: Returns model details, vocabulary size, and active hyperparameters.

---

## 🛡️ Git & Security Policy

- Sensitive credentials (`.env`), heavy model weights (`*.pt`, `checkpoint.pt`), temporary cache, and node modules (`node_modules/`) are strictly ignored via `.gitignore`.
- Model metadata configs (`config.json` and `vocab.json`) remain version-controlled to allow instant setup.
