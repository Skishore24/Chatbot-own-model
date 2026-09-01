# Genkit AI — Local Development Guide

## Environment Setup

### 1. Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## Running Automated Tests
```bash
cd backend
python -m pytest tests/ -v
```

---

## Running Benchmarks and Evaluation
```bash
cd backend
python scripts/evaluate.py
```

---

## Code Quality and Style Guidelines
- **Python**: Follow PEP 8 standards with explicit type annotations.
- **Async/Await**: Maintain asynchronous execution in all I/O bound endpoints.
- **SQL Security**: Use parameterized queries through `db_manager` or repositories.
- **No External AI APIs**: All model weights, inference algorithms, and RAG retrieval pipelines must remain 100% self-hosted on local hardware.
