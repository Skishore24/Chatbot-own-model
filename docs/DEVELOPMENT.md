# Local Development & Testing Guide

## 1. Prerequisites

- Python 3.10+ (PyTorch with CUDA support recommended)
- Node.js 18+ (for React Vite frontend)
- MySQL Server (optional; automatically falls back to local SQLite `genkit.db`)

---

## 2. Environment Setup

### Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

---

## 3. Running Services

### Start Backend Server
```bash
cd backend
python app/main.py
# Server runs at http://localhost:8000
# OpenAPI Docs: http://localhost:8000/docs
```

### Start Frontend Dev Server
```bash
cd frontend
npm run dev
# Frontend runs at http://localhost:5173
```

---

## 4. Running Verification Tests

```bash
# Run unit & integration test suite
cd backend
python -m unittest discover -s tests

# Run benchmark evaluation suite
python evaluate.py
```
