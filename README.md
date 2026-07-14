# Genkit AI Chatbot - Custom GPT Model with GPU Acceleration

An end-to-end interactive chatbot built with a custom-trained Causal Transformer (GPT) model and a Retrieval-Augmented Generation (RAG) system using TF-IDF. The system incorporates SQLite for conversation caching, lead capturing, and feedback logging, and features a premium floating chat widget frontend.

---

## 📂 Project Directory Structure

```text
Chatbot-own-model/
├── backend/
│   ├── genkit-model/     # Holds trained model.pt, config.json, vocab.json
│   ├── chatbot.py        # Core generation stream and pipeline orchestration
│   ├── config.py         # App configurations, paths, logging setup
│   ├── database.py       # SQL schemas, connection management, session histories
│   ├── dataset.json      # Structured QA data representing Genkit domain
│   ├── main.py           # FastAPI entrypoint, lifespans, schemas & routes
│   ├── ml_model.py       # Custom PyTorch GPT Architecture & simple word tokenizer
│   ├── requirements.txt  # Project dependencies list
│   ├── train.py          # Dataset expansion and GPU training loop
│   ├── vector_store.py   # Custom TF-IDF Vectorizer and RAG retriever
│   └── genkit.db         # Local SQLite database (auto-generated)
│
├── frontend/             # Interactive UI
│   ├── images/           # Image assets (e.g. logo1.png)
│   ├── index.html        # Floating chat widget layout
│   ├── script.js         # Real-time event streaming and form submissions
│   └── style.css         # Premium styling, animations, and typography
│
├── install_gpu.bat       # Automated Windows CUDA Setup [NEW]
└── README.md             # Project documentation [NEW]
```

---

## ⚡ Prerequisites

- **Python**: Version 3.10 to 3.14.
- **NVIDIA GPU**: Required for GPU acceleration (e.g., GeForce RTX 3050 Laptop).
- **GPU Drivers**: Make sure your NVIDIA drivers are up to date.

---

## 🚀 Quick Start Guide

### Step 1: Install CUDA-Enabled Dependencies
We have provided an automated Windows script to configure your environment:
1. Double-click the `install_gpu.bat` script at the root of the project.
2. It will automatically upgrade `pip`, install the correct GPU-enabled PyTorch build (`torch==2.13.0+cu126`), install all other dependencies from `backend/requirements.txt`, and output a device diagnostic.

> **Manual Install Command (Windows with CUDA 12.6):**
> If you prefer running commands manually in your shell:
> ```bash
> pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
> pip install -r backend/requirements.txt
> ```

### Step 2: Train the Custom GPT Model
Execute the training loop to parse `dataset.json`, train the custom tokenizer, and fit weights:
```bash
python backend/train.py
```
*Note: The script automatically detects the GPU and conducts training on the device (`cuda`).*

### Step 3: Run the FastAPI Server
Start the Uvicorn production server to host the API and serve the frontend:
```bash
python backend/main.py
```
The server will bind to `http://localhost:8000`.

### Step 4: Access the Frontend
Open `frontend/index.html` in any web browser to interact with your custom-trained AI assistant via the floating widget.

---

## 🛠️ GPU Verification & Diagnostics

To verify that your installation is correctly using the GPU, run:
```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
Expected output (if GPU installation succeeded):
```text
GPU Available: True
Device Name: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

## 🧬 Architectural Overview

*   **GPT Architecture (`backend/ml_model.py`)**: A mini-GPT architecture using Multi-Head Causal Self-Attention, Layer Normalization, GeLU activation, and dropout layers. Supports device portability (`idx.device` inference mapping).
*   **Vector Retrieval (`backend/vector_store.py`)**: A customized sparse TF-IDF Vectorizer built with NumPy. It matches user queries to the local knowledge base segments to fetch accurate context.
*   **SQLite Caching (`backend/database.py`)**: Tracks three relational tables with indexes for optimized lookup speeds:
    *   `chats`: Cache histories of prompt and generation interactions.
    *   `leads`: Captured names and email addresses.
    *   `feedback`: Ratings (1-5 stars) logged by visitors.
