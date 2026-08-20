# Genkit AI — Enterprise Production Deployment Guide

This comprehensive guide explains how to deploy, configure, train, monitor, and maintain the **Genkit AI** enterprise assistant in production.

---

## 1. Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- (Optional) NVIDIA CUDA 12+ GPU with PyTorch support
- (Optional) MySQL 8.0+

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/Skishore24/Chatbot-own-model.git
cd Chatbot-own-model

# Create and activate Python virtual environment
python -m venv backend/venv
source backend/venv/bin/activate  # On Linux/macOS
# .\backend\venv\Scripts\activate  # On Windows

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm ci
npm run dev
```

---

## 2. Model Training

The neural model is a custom decoder-only GPT architecture with RMSNorm, RoPE, Grouped-Query Attention (GQA), and SwiGLU feed-forward networks.

To train the model from scratch on verified Genkit domain data:
```bash
python backend/train.py
```
Or with custom arguments:
```bash
python backend/training/train_model.py --epochs 60 --batch_size 4 --lr 0.0003
```

---

## 3. Model Checkpoint Creation

During training, the system automatically validates checkpoints and saves:
- `backend/genkit-model/model_v6.pt` (PyTorch state dictionary & architecture weights)
- `backend/genkit-model/bpe_tokenizer_v6.json` (Byte-Fallback BPE vocabulary & merges)
- `backend/genkit-model/config_v6.json` (Hyperparameters configuration)

> **Checkpoint Safety**: If no checkpoint exists, the application safely operates in deterministic, verified RAG-grounded mode. It will never output random text from untrained weights.

---

## 4. Environment Variables

Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Key environment configurations:
| Variable | Default | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `Genkit AI Assistant` | Application name displayed in health endpoints |
| `ENVIRONMENT` | `production` | `development`, `staging`, or `production` |
| `DEBUG` | `false` | Enable/disable verbose debug output |
| `HOST` | `0.0.0.0` | Binding host address |
| `PORT` | `8000` | Port for FastAPI server |
| `SECRET_KEY` | `secure-random-key` | 32-byte secret key for security tokens |
| `ALLOWED_ORIGINS` | `http://localhost:5173,https://genkit.in` | Allowed CORS origins (comma-separated) |
| `DEVICE` | `cpu` | `cuda` for GPU or `cpu` for CPU inference |
| `MYSQL_HOST` | `localhost` | MySQL host address |
| `MYSQL_DATABASE` | `genkit_ai` | Database name |
| `MYSQL_USER` | `root` | Database username |
| `MYSQL_PASSWORD` | `""` | Database password |

---

## 5. Docker Build

Build the production container image from the repository root:
```bash
docker build -f docker/Dockerfile -t genkit-backend:latest .
```

---

## 6. Docker Run

Run the container in detached mode:
```bash
docker run -d \
  --name genkit-ai-backend \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  genkit-backend:latest
```

---

## 7. CPU Deployment

By default, the Docker image and backend seamlessly run on standard CPU instances without requiring CUDA packages or GPU drivers.
- Recommended CPU Specs: 2+ vCPU, 4GB+ RAM.
- Startup is instant, utilizing PyTorch CPU inference and local inverted BM25/TF-IDF indexing.

---

## 8. GPU Deployment (CUDA)

For hardware acceleration with NVIDIA GPUs (e.g. RTX 3050, T4, A10G):
1. Install NVIDIA Container Toolkit on host.
2. Run container with `--gpus all`:
```bash
docker run -d \
  --gpus all \
  --name genkit-ai-gpu \
  -p 8000:8000 \
  --env-file .env \
  genkit-backend:latest
```
PyTorch will automatically detect `torch.cuda.is_available() == True` and allocate weights to CUDA memory.

---

## 9. Database Setup (MySQL)

1. Create the MySQL database:
```sql
CREATE DATABASE IF NOT EXISTS genkit_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```
2. Configure credentials in `.env`:
```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=genkit_ai
MYSQL_USER=genkit_user
MYSQL_PASSWORD=your_secure_password
```
3. The application automatically initializes tables (`chat_sessions`, `chat_messages`, `leads`, `feedback`) upon first successful connection.
4. If MySQL is offline, the app operates gracefully without crashing.

---

## 10. Frontend Deployment

Build the static React bundle:
```bash
cd frontend
npm ci
npm run build
```
Deploy the resulting `frontend/dist/` directory using Nginx, Cloudflare Pages, Vercel, or AWS S3 + CloudFront.

Set the API URL during build:
```bash
VITE_API_URL=https://api.genkit.in npm run build
```

---

## 11. CORS Configuration

Set `ALLOWED_ORIGINS` in `.env` to restrict cross-origin access:
```env
ALLOWED_ORIGINS=https://genkit.in,https://www.genkit.in
```

---

## 12. Domain Configuration & Reverse Proxy (Nginx)

Example Nginx configuration:
```nginx
server {
    listen 80;
    server_name api.genkit.in;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.genkit.in;

    ssl_certificate /etc/letsencrypt/live/api.genkit.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.genkit.in/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Server-Sent Events (SSE) buffering disabled
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

---

## 13. HTTPS Setup

Generate SSL certificates with Certbot:
```bash
sudo certbot --nginx -d api.genkit.in
```

---

## 14. Health Checks

- Root Info: `GET http://localhost:8000/`
- Standard Health: `GET http://localhost:8000/health`
- API v1 Health: `GET http://localhost:8000/api/v1/health`

Example Health Check Response:
```json
{
  "status": "healthy",
  "application": "Genkit AI Assistant",
  "version": "6.0.0",
  "model": {
    "status": "ready",
    "device": "cpu",
    "checkpoint_exists": true,
    "vocab_size": 10000
  },
  "rag": {
    "status": "ready",
    "documents": 12
  },
  "database": {
    "status": "ready",
    "database": "genkit_ai"
  }
}
```

---

## 15. Monitoring

Monitor container resource usage:
```bash
docker stats genkit-ai-backend
```

---

## 16. Logs

View real-time application logs:
```bash
docker logs -f genkit-ai-backend
```
Backend logs are also written to `backend/logs/`.

---

## 17. Rollback

To rollback to a previous Docker container release:
```bash
docker stop genkit-ai-backend
docker rm genkit-ai-backend
docker run -d --name genkit-ai-backend -p 8000:8000 --env-file .env genkit-backend:previous-tag
```

---

## 18. Troubleshooting

1. **Model Checkpoint Incompatible / Missing**:
   - Status will show `MODEL_NOT_TRAINED` or `MODEL_INCOMPATIBLE`.
   - Run `python backend/train.py` to regenerate weights and vocabulary.
2. **Database Connection Error**:
   - Verify MySQL host, port, user, and password in `.env`.
   - The application will continue serving chats with RAG in offline mode.
3. **CORS Errors in Frontend**:
   - Add frontend domain to `ALLOWED_ORIGINS` in `.env`.
4. **SSE Streaming Broken Behind Proxy**:
   - Ensure `proxy_buffering off;` and `X-Accel-Buffering: no` headers are present.
