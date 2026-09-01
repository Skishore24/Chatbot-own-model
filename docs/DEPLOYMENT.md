# Genkit AI — Production Deployment Guide

## Prerequisites
- Docker & Docker Compose installed
- NVIDIA Container Toolkit (optional for GPU-accelerated Docker inference)
- Node.js 20+ & Python 3.10+ (for bare-metal execution)

---

## 1. Production Deployment with Docker Compose

### Step 1: Configure Environment Variables
Copy and customize `.env.example`:
```bash
cp .env.example .env
```
Ensure you provide strong production credentials:
```env
ENVIRONMENT=production
SECRET_KEY=your-secure-32-character-secret-key-here
MYSQL_ROOT_PASSWORD=your_mysql_root_password
MYSQL_PASSWORD=your_mysql_user_password
VITE_API_URL=https://api.yourdomain.com
```

### Step 2: Launch Docker Compose
```bash
docker-compose -f docker-compose.prod.yml up -d --build
```

### Step 3: Verify Container Health
```bash
docker-compose ps
curl -f http://localhost:8000/api/v1/health
```

---

## 2. Bare-Metal Linux / Windows Server Deployment

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Verify model checkpoint integrity
python scripts/verify_checkpoint.py

# Launch FastAPI backend with Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend Setup
```bash
cd frontend
npm ci
npm run build

# Serve dist/ with Nginx or static server
```

---

## 3. Reverse Proxy Configuration (Nginx)

Place `nginx/frontend.conf` in `/etc/nginx/conf.d/genkit.conf`:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        root /var/www/genkit/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
```
