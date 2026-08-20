# Multi-Stage Production Dockerfile for Genkit AI v6.0 Enterprise
FROM python:3.11-slim as builder

WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source
COPY backend/ /app/backend/

# Set Environment Variables
ENV PYTHONPATH=/app/backend
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
