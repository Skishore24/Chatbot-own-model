# 🐳 GENKIT AI — Deployment & Infrastructure Manual

## Docker Multi-Stage Build

The application uses multi-stage Docker builds to keep production image sizes small and secure:
1. `builder`: Installs C++ compilation tools and Python packages.
2. `runtime`: Copies virtual environment and application code, running under a non-root user.

## Production Microservices Commands

```bash
# Build and start services in background
docker-compose up --build -d

# Inspect service logs
docker-compose logs -f genkit-backend

# Stop microservices stack
docker-compose down
```
