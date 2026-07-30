# 🔍 GENKIT AI — Operational Troubleshooting & Diagnostics

## Common Failure Diagnoses

### 1. Database Connection Warning
- **Symptom**: `MySQL Database connection unavailable. Running in in-memory session mode.`
- **Fix**: Ensure MySQL service is running on port 3306 or launch via `docker-compose up -d mysql_db`.

### 2. Module Import Errors
- **Symptom**: `ModuleNotFoundError: No module named 'app'`
- **Fix**: Run commands from `backend/` directory or run `python backend/main.py` from project root.
