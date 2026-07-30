# 📡 GENKIT AI — REST & SSE Streaming API Reference

## Endpoints Summary

### 1. Real-time Token Stream (SSE)
- **URL**: `POST /api/v5/chat/stream`
- **Content-Type**: `application/json`
- **Response**: `text/event-stream`
- **Payload**:
```json
{
  "message": "What are Genkit's AI services?",
  "session_id": "session_12345"
}
```

### 2. Synchronous Chat Query
- **URL**: `POST /api/v5/chat/query`
- **Content-Type**: `application/json`
- **Response**: `application/json`

### 3. Business Lead Submission
- **URL**: `POST /api/v5/lead`
- **Content-Type**: `application/json`

### 4. Health Check
- **URL**: `GET /api/v5/health`
- **Response**: `application/json`
