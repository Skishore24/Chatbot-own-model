# FastAPI API Reference & Streaming Protocol

Base URL: `http://localhost:8000/api/v1` (with `/api/v5` alias support)

---

## 1. Endpoints Summary

| Method | Endpoint | Description | Auth / Rate Limit |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Server & hardware health status | None |
| `GET` | `/api/v1/health` | Detailed subsystem diagnostics | None |
| `GET` | `/api/v1/model` | Custom LLM architecture metadata | None |
| `POST` | `/api/v1/chat` | Synchronous grounded chat response | 60 req/min |
| `POST` | `/api/v1/chat/stream` | Real SSE token-by-token streaming | 60 req/min |
| `POST` | `/api/v1/leads` | Capture business lead from widget | None |
| `GET` | `/api/v1/history` | Retrieve chronological chat messages | None |

---

## 2. Server-Sent Events (SSE) Protocol

Endpoint: `POST /api/v1/chat/stream`

### Request Body
```json
{
  "message": "What services does Genkit provide?",
  "session_id": "user_session_abc123"
}
```

### Event Sequence

#### Event 1: `start`
```json
data: {
  "event": "start",
  "session_id": "user_session_abc123",
  "intent": "Services",
  "grounded": true,
  "sources": [
    {"id": "services_1", "title": "Web Development", "category": "Services", "score": 2.45}
  ]
}
```

#### Event 2..N: `token`
```json
data: {"event": "token", "chunk": "Genkit "}

data: {"event": "token", "chunk": "provides "}

data: {"event": "token", "chunk": "custom web and mobile development..."}
```

#### Event Final: `end`
```json
data: {
  "event": "end",
  "answer": "Genkit provides custom web and mobile development...",
  "confidence": 0.94,
  "latency_ms": 142.5
}
```

---

## 3. Lead Capture

Endpoint: `POST /api/v1/leads`

### Request Body
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "phone": "+1 (555) 019-2834",
  "message": "Interested in enterprise AI model training.",
  "session_id": "user_session_abc123"
}
```

### Response
```json
{
  "success": true,
  "message": "Thank you! Our Genkit team will reach out to you shortly.",
  "lead_id": 1
}
```
