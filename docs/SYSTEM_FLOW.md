# 🔄 GENKIT AI — End-to-End System Flow & Sequence Diagrams

## Query Execution Lifecycle

```text
[ User UI ] ──► [ Security Gating ] ──► [ Domain Guard ] ──► [ Coreference ]
                                                                   │
                                                                   ▼
[ React Stream ] ◄── [ PyTorch GQA GPT ] ◄── [ Tokenizer ] ◄── [ RAG & GraphRAG ]
```

### Sequence Steps
1. **User Action**: Client submits query via React frontend (`ChatInput.jsx`).
2. **Security Gating**: Input sanitized for XSS and SQLi; scanned for prompt injection attacks.
3. **Domain Safety Check**: Dual-pass centroid cosine classifier checks domain fit ($\ge 0.22$). Out-of-domain queries trigger immediate refusal.
4. **Coreference Resolution**: Pronouns ("it", "they", "that service") are resolved using entity memory history.
5. **Hybrid Retrieval**: Sparse BM25 + PyTorch INT8 HNSW dense vector search executed in parallel; merged via Reciprocal Rank Fusion (RRF).
6. **GraphRAG Sub-Graph Extraction**: BFS entity relation search extracts facts up to depth $h=2$.
7. **Neural Reranking**: Candidate passages ordered by cross-attention score.
8. **Prompt Compilation**: Structured prompt created with `<context_start>`, `<query_start>`, and `<ans_start>` control tags.
9. **Byte-Fallback Tokenization**: Text converted to Int64 tensor tokens (0% `<unk>`).
10. **Causal Transformer Generation**: Logits generated via GQA, RoPE, RMSNorm, SwiGLU, and Paged KV-Cache.
11. **SSE Token Streaming**: Tokens streamed real-time to React client; turn persisted asynchronously to MySQL.
