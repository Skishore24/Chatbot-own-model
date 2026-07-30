# 📜 GENKIT AI — Version Release History

## [5.0.0] - 2026-07-30
### Added
- Complete Google/OpenAI Clean Architecture Rebuild.
- 16,000 Vocab Byte-Fallback BPE Tokenizer (0% `<unk>` tokens).
- Custom PyTorch Grouped-Query Attention (GQA) & Paged KV-Cache Transformer Decoder.
- GraphRAG BFS Sub-Graph Extraction ($h=2$).
- Parallel Sparse BM25 + PyTorch INT8 HNSW Vector Matrix Search with Reciprocal Rank Fusion (RRF).
- Server-Sent Events (SSE) Real-Time Token Streaming (`/api/v5/chat/stream`).
- 100% Async MySQL Connection Pool (`aiomysql`).
