# Milvus Plugin for Dify

A plugin that integrates Milvus vector database with the Dify platform, providing vector operations for collection management, data insertion, search, and querying.

[中文文档](./README_zh.md) | [English](./README.md)

## Features

### 🗂️ Collection Management
- **List Collections**: View all available collections
- **Describe Collection**: Get detailed collection information
- **Collection Stats**: Retrieve collection statistics
- **Check Existence**: Verify if collections exist

### 📥 Data Operations
- **Insert Data**: Add vectors and metadata to collections
- **Upsert Data**: Insert or update existing data
- **Query Data**: Retrieve data by ID or filter conditions
- **Delete Data**: Remove data from collections
Note: vectors must be precomputed by the caller before insertion. Provide vectors directly in your entities (e.g., `{ "vector": [0.1, 0.2, ...] }`).

### 🔍 Vector Search
- **Similarity Search**: Find similar vectors using various metrics
- **Filtered Search**: Combine vector similarity with metadata filters
- **Multi-Vector Search**: Search with multiple query vectors
- **Custom Parameters**: Adjust search behavior parameters

### 🔀 Hybrid Search (V2)
- Combine results from multiple vector fields in one request and rerank with a strategy.
- Endpoint: `/v2/vectordb/entities/hybrid_search` (the plugin auto-injects `dbName`).
- Tool: Milvus Hybrid Search (V2).

Parameters (Dify tool form)
- `collection_name` (string, required)
- `searches_json` (string, required): JSON array of search objects with precomputed vectors. Each object should include at least:
  - `data`: list of embeddings (precomputed), e.g. `[[0.1, 0.2, ...]]`
  - `annsField`: target vector field name
  - `limit`: per-route top-K
  - Optional per-route keys supported by REST API: `outputFields`, `metricType`, `filter`, `params`, `radius`, `range_filter`, `ignoreGrowing`
- `rerank_strategy` (select): `rrf` or `weighted`
- `rerank_params` (string): JSON. For `rrf`: `{ "k": 10 }`; for `weighted`: `{ "weights": [0.6, 0.4] }`
- Optional top-level: `limit`, `offset`, `output_fields` (comma sep), `partition_names` (comma sep), `consistency_level`, `grouping_field`, `group_size`, `strict_group_size` (boolean), `function_score` (string JSON)

Built-in validation
- Each search item must have `annsField` (non-empty) and `limit` (> 0 integer), and provide `data` (non-empty array) with numeric vectors.
- If `rerank_strategy = weighted`, `rerank_params.weights` must be numeric and its length must equal the number of search routes.
- If top-level `limit` is provided, ensure `limit + offset < 16384` (API limit).

Example
```
searches_json = [
  {"data": [[0.6734, 0.7392]], "annsField": "float_vector_1", "limit": 10, "outputFields": ["*"]},
  {"data": [[0.0753, 0.9971]], "annsField": "float_vector_2", "limit": 10, "outputFields": ["*"]}
]
rerank_strategy = rrf
rerank_params = {"k": 10}
limit = 3
output_fields = user_id,word_count,book_describe
```

## Installation & Configuration

### Connection Configuration
Configure your Milvus connection in the Dify platform:

- **URI**: Milvus server address (e.g., `http://localhost:19530`)
- **Token**: Authentication token (optional, format: `username:password`)
- **Database**: Target database name (default: `default`)


## License

MIT License
