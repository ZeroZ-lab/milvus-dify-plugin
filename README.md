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

### 🔍 Vector Search
- **Similarity Search**: Find similar vectors using various metrics
- **Filtered Search**: Combine vector similarity with metadata filters
- **Multi-Vector Search**: Search with multiple query vectors
- **Custom Parameters**: Adjust search behavior parameters

## Installation & Configuration

### Connection Configuration
Configure your Milvus connection in the Dify platform:

- **URI**: Milvus server address (e.g., `http://localhost:19530`)
- **Token**: Authentication token (optional, format: `username:password`)
- **Database**: Target database name (default: `default`)

## Usage Examples

### Collection Operations
```python
# List all collections
{"operation": "list"}

# Describe collection
{"operation": "describe", "collection_name": "my_collection"}

# Create collection
{"operation": "create", "collection_name": "my_collection"}
```

### Data Operations
```python
# Insert data
{
  "collection_name": "my_collection",
  "data": [{"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": "sample"}]
}

# Vector search
{
  "collection_name": "my_collection",
  "query_vector": [0.1, 0.2, 0.3],
  "limit": 10
}
```

## License

MIT License



