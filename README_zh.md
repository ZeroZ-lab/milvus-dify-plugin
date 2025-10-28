# Milvus 插件 (Dify)

一个将 Milvus 向量数据库与 Dify 平台集成的插件，提供集合管理、数据插入、搜索和查询等向量操作功能。

[中文文档](./README_zh.md) | [English](./README.md)

## 功能特性

### 🗂️ 集合管理
- **列出集合**: 查看所有可用集合
- **描述集合**: 获取集合的详细信息
- **集合统计**: 检索集合统计数据
- **检查存在性**: 验证集合是否存在

### 📥 数据操作
- **插入数据**: 向集合中添加向量和元数据
- **更新插入数据**: 插入或更新现有数据
- **查询数据**: 通过ID或过滤条件检索数据
- **删除数据**: 从集合中移除数据

### 🔍 向量搜索
- **相似度搜索**: 使用各种指标查找相似向量
- **过滤搜索**: 结合向量相似度和元数据过滤
- **多向量搜索**: 使用多个查询向量进行搜索
- **自定义参数**: 调整搜索行为参数

### 🔀 混合搜索 (V2)
- 将多个向量字段的检索结果在一次请求中合并，并按策略进行重排。
- 接口路径：`/v2/vectordb/entities/hybrid_search`（插件会自动附加 `dbName`）。
- 工具名称：Milvus Hybrid Search (V2)。

参数（Dify 工具表单）
- `collection_name`（必填）
- `searches_json`（必填，字符串）：`search` 对象数组的 JSON。每个对象至少包含：
  - `data`：向量嵌入列表，如 `[[0.1, 0.2, ...]]`
  - `annsField`：目标向量字段名
  - `limit`：该路检索的 Top-K
  - 每路可选键（按 REST 文档透传）：`outputFields`、`metricType`、`filter`、`params`、`radius`、`range_filter`、`ignoreGrowing`
- `rerank_strategy`（下拉）：`rrf` 或 `weighted`
- `rerank_params`（字符串）：JSON。`rrf` 示例：`{"k": 10}`；`weighted` 示例：`{"weights": [0.6, 0.4]}`
- 顶层可选：`limit`、`offset`、`output_fields`（逗号分隔）、`partition_names`（逗号分隔）、`consistency_level`、`grouping_field`、`group_size`、`strict_group_size`（布尔）、`function_score`（字符串 JSON）

内置校验
- 每路检索必须包含 `annsField`（非空）、`data`（非空数组）、`limit`（> 0 的整数）。
- 当 `rerank_strategy = weighted` 时，`rerank_params.weights` 必须为数值数组，且长度与检索路数一致。
- 若提供顶层 `limit`，需确保 `limit + offset < 16384`（接口限制）。

示例
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

## 安装与配置

### 连接配置
在 Dify 平台中配置您的 Milvus 连接:

- **URI**: Milvus 服务器地址 (例如 `http://localhost:19530`)
- **Token**: 认证令牌 (可选，格式: `username:password`)
- **Database**: 目标数据库名称 (默认: `default`)

## 许可证

MIT 许可证 
