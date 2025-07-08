# Milvus Plugin for Dify

集成 Milvus 向量数据库与 Dify 平台的插件，提供集合管理、数据插入、搜索和查询等向量操作功能。

## 功能特性

### 🗂️ 集合管理
- **列出集合**: 查看所有可用集合
- **描述集合**: 获取集合详细信息
- **集合统计**: 获取集合统计数据
- **检查存在**: 验证集合是否存在

### 📥 数据操作
- **插入数据**: 向集合添加向量和元数据
- **更新数据**: 插入或更新现有数据
- **查询数据**: 通过ID或过滤条件检索数据
- **删除数据**: 从集合中删除数据

### 🔍 向量搜索
- **相似性搜索**: 使用各种度量标准查找相似向量
- **过滤搜索**: 结合向量相似性和元数据过滤
- **多向量搜索**: 使用多个查询向量搜索
- **自定义参数**: 调整搜索行为参数

## 安装配置

### 连接配置
在 Dify 平台中配置 Milvus 连接：

- **URI**: Milvus 服务器地址 (例如: `http://localhost:19530`)
- **Token**: 认证令牌 (可选，格式: `username:password`)
- **Database**: 目标数据库名称 (默认: `default`)

## 使用示例

### 集合操作
```python
# 列出所有集合
{"operation": "list"}

# 描述集合
{"operation": "describe", "collection_name": "my_collection"}

# 创建集合
{"operation": "create", "collection_name": "my_collection"}
```

### 数据操作
```python
# 插入数据
{
  "collection_name": "my_collection",
  "data": [{"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": "sample"}]
}

# 向量搜索
{
  "collection_name": "my_collection",
  "query_vector": [0.1, 0.2, 0.3],
  "limit": 10
}
```

## 许可证

MIT License 