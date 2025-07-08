# Milvus Plugin 部署和测试指南

## 🚀 部署方式

### 方式一：本地调试（推荐）

1. **设置环境变量**
   ```bash
   cp .env.example .env
   ```

2. **编辑 .env 文件**
   ```bash
   # 远程调试模式
   INSTALL_METHOD=remote
   REMOTE_INSTALL_URL=debug.dify.ai:5003  # 或您的 Dify 实例地址
   REMOTE_INSTALL_KEY=your_debug_key_here  # 从 Dify 插件管理页面获取
   ```

3. **获取调试密钥**
   - 登录您的 Dify 实例
   - 进入插件管理页面
   - 点击右上角的调试按钮（🐛图标）
   - 复制调试密钥

4. **启动插件**
   ```bash
   python -m main
   ```

### 方式二：打包部署

1. **创建插件包**
   ```bash
   # 确保所有文件都在项目根目录
   tar -czf milvus-plugin.tar.gz .
   ```

2. **上传到 Dify**
   - 在 Dify 插件管理页面
   - 选择"上传插件"
   - 上传打包的文件

## 🧪 测试步骤

### 1. 准备 Milvus 实例

#### 本地 Milvus (Docker)
```bash
# 下载 docker-compose.yml
curl -O https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml

# 启动 Milvus
docker-compose -f milvus-standalone-docker-compose.yml up -d

# 验证运行状态
docker-compose -f milvus-standalone-docker-compose.yml ps
```

#### 云端 Milvus (Zilliz Cloud)
1. 注册 [Zilliz Cloud](https://cloud.zilliz.com/)
2. 创建免费集群
3. 获取连接信息

### 2. 在 Dify 中配置插件

1. **连接配置**
   - URI: `http://localhost:19530` (本地) 或云端地址
   - Token: 留空 (本地) 或提供认证信息
   - Database: `default`

2. **测试连接**
   - 保存配置后会自动验证连接

### 3. 功能测试

#### 测试 1: 创建集合
```yaml
工具: Milvus Collection Manager
参数:
  operation: create
  collection_name: test_collection
  dimension: 384
  metric_type: COSINE
  auto_id: true
  description: "测试集合"
```

#### 测试 2: 插入数据
```yaml
工具: Milvus Data Insert
参数:
  collection_name: test_collection
  data: |
    [
      {
        "vector": [0.1, 0.2, 0.3, ...],  # 384维向量
        "text": "这是第一个测试文档",
        "category": "test"
      },
      {
        "vector": [0.4, 0.5, 0.6, ...],  # 384维向量
        "text": "这是第二个测试文档", 
        "category": "test"
      }
    ]
```

#### 测试 3: 向量搜索
```yaml
工具: Milvus Vector Search
参数:
  collection_name: test_collection
  query_vector: "[0.1, 0.2, 0.3, ...]"  # 384维查询向量
  limit: 5
  output_fields: "text,category"
  metric_type: COSINE
```

#### 测试 4: 数据查询
```yaml
工具: Milvus Data Query
参数:
  collection_name: test_collection
  filter: 'category == "test"'
  output_fields: "*"
  limit: 10
```

## 🔍 故障排除

### 常见问题

1. **连接失败**
   ```
   问题: Failed to connect to Milvus
   解决: 检查 Milvus 是否运行，URI 是否正确
   ```

2. **认证失败**
   ```
   问题: Authentication failed
   解决: 检查用户名密码或 API 密钥是否正确
   ```

3. **集合不存在**
   ```
   问题: Collection 'xxx' does not exist
   解决: 先创建集合再进行其他操作
   ```

4. **维度不匹配**
   ```
   问题: Vector dimension mismatch
   解决: 确保向量维度与集合定义一致
   ```

### 调试技巧

1. **查看日志**
   - 在 Dify 插件管理页面查看插件日志
   - 检查错误消息和堆栈跟踪

2. **逐步测试**
   - 先测试集合管理功能
   - 再测试数据插入功能
   - 最后测试搜索功能

3. **验证数据**
   - 使用 Milvus 官方客户端验证数据
   - 检查集合统计信息

## 📊 性能建议

### 数据插入
- 批量插入而非单条插入
- 单次插入建议不超过 1000 条记录
- 大量数据可分批处理

### 向量搜索
- 合理设置 limit 值 (建议 ≤ 100)
- 使用过滤条件减少搜索范围
- 根据需求调整搜索精度级别

### 集合设计
- 选择合适的索引类型
- 考虑数据分区策略
- 定期清理无用数据

## 🔗 相关资源

- [Milvus 官方文档](https://milvus.io/docs)
- [Dify 插件开发指南](https://docs.dify.ai/plugins)
- [PyMilvus API 参考](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md) 