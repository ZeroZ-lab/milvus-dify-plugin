# 🚀 Milvus Plugin 快速开始指南

## ⚠️ 重要说明

**这个插件是为 Dify 平台设计的，无法在本地直接运行。** `dify_plugin` 模块只在 Dify 环境中可用。

## 📋 前置条件

1. **Dify 平台访问权限**
   - Dify Cloud 账号 或
   - 自部署的 Dify 实例

2. **Milvus 数据库实例**
   - 本地 Milvus (Docker) 或
   - Zilliz Cloud 账号

## 🎯 部署步骤

### 步骤 1: 准备 Milvus 实例

#### 选项 A: 本地 Milvus (推荐测试)
```bash
# 下载并启动 Milvus
curl -O https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml
docker-compose -f milvus-standalone-docker-compose.yml up -d

# 验证运行状态
docker-compose -f milvus-standalone-docker-compose.yml ps
```

#### 选项 B: Zilliz Cloud (推荐生产)
1. 访问 [Zilliz Cloud](https://cloud.zilliz.com/)
2. 注册账号并创建免费集群
3. 记录连接信息 (URI 和 Token)

### 步骤 2: 在 Dify 中部署插件

#### 方法 1: 调试模式 (开发推荐)

1. **获取调试密钥**
   - 登录 Dify 平台
   - 进入 "插件" → "插件管理"
   - 点击右上角的调试图标 🐛
   - 复制调试密钥

2. **配置环境**
   ```bash
   # 复制配置文件
   cp .env.example .env
   
   # 编辑配置
   nano .env
   ```
   
   填入以下内容：
   ```env
   INSTALL_METHOD=remote
   REMOTE_INSTALL_URL=debug.dify.ai:5003  # 或您的 Dify 实例地址
   REMOTE_INSTALL_KEY=your_debug_key_here  # 替换为实际的调试密钥
   ```

3. **启动调试**
   ```bash
   ./start_plugin.sh
   ```

#### 方法 2: 打包上传 (生产推荐)

1. **创建插件包**
   ```bash
   # 清理临时文件
   rm -f test_plugin.py test_basic.py start_plugin.sh
   
   # 创建压缩包
   tar --exclude='.git' --exclude='*.pyc' --exclude='__pycache__' \
       -czf milvus-plugin.tar.gz .
   ```

2. **上传插件**
   - 在 Dify 插件管理页面
   - 点击 "安装插件" → "上传插件包"
   - 选择 `milvus-plugin.tar.gz` 文件
   - 等待安装完成

### 步骤 3: 配置插件

1. **填写连接信息**
   - **Milvus URI**: 
     - 本地: `http://localhost:19530`
     - Zilliz Cloud: `https://your-cluster.api.gcp-us-west1.zillizcloud.com:19530`
   - **认证令牌**: 
     - 本地: 留空
     - Zilliz Cloud: `username:password` 格式
   - **数据库名称**: `default` (通常)

2. **测试连接**
   - 保存配置后会自动验证连接
   - 确保显示 "连接成功"

## 🧪 功能测试

### 测试 1: 创建集合
```yaml
工具: Milvus Collection Manager
操作: create
集合名称: demo_collection
向量维度: 768
距离度量: COSINE
自动ID: true
描述: "演示集合"
```

### 测试 2: 插入测试数据
```yaml
工具: Milvus Data Insert
集合名称: demo_collection
数据: |
  [
    {
      "vector": [0.1, 0.2, 0.3, ...],  # 768维向量
      "title": "测试文档1",
      "content": "这是第一个测试文档的内容"
    },
    {
      "vector": [0.4, 0.5, 0.6, ...],  # 768维向量
      "title": "测试文档2", 
      "content": "这是第二个测试文档的内容"
    }
  ]
```

### 测试 3: 向量搜索
```yaml
工具: Milvus Vector Search
集合名称: demo_collection
查询向量: "[0.1, 0.2, 0.3, ...]"  # 768维
限制数量: 5
输出字段: "title,content"
```

## 🔧 常见问题解决

### Q1: 本地无法运行插件
**A**: 这是正常的！插件必须在 Dify 环境中运行，本地只能进行代码检查。

### Q2: 连接 Milvus 失败
**A**: 检查以下项目：
- Milvus 服务是否运行
- URI 格式是否正确
- 网络连接是否正常
- 认证信息是否正确

### Q3: 向量维度错误
**A**: 确保：
- 插入的向量维度与集合定义一致
- 搜索向量维度与集合定义一致
- 所有向量都是数字列表

### Q4: 插件安装失败
**A**: 检查：
- 文件格式是否为 .tar.gz
- 文件大小是否超过限制
- manifest.yaml 格式是否正确

## 📚 下一步

1. **学习向量化**: 了解如何将文本/图像转换为向量
2. **集成 AI 模型**: 结合 Embedding 模型使用
3. **优化性能**: 根据数据量调整索引和搜索参数
4. **生产部署**: 配置高可用 Milvus 集群

## 🆘 获取帮助

- 查看 [详细文档](README.md)
- 参考 [部署指南](DEPLOYMENT.md)
- 访问 [Milvus 官方文档](https://milvus.io/docs)
- 查看 [Dify 插件文档](https://docs.dify.ai/plugins) 