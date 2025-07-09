from typing import Any, List, Dict, Optional
from collections.abc import Generator
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.entities.model.text_embedding import TextEmbeddingModelConfig
from dify_plugin.entities.model import ModelType
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)

class MilvusSearchTool(MilvusBaseTool, Tool):
    """Milvus 向量搜索工具"""

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """执行搜索工具"""
        try:
            # 解析和验证参数
            collection_name = tool_parameters.get("collection_name")
            vector_str = tool_parameters.get("query_vector")
            query_text = tool_parameters.get("query_text")
            embedding_model = tool_parameters.get("embedding_model")

            if not collection_name or not self._validate_collection_name(collection_name):
                raise ValueError("Invalid or missing collection name.")

            # 检查是否至少提供了一种查询方式
            if not vector_str and not query_text:
                raise ValueError("Either query vector or query text is required.")

            # 如果提供了查询文本，则将其转换为向量
            if query_text:
                # 检查是否提供了嵌入模型
                if not embedding_model and query_text:
                    raise ValueError("Embedding model is required when using query text.")
                
                logger.info(f"🔤 [INFO] Converting query text to vector: '{query_text[:50]}...'")
                vector_data = self._text_to_embedding(query_text, embedding_model)
            else:
                # 否则解析提供的向量数据
                try:
                    vector_data = self._parse_vector_data(str(vector_str))
                except ValueError as e:
                    raise ValueError(str(e))

            # 获取其他参数
            limit = int(tool_parameters.get("limit", 10))
            output_fields_str = tool_parameters.get("output_fields")
            filter_expr = tool_parameters.get("filter")
            search_params_str = tool_parameters.get("search_params")
            anns_field = tool_parameters.get("anns_field", "vector")

            # 准备参数
            search_params = self._parse_search_params(search_params_str)
            output_fields = [f.strip() for f in output_fields_str.split(',')] if output_fields_str else None

            # 执行搜索
            with self._get_milvus_client(self.runtime.credentials) as client:
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist.")

                logger.info(f"🔍 [DEBUG] Searching in collection '{collection_name}' with limit={limit}, anns_field='{anns_field}'")

                results = client.search(
                    collection_name=collection_name,
                    data=[vector_data],
                    anns_field=anns_field,
                    limit=limit,
                    output_fields=output_fields,
                    filter=filter_expr,
                    search_params=search_params,
                    partition_names=None # partition_names not supported in this tool
                )

                logger.info(f"✅ [DEBUG] Search completed. Found {len(results)} results.")
                
                response_data = {
                    "operation": "search",
                    "collection_name": collection_name,
                    "results": results,
                    "result_count": len(results)
                }
                yield from self._create_success_message(response_data)

        except Exception as e:
            yield from self._handle_error(e)

    def _handle_error(self, error: Exception) -> Generator[ToolInvokeMessage, None, None]:
        """统一的错误处理"""
        error_msg = str(error)
        yield self.create_json_message({
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        })

    def _create_success_message(self, data: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """创建成功响应消息"""
        response = {
            "success": True,
            **data
        }
        yield self.create_json_message(response)
        
    def _text_to_embedding(self, text: str, model_info: Any) -> List[float]:
        """将文本转换为嵌入向量
        
        使用Dify平台提供的文本嵌入功能将文本转换为向量。
        根据manifest.yaml中配置的text-embedding权限，Dify会自动提供嵌入功能。
        
        Args:
            text: 要转换为向量的文本
            model_info: 嵌入模型信息
        
        Returns:
            嵌入向量，表示为浮点数列表
        """
        try:
            logger.info(f"📊 [INFO] 请求文本嵌入，文本: '{text[:30]}...'")
            logger.info(f"🔧 [INFO] 使用指定的嵌入模型: {model_info}")
            
            # 从model_info中提取模型名称
            model_name = ""
            provider = ""
            
            if isinstance(model_info, dict):
                # 如果是字典，尝试获取model字段
                if "model" in model_info:
                    model_name = model_info["model"]
                # 尝试获取provider字段
                if "provider" in model_info:
                    provider = model_info["provider"]
            else:
                # 否则直接使用model_info作为模型名称
                model_name = str(model_info)
            
            # 如果没有提供provider，使用默认值
            if not provider:
                provider = "default"
                
            logger.info(f"🔧 [INFO] 提取的模型名称: {model_name}, 提供者: {provider}")
            
            # 创建TextEmbeddingModelConfig
            model_config = TextEmbeddingModelConfig(
                model=model_name,
                provider=provider,
                model_type=ModelType.TEXT_EMBEDDING
            )
            
            logger.info(f"🔧 [INFO] 创建的模型配置: {model_config}")
            
            # 调用text_embedding.invoke方法
            embedding_result = self.session.model.text_embedding.invoke(
                model_config=model_config,
                texts=[text]
            )
            
            logger.info(f"✅ [INFO] 嵌入结果类型: {type(embedding_result)}")
            
            # 检查结果
            if embedding_result and hasattr(embedding_result, 'embeddings') and embedding_result.embeddings:
                # 获取第一个文本的嵌入向量
                embedding_vector = embedding_result.embeddings[0]
                logger.info(f"✅ [INFO] 成功生成嵌入向量，维度: {len(embedding_vector)}")
                return embedding_vector
            else:
                logger.error("❌ [ERROR] 嵌入结果为空或格式不正确 (Debug for ai)")
                raise ValueError("无法为文本生成嵌入向量: 结果为空或格式不正确")
            
        except Exception as e:
            logger.error(f"❌ [ERROR] 文本嵌入失败: {str(e)}")
            raise ValueError(f"文本嵌入转换失败: {str(e)}")