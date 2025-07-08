from typing import Any, List, Dict
from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool


class MilvusInsertTool(MilvusBaseTool, Tool):
    """Milvus 数据插入工具"""
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = tool_parameters.get("collection_name")
            data = tool_parameters.get("data")
            
            if not collection_name:
                raise ValueError("Collection name is required")
            
            if not self._validate_collection_name(collection_name):
                raise ValueError("Invalid collection name format")
            
            if not data:
                raise ValueError("Data is required")
            
            # 解析数据
            parsed_data = self._parse_insert_data(data)
            
            with self._get_milvus_client(self.runtime.credentials) as client:
                # 检查集合是否存在
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist")
                
                result = self._perform_insert(client, collection_name, parsed_data, tool_parameters)
                yield from self._create_success_message(result)
                
        except Exception as e:
            yield from self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> Generator[ToolInvokeMessage]:
        """统一的错误处理"""
        error_msg = str(error)
        yield self.create_json_message({
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        })
    
    def _create_success_message(self, data: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """创建成功响应消息"""
        response = {
            "success": True,
            **data
        }
        yield self.create_json_message(response)
    
    def _parse_insert_data(self, data: str) -> List[Dict[str, Any]]:
        """解析插入数据"""
        try:
            import json
            
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
            
            if not isinstance(parsed, list):
                raise ValueError("Data must be a list of entities")
            
            if not parsed:
                raise ValueError("Data cannot be empty")
            
            # 验证数据结构
            for i, entity in enumerate(parsed):
                if not isinstance(entity, dict):
                    raise ValueError(f"Entity at index {i} must be a dictionary")
                
                if not entity:
                    raise ValueError(f"Entity at index {i} cannot be empty")
            
            return parsed
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    def _perform_insert(self, client, collection_name: str, data: List[Dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        """执行数据插入"""
        # 获取分区名称（可选）
        partition_name = params.get("partition_name")
        
        # 执行插入
        try:
            result = client.insert(
                collection_name=collection_name,
                data=data,
                partition_name=partition_name
            )
            
            # 处理返回结果 - HTTP API 返回格式
            insert_count = len(data)  # HTTP API 不返回计数，用数据长度
            ids = result.get("data", {}).get("insertIds", []) if result else []
            
            # 获取插入的向量维度信息（如果有向量字段）
            vector_info = self._analyze_vector_data(data)
            
            return {
                "operation": "insert",
                "collection_name": collection_name,
                "partition_name": partition_name,
                "insert_count": insert_count,
                "ids": ids,
                "vector_info": vector_info,
                "data_preview": data[:3] if len(data) > 3 else data  # 显示前3条数据预览
            }
            
        except Exception as e:
            raise ValueError(f"Insert failed: {str(e)}")
    
    def _analyze_vector_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析向量数据信息"""
        vector_info = {
            "has_vector": False,
            "vector_fields": [],
            "dimensions": {}
        }
        
        if not data:
            return vector_info
        
        # 分析第一条数据来确定向量字段
        first_entity = data[0]
        
        for field_name, field_value in first_entity.items():
            if isinstance(field_value, list) and field_value:
                # 检查是否是数字列表（向量）
                if all(isinstance(x, (int, float)) for x in field_value):
                    vector_info["has_vector"] = True
                    vector_info["vector_fields"].append(field_name)
                    vector_info["dimensions"][field_name] = len(field_value)
        
        return vector_info 