from typing import Any, List, Union
from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool


class MilvusDeleteTool(MilvusBaseTool, Tool):
    """Milvus 数据删除工具"""
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            # 功能暂未实现
            raise ValueError("Delete operation is not yet implemented. This feature is under development.")
            
            # 以下代码待实现
            # collection_name = tool_parameters.get("collection_name")
            # 
            # if not collection_name:
            #     raise ValueError("Collection name is required")
            # 
            # if not self._validate_collection_name(collection_name):
            #     raise ValueError("Invalid collection name format")
            # 
            # with self._get_milvus_client(self.runtime.credentials) as client:
            #     # 检查集合是否存在
            #     if not client.has_collection(collection_name):
            #         raise ValueError(f"Collection '{collection_name}' does not exist")
            #     
            #     result = self._perform_delete(client, collection_name, tool_parameters)
            #     yield from self._create_success_message(result)
                
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
    
    def _perform_delete(self, client, collection_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """执行数据删除"""
        # 获取删除参数
        ids = params.get("ids")
        filter_expr = params.get("filter")
        
        # 必须提供 ids 或 filter 之一
        if not ids and not filter_expr:
            raise ValueError("Either 'ids' or 'filter' must be provided")
        
        # 如果两者都提供，优先使用 ids
        if ids and filter_expr:
            filter_expr = None
        
        # 获取分区名称（可选）
        partition_name = params.get("partition_name")
        
        # 解析 ids 参数
        if ids:
            ids = self._parse_ids(ids)
        
        try:
            # 执行删除 - 使用 HTTP API
            delete_result = client.delete(
                collection_name=collection_name,
                ids=ids if ids else None,
                filter=filter_expr,
                partition_name=partition_name
            )
            
            # 处理删除结果 - HTTP API 返回格式
            delete_count = 0
            deleted_ids = None
            
            if delete_result and "data" in delete_result:
                data = delete_result["data"]
                delete_count = data.get("deleteCount", 0)
                deleted_ids = data.get("deleteIds", [])
            else:
                # 如果没有返回详细信息，估算删除数量
                delete_count = len(ids) if ids else 0
            
            return {
                "operation": "delete",
                "collection_name": collection_name,
                "delete_type": "by_ids" if ids else "by_filter",
                "ids": ids,
                "filter": filter_expr,
                "partition_name": partition_name,
                "delete_count": delete_count,
                "deleted_ids": deleted_ids
            }
            
        except Exception as e:
            raise ValueError(f"Delete failed: {str(e)}")
    
    def _parse_ids(self, ids: Union[str, List]) -> List:
        """解析 ID 列表"""
        if isinstance(ids, str):
            try:
                import json
                parsed_ids = json.loads(ids)
            except json.JSONDecodeError:
                # 尝试按逗号分隔解析
                parsed_ids = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]
        else:
            parsed_ids = ids
        
        if not isinstance(parsed_ids, list):
            raise ValueError("IDs must be a list")
        
        if not parsed_ids:
            raise ValueError("IDs list cannot be empty")
        
        # 尝试转换为数字（如果可能）
        converted_ids = []
        for id_val in parsed_ids:
            if isinstance(id_val, str) and id_val.isdigit():
                converted_ids.append(int(id_val))
            else:
                converted_ids.append(id_val)
        
        return converted_ids 