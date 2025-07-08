from typing import Any, Optional
from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool


class MilvusSearchTool(MilvusBaseTool, Tool):
    """Milvus 向量搜索工具"""
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = tool_parameters.get("collection_name")
            query_vector = tool_parameters.get("query_vector")
            
            if not collection_name:
                raise ValueError("Collection name is required")
            
            if not self._validate_collection_name(collection_name):
                raise ValueError("Invalid collection name format")
            
            if not query_vector:
                raise ValueError("Query vector is required")
            
            # 解析查询向量
            if isinstance(query_vector, str):
                query_vector = self._parse_vector_data(query_vector)
            
            if not isinstance(query_vector, list):
                raise ValueError("Query vector must be a list")
            
            with self._get_milvus_client(self.runtime.credentials) as client:
                # 检查集合是否存在
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist")
                
                result = self._perform_search(client, collection_name, query_vector, tool_parameters)
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
    
    def _perform_search(self, client, collection_name: str, query_vector: list, params: dict[str, Any]) -> dict[str, Any]:
        """执行向量搜索"""
        # 获取搜索参数
        limit = params.get("limit", 10)
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 10
        
        if limit <= 0 or limit > 1000:
            limit = min(max(limit, 1), 1000)
        
        # 获取输出字段
        output_fields = params.get("output_fields")
        if output_fields and isinstance(output_fields, str):
            output_fields = [field.strip() for field in output_fields.split(",") if field.strip()]
        elif not output_fields:
            output_fields = None
        
        # 获取过滤条件
        filter_expr = params.get("filter")
        
        # 获取搜索参数
        search_params = self._build_search_params(params)
        
        # 获取分区名称
        partition_names = params.get("partition_names")
        if partition_names and isinstance(partition_names, str):
            partition_names = [name.strip() for name in partition_names.split(",") if name.strip()]
        
        # 获取向量字段名称
        anns_field = params.get("anns_field", "vector")
        
        # 执行搜索 - 使用 HTTP API
        search_results = client.search(
            collection_name=collection_name,
            data=[query_vector],  # 需要包装为列表
            anns_field=anns_field,
            limit=limit,
            output_fields=output_fields,
            filter=filter_expr,
            search_params=search_params,
            partition_names=partition_names
        )
        
        # 处理搜索结果 - HTTP API 返回格式不同
        results = []
        if search_results and len(search_results) > 0:
            # HTTP API 返回的格式：[[{"id": x, "distance": y, ...}]] 或 [{"id": x, "distance": y, ...}]
            search_data = search_results[0] if isinstance(search_results[0], list) else search_results
            
            for hit in search_data:
                if isinstance(hit, dict):
                    result_item = {
                        "id": hit.get("id"),
                        "distance": hit.get("distance"),
                        "score": hit.get("distance")  # 兼容性
                    }
                    
                    # 添加实体数据
                    if "entity" in hit:
                        result_item["entity"] = hit["entity"]
                    # 如果没有entity字段，直接添加其他字段
                    else:
                        for key, value in hit.items():
                            if key not in ["id", "distance"]:
                                result_item[key] = value
                    
                    results.append(result_item)
        
        return {
            "operation": "search",
            "collection_name": collection_name,
            "query_vector_dimension": len(query_vector),
            "anns_field": anns_field,
            "limit": limit,
            "filter": filter_expr,
            "search_params": search_params,
            "partition_names": partition_names,
            "results": results,
            "result_count": len(results)
        }
    
    def _build_search_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """构建搜索参数"""
        search_params = {}
        
        # 距离度量类型
        metric_type = params.get("metric_type")
        if metric_type:
            search_params["metric_type"] = metric_type
        
        # 搜索精度级别
        level = params.get("level")
        if level:
            try:
                level = int(level)
                if 1 <= level <= 5:
                    search_params["level"] = level
            except (ValueError, TypeError):
                pass
        
        # 范围搜索参数
        radius = params.get("radius")
        range_filter = params.get("range_filter")
        
        if radius is not None:
            try:
                search_params["radius"] = float(radius)
            except (ValueError, TypeError):
                pass
        
        if range_filter is not None:
            try:
                search_params["range_filter"] = float(range_filter)
            except (ValueError, TypeError):
                pass
        
        # 其他搜索参数
        extra_params = params.get("search_params")
        if extra_params:
            if isinstance(extra_params, str):
                extra_params = self._parse_search_params(extra_params)
            if isinstance(extra_params, dict):
                search_params.update(extra_params)
        
        return search_params 