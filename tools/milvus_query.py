from typing import Any, List, Union
from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool
from .json_utils import parse_json_relaxed


class MilvusQueryTool(MilvusBaseTool, Tool):
    """Milvus data query tool"""
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            collection_name = tool_parameters.get("collection_name")
            
            if not collection_name:
                raise ValueError("Collection name is required")
            
            if not self._validate_collection_name(collection_name):
                raise ValueError("Invalid collection name format")
            
            with self._get_milvus_client(self.runtime.credentials) as client:
                # Ensure the collection exists before executing
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist")
                
                result = self._perform_query(client, collection_name, tool_parameters)
                yield from self._create_success_message(result)
                
        except Exception as e:
            yield from self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> Generator[ToolInvokeMessage]:
        """Standardized error response"""
        error_msg = str(error)
        yield self.create_json_message({
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        })
    
    def _create_success_message(self, data: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Standardized success response"""
        response = {
            "success": True,
            **data
        }
        yield self.create_json_message(response)
    
    def _perform_query(self, client, collection_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute data query based on IDs or filter"""
        # Read query parameters
        ids = params.get("ids")
        filter_expr = params.get("filter")
        
        # Require at least one of ids or filter
        if not ids and not filter_expr:
            raise ValueError("Either 'ids' or 'filter' must be provided")
        
        # Normalize output fields
        output_fields = params.get("output_fields")
        if output_fields and isinstance(output_fields, str):
            output_fields = [field.strip() for field in output_fields.split(",") if field.strip()]
        elif not output_fields:
            output_fields = None
        
        # Normalize partition names (optional)
        partition_names = params.get("partition_names")
        if partition_names and isinstance(partition_names, str):
            partition_names = [name.strip() for name in partition_names.split(",") if name.strip()]
        
        # Normalize optional limit
        limit = params.get("limit")
        if limit:
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                limit = None
        
        # Normalize ids payload when provided
        if ids:
            ids = self._parse_ids(ids)
        
        primary_field_param = params.get("primary_field")

        try:
            if ids:
                resolved_primary = self._resolve_primary_field(client, collection_name, primary_field_param)
                # Run get operation when querying by IDs
                results = client.get(
                    collection_name=collection_name,
                    ids=ids,
                    output_fields=output_fields,
                    partition_names=partition_names,
                    primary_field=resolved_primary
                )
            else:
                # Run query operation when filtering
                results = client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=output_fields,
                    partition_names=partition_names,
                    limit=limit
                )

            return {
                "operation": "query",
                "collection_name": collection_name,
                "query_type": "by_ids" if ids else "by_filter",
                "ids": ids,
                "filter": filter_expr,
                "output_fields": output_fields,
                "partition_names": partition_names,
                "limit": limit,
                "results": results,
                "result_count": len(results) if results else 0
            }
            
        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")
    
    def _parse_ids(self, ids: Union[str, List]) -> List:
        """Parse ID list using relaxed JSON parsing."""
        if isinstance(ids, str):
            try:
                parsed_ids = parse_json_relaxed(ids, expect_types=(list,))
            except Exception:
                # Fall back to comma-separated parsing
                parsed_ids = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]
        else:
            parsed_ids = ids
        
        if not isinstance(parsed_ids, list):
            raise ValueError("IDs must be a list")
        
        if not parsed_ids:
            raise ValueError("IDs list cannot be empty")
        
        # Convert string digits to integers when possible
        converted_ids = []
        for id_val in parsed_ids:
            if isinstance(id_val, str) and id_val.isdigit():
                converted_ids.append(int(id_val))
            else:
                converted_ids.append(id_val)
        
        return converted_ids 
