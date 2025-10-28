from typing import Any, List, Union, Optional
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool
from .json_utils import parse_json_relaxed

logger = logging.getLogger(__name__)

class MilvusDeleteTool(MilvusBaseTool, Tool):
    """Milvus data deletion tool"""
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """Execute the delete tool"""
        try:
            collection_name = tool_parameters.get("collection_name")
            ids_param = tool_parameters.get("ids")
            filter_expr = tool_parameters.get("filter")
            partition_name = tool_parameters.get("partition_name", "")

            if not collection_name or not self._validate_collection_name(collection_name):
                raise ValueError("Invalid or missing collection name.")

            logger.debug(f"🔍 [DEBUG] Delete parameters - collection: {collection_name}, ids: {ids_param}, filter: {filter_expr}")
            
            # Parse IDs parameter when provided
            ids = None
            if ids_param:
                ids = self._parse_ids(ids_param)
                logger.debug(f"🔢 [DEBUG] Parsed IDs: {ids}")
                
            # Validate that IDs or filter is present
            if not ids and not filter_expr:
                raise ValueError("Either 'ids' or 'filter' must be provided for the delete operation.")

            with self._get_milvus_client(self.runtime.credentials) as client:
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist.")

                primary_field_param = tool_parameters.get("primary_field")
                resolved_primary = None
                if ids and not filter_expr:
                    resolved_primary = self._resolve_primary_field(client, collection_name, primary_field_param)

                client.delete(
                    collection_name=collection_name,
                    ids=ids,
                    filter=filter_expr,
                    partition_name=partition_name if partition_name else None,
                    primary_field=resolved_primary
                )

                response_data = {
                    "operation": "delete",
                    "collection_name": collection_name,
                    "success": True,
                    "message": f"Delete operation was successful for collection '{collection_name}'."
                }
                yield from self._create_success_message(response_data)
                
        except Exception as e:
            logger.error(f"❌ [DEBUG] Delete operation failed: {str(e)}")
            yield from self._handle_error(e)

    def _parse_ids(self, ids_param: Union[str, List]) -> List:
        """Parse IDs using relaxed JSON parsing."""
        if isinstance(ids_param, list):
            return ids_param

        if isinstance(ids_param, str):
            try:
                parsed = parse_json_relaxed(ids_param, expect_types=(list,))
                return parsed
            except Exception:
                # Fallback: treat the raw string as a single ID
                return [ids_param]

        # Wrap non-string/list truthy values as a single ID
        if ids_param is not None:
            return [ids_param]

        return []

    def _handle_error(self, error: Exception) -> Generator[ToolInvokeMessage, None, None]:
        """Standardized error response"""
        error_msg = str(error)
        yield self.create_json_message({
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        })
    
    def _create_success_message(self, data: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """Standardized success response"""
        response = {
            "success": True,
            **data
        }
        yield self.create_json_message(response) 
