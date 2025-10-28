from typing import Any
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool

# Configure module-level logger
logger = logging.getLogger(__name__)


class MilvusCollectionTool(MilvusBaseTool, Tool):
    """Milvus collection management tool"""
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        logger.info(f"🚀 [DEBUG] MilvusCollectionTool._invoke() called with params: {tool_parameters}")
        
        try:
            operation = tool_parameters.get("operation")
            collection_name: str | None = tool_parameters.get("collection_name")
            
            logger.debug(f"📋 [DEBUG] Operation: {operation}, Collection: {collection_name}")
            
            if not operation:
                raise ValueError("Operation is required")
            
            # Validate requested operation
            if operation in ["create", "drop"]:
                logger.warning(f"⚠️ [DEBUG] Operation '{operation}' is not implemented")
                raise ValueError(f"Operation '{operation}' is not implemented. Available operations: (list, describe, stats, exists).")
            
            if operation in ["describe", "stats", "exists"] and not collection_name:
                raise ValueError("Collection name is required for this operation")
            
            if collection_name and not self._validate_collection_name(collection_name):
                raise ValueError("Invalid collection name format")
            
            logger.info("🔗 [DEBUG] Attempting to connect to Milvus...")
            
            with self._get_milvus_client(self.runtime.credentials) as client:
                logger.info("✅ [DEBUG] Successfully connected to Milvus")
                
                if operation == "list":
                    logger.debug("📝 [DEBUG] Executing list operation")
                    result = self._list_collections(client)
                # elif operation == "create":
                #     logger.debug("🆕 [DEBUG] Executing create operation")
                #     result = self._create_collection(client, tool_parameters)
                # elif operation == "drop":
                #     logger.debug("🗑️ [DEBUG] Executing drop operation")
                #     if collection_name is None: # Explicit check for linter
                #          raise ValueError("Collection name cannot be None for drop operation.")
                #     result = self._drop_collection(client, collection_name)
                elif operation == "describe":
                    logger.debug("📄 [DEBUG] Executing describe operation")
                    if collection_name is None: # Explicit check for linter
                         raise ValueError("Collection name cannot be None for describe operation.")
                    result = self._describe_collection(client, collection_name)
                elif operation == "stats":
                    logger.debug("📊 [DEBUG] Executing stats operation")
                    if collection_name is None: # Explicit check for linter
                         raise ValueError("Collection name cannot be None for stats operation.")
                    result = self._get_collection_stats(client, collection_name)
                elif operation == "exists":
                    logger.debug("🔍 [DEBUG] Executing exists operation")
                    if collection_name is None: # Explicit check for linter
                         raise ValueError("Collection name cannot be None for exists operation.")
                    result = self._collection_exists(client, collection_name)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                logger.info(f"✅ [DEBUG] Operation completed successfully, result: {result}")
                yield from self._create_success_message(result)
                
        except Exception as e:
            logger.error(f"❌ [DEBUG] Error in _invoke(): {type(e).__name__}: {str(e)}", exc_info=True)
            yield from self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> Generator[ToolInvokeMessage]:
        """Standardized error handling"""
        logger.error(f"🚨 [DEBUG] _handle_error() called with: {type(error).__name__}: {str(error)}")
        error_msg = str(error)
        response = {
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        }
        logger.debug(f"📤 [DEBUG] Sending error response: {response}")
        yield self.create_json_message(response)
    
    def _create_success_message(self, data: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Standardized success response"""
        logger.debug(f"🎉 [DEBUG] _create_success_message() called with data: {data}")
        response = {
            "success": True,
            **data
        }
        logger.debug(f"📤 [DEBUG] Sending success response: {response}")
        yield self.create_json_message(response)
    
    def _list_collections(self, client) -> dict[str, Any]:
        """Return all collections"""
        logger.debug("📋 [DEBUG] _list_collections() called")
        collections = client.list_collections()
        logger.info(f"📋 [DEBUG] Found {len(collections)} collections: {collections}")
        return {
            "operation": "list",
            "collections": collections,
            "count": len(collections)
        }
    
    # TODO: collection creation helper
    # def _create_collection(self, client, params: dict[str, Any]) -> dict[str, Any]:
    #     """Create a collection (not implemented)"""
    #     logger.debug(f"🆕 [DEBUG] _create_collection() called with params: {params}")
    #     collection_name = params.get("collection_name")
    #     dimension = params.get("dimension")
    #     
    #     if not dimension:
    #         raise ValueError("Dimension is required for creating collection")
    #     
    #     try:
    #         dimension = int(dimension)
    #     except (ValueError, TypeError):
    #         raise ValueError("Dimension must be a valid integer")
    #     
    #     if dimension <= 0 or dimension > 32768:
    #         raise ValueError("Dimension must be between 1 and 32768")
    #     
    #     # Read optional parameters
    #     metric_type = params.get("metric_type", "COSINE")
    #     auto_id = params.get("auto_id", True)
    #     description = params.get("description", "")
    #     
    #     logger.info(f"🆕 [DEBUG] Creating collection: {collection_name}, dim: {dimension}, metric: {metric_type}")
    #     
    #     # Perform create operation
    #     client.create_collection(
    #         collection_name=collection_name,
    #         dimension=dimension,
    #         metric_type=metric_type,
    #         auto_id=auto_id,
    #         description=description
    #     )
    #     
    #     logger.info("✅ [DEBUG] Collection created successfully")
    #     
    #     return {
    #         "operation": "create",
    #         "collection_name": collection_name,
    #         "dimension": dimension,
    #         "metric_type": metric_type,
    #         "auto_id": auto_id,
    #         "description": description
    #     }
    
    # TODO: collection drop helper
    # def _drop_collection(self, client, collection_name: str) -> dict[str, Any]:
    #     """Drop a collection (not implemented)"""
    #     logger.debug(f"🗑️ [DEBUG] _drop_collection() called for: {collection_name}")
    #     if not client.has_collection(collection_name):
    #         raise ValueError(f"Collection '{collection_name}' does not exist")
    #     
    #     client.drop_collection(collection_name)
    #     logger.info("✅ [DEBUG] Collection dropped successfully")
    #     
    #     return {
    #         "operation": "drop",
    #         "collection_name": collection_name
    #     }
    
    def _describe_collection(self, client, collection_name: str) -> dict[str, Any]:
        """Describe target collection"""
        logger.debug(f"📄 [DEBUG] _describe_collection() called for: {collection_name}")
        if not client.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        description = client.describe_collection(collection_name)
        logger.debug(f"📄 [DEBUG] Collection description: {description}")
        
        return {
            "operation": "describe",
            "collection_name": collection_name,
            "description": description
        }
    
    def _get_collection_stats(self, client, collection_name: str) -> dict[str, Any]:
        """Fetch collection statistics"""
        logger.debug(f"📊 [DEBUG] _get_collection_stats() called for: {collection_name}")
        if not client.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        stats = client.get_collection_stats(collection_name)
        logger.debug(f"📊 [DEBUG] Collection stats: {stats}")
        
        return {
            "operation": "stats",
            "collection_name": collection_name,
            "stats": stats
        }
    
    def _collection_exists(self, client, collection_name: str) -> dict[str, Any]:
        """Check if collection exists"""
        logger.debug(f"🔍 [DEBUG] _collection_exists() called for: {collection_name}")
        exists = client.has_collection(collection_name)
        logger.info(f"🔍 [DEBUG] Collection exists: {exists}")
        
        return {
            "operation": "exists",
            "collection_name": collection_name,
            "exists": exists
        }


# Emit debug message once module is imported
logger.info("📦 [DEBUG] milvus_collection.py module loaded")
