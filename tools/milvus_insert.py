from typing import Any, List, Dict
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool
from .json_utils import parse_json_relaxed

logger = logging.getLogger(__name__)


class MilvusInsertTool(MilvusBaseTool, Tool):
    """Milvus data insertion tool"""
    
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
            
            logger.debug(f"🔍 [DEBUG] Received data type: {type(data)}")
            if isinstance(data, str):
                logger.debug(f"🔍 [DEBUG] Data preview: {data[:100]}...")

            # Parse incoming data payload
            parsed_data = self._parse_insert_data(data)

            with self._get_milvus_client(self.runtime.credentials) as client:
                # Verify target collection exists before insertion
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist")

                result = self._perform_insert(client, collection_name, parsed_data, tool_parameters)
                yield from self._create_success_message(result)

        except Exception as e:
            logger.error(f"❌ [ERROR] Insert operation failed: {str(e)}")
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

    def _parse_insert_data(self, data: str) -> List[Dict[str, Any]]:
        """Parse and normalize insert payload"""
        try:
            logger.debug("🔄 [DEBUG] Begin parsing insert payload")

            if isinstance(data, list):
                logger.debug("✅ [DEBUG] Payload already a list")
                return data

            if isinstance(data, str):
                text = data.strip()
                logger.debug(f"🔍 [DEBUG] Payload preview: {text[:100]}...")

                try:
                    outer = parse_json_relaxed(text, expect_types=(list, dict))
                except Exception as e:
                    logger.debug(f"⚠️ [DEBUG] Outer parsing failed: {e}")
                else:
                    if isinstance(outer, list):
                        logger.debug("✅ [DEBUG] Parsed payload as list")
                        return outer

                    if isinstance(outer, dict):
                        payload = outer.get("data")
                        if payload is None:
                            raise ValueError("Data must contain a 'data' field with JSON array string")

                        try:
                            inner_list = parse_json_relaxed(payload, expect_types=(list,))
                            logger.debug("✅ [DEBUG] Parsed nested payload")
                            return inner_list
                        except Exception as e:
                            logger.debug(f"⚠️ [DEBUG] Nested payload parsing failed: {e}")

            raise ValueError("Failed to parse data payload")

        except Exception as e:
            logger.error(f"❌ [ERROR] Error occurred while parsing data payload: {str(e)}")
            raise ValueError(f"Failed to parse data: {str(e)}")

    def _perform_insert(self, client, collection_name: str, data: List[Dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        """Execute data insertion"""
        # Optional partition name
        partition_name = params.get("partition_name")
        
        # Execute insert call
        try:
            logger.debug(f"🔄 [DEBUG] Inserting into collection={collection_name}, count={len(data)}")
            
            # Validate entity structure
            for i, entity in enumerate(data):
                if not isinstance(entity, dict):
                    raise ValueError(f"Entity at index {i} must be a dictionary")
                if not entity:
                    raise ValueError(f"Entity at index {i} cannot be empty")

            # Enforce schema constraints (types, dimensions, primary key rules)
            self._validate_and_coerce_entities(client, collection_name, data)

            result = client.insert(
                collection_name=collection_name,
                data=data,
                partition_name=partition_name
            )
            
            # HTTP API does not return affected count; default to number of entities
            insert_count = len(data)
            ids = result.get("data", {}).get("insertIds", []) if result else []
            
            logger.debug(f"✅ [DEBUG] Insert succeeded: returned_id_count={len(ids)}")
            
            # Inspect vector field dimensionality for reporting
            vector_info = self._analyze_vector_data(data)
            
            return {
                "operation": "insert",
                "collection_name": collection_name,
                "partition_name": partition_name,
                "insert_count": insert_count,
                "ids": ids,
                "vector_info": vector_info,
                "data_preview": data[:3] if len(data) > 3 else data
            }
            
        except Exception as e:
            logger.error(f"❌ [ERROR] Insert failed: {str(e)}")
            raise ValueError(f"Insert failed: {str(e)}")

    
    def _analyze_vector_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Inspect vector fields to gather metadata for the response"""
        vector_info = {
            "has_vector": False,
            "vector_fields": [],
            "dimensions": {}
        }
        
        if not data:
            return vector_info
        
        # Use the first entity to infer vector field characteristics
        first_entity = data[0]

        for field_name, field_value in first_entity.items():
            if isinstance(field_value, list) and field_value:
                # Confirm the field is a numeric vector
                if all(isinstance(x, (int, float)) for x in field_value):
                    vector_info["has_vector"] = True
                    vector_info["vector_fields"].append(field_name)
                    vector_info["dimensions"][field_name] = len(field_value)
        
        return vector_info 

    
