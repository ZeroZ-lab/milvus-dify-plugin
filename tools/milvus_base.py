from typing import Any, Optional, Dict, List
from collections.abc import Generator
from contextlib import contextmanager
import requests
import json
import time
import logging
import re
from .json_utils import parse_json_relaxed

logger = logging.getLogger(__name__)

class MilvusBaseTool:
    """Base helpers shared by Milvus tools (connection, schema utilities, validation)."""
    
    @contextmanager
    def _get_milvus_client(self, credentials: dict[str, Any]):
        """Context manager that yields a Milvus HTTP client.

        Only wraps connectivity tests so downstream errors propagate without being
        mislabeled as connection failures.
        """
        uri = credentials.get("uri")
        token = credentials.get("token")
        database = credentials.get("database", "default")

        if not uri:
            raise ValueError("URI is required")

        if not uri.startswith(("http://", "https://")):
            uri = f"http://{uri}"
        uri = uri.rstrip('/')

        client = MilvusHttpClient(
            uri=uri,
            token=token if token else "",
            database=database,
            timeout=30.0
        )

        # Wrap connectivity test only; downstream errors should propagate as-is
        try:
            client.test_connection()
            logger.info(f"✅ [DEBUG] Successfully connected to Milvus HTTP API at {uri}")
        except Exception as e:
            logger.error(f"❌ [DEBUG] Failed to connect to Milvus: {str(e)}")
            client.close()
            raise ValueError(f"Failed to connect to Milvus: {str(e)}")

        try:
            yield client
        finally:
            client.close()
    
    def _validate_collection_name(self, collection_name: str) -> bool:
        """Return True if the collection name satisfies Milvus naming rules."""
        if not collection_name or not isinstance(collection_name, str):
            return False
        if len(collection_name) > 255:
            return False
        # Collection names may contain letters, digits, underscores and must not start with a digit
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', collection_name))

    def _parse_vector_data(self, data: str) -> list:
        """Parse vector JSON into a Python list."""
        try:
            if isinstance(data, str):
                parsed = parse_json_relaxed(data, expect_types=(list,))
                return parsed
            return data
        except (json.JSONDecodeError, TypeError):
            raise ValueError("Invalid vector data format. Expected JSON array.")
    
    def _parse_search_params(self, params_str: Optional[str]) -> dict:
        """Parse search-parameter JSON into a dict."""
        if not params_str:
            return {}
        
        try:
            return json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _sanitize_field_name(self, name: Optional[str]) -> Optional[str]:
        if not isinstance(name, str):
            return None
        candidate = name.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
            return candidate
        return None

    def _resolve_primary_field(self, client, collection_name: str, override: Optional[str]) -> Optional[str]:
        sanitized = self._sanitize_field_name(override)
        if sanitized:
            return sanitized

        try:
            desc = client.describe_collection(collection_name)
        except Exception as e:
            logger.debug(f"⚠️ [DEBUG] describe_collection failed while resolving primary field: {e}")
            return None

        schema_info = self._extract_schema_info(desc)
        primary = schema_info.get("primary_field")
        return self._sanitize_field_name(primary)

    def _extract_schema_info(self, desc: Any) -> dict[str, Any]:
        """Extract fields, primary key and auto-id flag from describe_collection output."""
        info: dict[str, Any] = {
            "fields": [],
            "primary_field": None,
            "auto_id": False,
        }

        if desc is None:
            return info

        auto_id_value: Optional[bool] = None
        fields: list[dict[str, Any]] = []

        def _maybe_set_auto(container: Any):
            nonlocal auto_id_value
            if auto_id_value is not None:
                return
            if isinstance(container, dict):
                for key in ("autoId", "autoID"):
                    if key in container:
                        val = container[key]
                        if isinstance(val, bool):
                            auto_id_value = val
                        elif isinstance(val, (int, str)):
                            auto_id_value = bool(val)
                        break

        def _collect_fields(container: Any) -> Optional[list[dict[str, Any]]]:
            if isinstance(container, dict):
                candidates = container.get("fields")
                if isinstance(candidates, list):
                    return [f for f in candidates if isinstance(f, dict)]
            return None

        if isinstance(desc, dict):
            _maybe_set_auto(desc)
            fields = _collect_fields(desc) or []
            if not fields:
                schema = desc.get("schema")
                if isinstance(schema, dict):
                    _maybe_set_auto(schema)
                    fields = _collect_fields(schema) or []
        elif isinstance(desc, list):
            for item in desc:
                if isinstance(item, dict):
                    _maybe_set_auto(item)
                    candidate = _collect_fields(item)
                    if candidate:
                        fields = candidate
                        break
            if not fields:
                fields = [item for item in desc if isinstance(item, dict)]
        else:
            return info

        info["fields"] = fields
        info["auto_id"] = bool(auto_id_value)

        for field in fields:
            if isinstance(field, dict) and field.get("isPrimary") is True:
                info["primary_field"] = field.get("fieldName") or field.get("name")
                break

        # Some responses omit isPrimary; fall back to schema-level metadata
        if info["primary_field"] is None:
            if isinstance(desc, dict):
                schema = desc.get("schema")
                if isinstance(schema, dict):
                    primary = schema.get("primaryField") or schema.get("primary_field")
                    if isinstance(primary, str) and primary:
                        info["primary_field"] = primary

        return info

    def _validate_and_coerce_entities(
        self,
        client,
        collection_name: str,
        entities: List[Dict[str, Any]],
    ) -> None:
        """Validate and coerce entities based on collection schema."""
        if not entities:
            return

        try:
            desc = client.describe_collection(collection_name) or {}
        except Exception as e:
            logger.debug(f"⚠️ [DEBUG] describe_collection failed, skip strict validation: {e}")
            return

        schema_info = self._extract_schema_info(desc)
        fields: List[Dict[str, Any]] = schema_info.get("fields", [])

        if not fields:
            logger.debug("⚠️ [DEBUG] describe_collection returned no field metadata; skip validation")
            return

        auto_id_enabled = bool(schema_info.get("auto_id"))
        primary_field = self._sanitize_field_name(schema_info.get("primary_field"))

        int_fields: set[str] = set()
        vector_dims: Dict[str, int] = {}

        for field in fields:
            name = self._sanitize_field_name(field.get("fieldName") or field.get("name"))
            if not name:
                continue
            dtype_val = field.get("dataType") or field.get("type")
            dtype = str(dtype_val).lower() if dtype_val else ""
            if dtype == "int64":
                int_fields.add(name)
            if "vector" in dtype:
                dim = None
                for key in ("elementTypeParams", "typeParams", "params"):
                    params = field.get(key)
                    if isinstance(params, dict) and params.get("dim") is not None:
                        dim = params.get("dim")
                        break
                if isinstance(dim, str) and dim.isdigit():
                    dim = int(dim)
                if isinstance(dim, int):
                    vector_dims[name] = dim

        for idx, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(f"Entity at index {idx} must be a dictionary")

            if auto_id_enabled and primary_field and primary_field in entity:
                raise ValueError(
                    f"Collection primary key '{primary_field}' is AutoID=true; do not provide it in entities (at index {idx})."
                )

            for field_name in int_fields:
                if field_name not in entity:
                    continue
                value = entity[field_name]
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    raise ValueError(f"Entity[{idx}].{field_name} expected Int64 but got empty/None.")
                if isinstance(value, str):
                    try:
                        entity[field_name] = int(value.strip())
                    except Exception:
                        raise ValueError(
                            f"Entity[{idx}].{field_name} expected Int64 but got string '{value}'."
                        )
                elif not isinstance(value, int):
                    raise ValueError(
                        f"Entity[{idx}].{field_name} expected Int64 but got {type(value).__name__}."
                    )

            for field_name, dim in vector_dims.items():
                if field_name not in entity:
                    continue
                vector = entity[field_name]
                if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
                    raise ValueError(f"Entity[{idx}].{field_name} must be a numeric array.")
                if dim and len(vector) != dim:
                    raise ValueError(
                        f"Entity[{idx}].{field_name} dimension mismatch: expected {dim}, got {len(vector)}."
                    )


class MilvusHttpClient:
    """Lightweight REST client for Milvus HTTP API."""
    
    def __init__(self, uri: str, token: str = "", database: str = "default", timeout: float = 30.0):
        self.uri = uri
        self.token = token
        self.database = database
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Apply authentication header when token is provided
        if token:
            self.session.headers['Authorization'] = f'Bearer {token}'
    
    def test_connection(self):
        """Verify connectivity by listing collections."""
        try:
            response = self._make_request('POST', '/v2/vectordb/collections/list', {})
            return response.get('code') == 0
        except Exception as e:
            raise ValueError(f"Connection test failed: {str(e)}")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[dict] = None, max_retries: int = 3) -> dict:
        """Send an HTTP request with retry and response validation."""
        url = f"{self.uri}{endpoint}"
        
        # Ensure dbName is present on every request payload
        if data is None:
            data = {}
        
        # Reuse caller-provided dictionary
        data['dbName'] = self.database
        
        logger.debug(f"🌐 [DEBUG] Making {method} request to: {url}")
        logger.debug(f"📦 [DEBUG] Request data: {data}")
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"🔄 [DEBUG] Attempt {attempt + 1}/{max_retries}")
                
                if method == 'GET':
                    response = self.session.get(url, headers=headers, timeout=self.timeout)
                elif method == 'POST':
                    response = self.session.post(url, json=data, headers=headers, timeout=self.timeout)
                elif method == 'PUT':
                    response = self.session.put(url, json=data, headers=headers, timeout=self.timeout)
                elif method == 'DELETE':
                    response = self.session.delete(url, headers=headers, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                logger.debug(f"📡 [DEBUG] Response status: {response.status_code}")
                
                # Abort on non-200 HTTP status codes
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"❌ [DEBUG] Request failed: {error_msg}")
                    raise ValueError(error_msg)
                
                # Parse JSON response body
                result = response.json()
                logger.debug(f"✅ [DEBUG] Response: {result}")
                
                # Validate Milvus-specific response code
                if result.get('code') != 0:
                    error_msg = result.get('message', 'Unknown error')
                    logger.error(f"❌ [DEBUG] Milvus API error: {error_msg}")
                    raise ValueError(f"Milvus API error: {error_msg}")
                
                return result
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ [DEBUG] Request attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Request failed after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff between retries
                time.sleep(2 ** attempt)
        
        # Should be unreachable but keeps type-checkers satisfied
        raise ValueError("Request failed after all retries")
    
    # Collection operations
    def list_collections(self) -> list:
        """Return list of collections."""
        response = self._make_request('POST', '/v2/vectordb/collections/list', {})
        return response.get('data', [])
    
    def has_collection(self, collection_name: str) -> bool:
        """Return True if the collection exists."""
        try:
            response = self._make_request('POST', '/v2/vectordb/collections/describe', {
                'collectionName': collection_name
            })
            return response.get('code') == 0
        except:
            return False
    
    def create_collection(self, collection_name: str, dimension: int, metric_type: str = "COSINE", 
                         auto_id: bool = True, description: str = ""):
        """Create a collection with default schema."""
        schema = {
            "autoID": auto_id,
            "fields": [
                {
                    "fieldName": "id",
                    "dataType": "Int64",
                    "isPrimary": True
                },
                {
                    "fieldName": "vector",
                    "dataType": "FloatVector",
                    "elementTypeParams": {
                        "dim": dimension
                    }
                }
            ]
        }
        
        index_params = [
            {
                "fieldName": "vector",
                "metricType": metric_type,
                "indexType": "AUTOINDEX"
            }
        ]
        
        data = {
            "collectionName": collection_name,
            "schema": schema,
            "indexParams": index_params
        }
        
        if description:
            data["description"] = description
        
        return self._make_request('POST', '/v2/vectordb/collections/create', data)
    
    def drop_collection(self, collection_name: str):
        """Drop a collection."""
        return self._make_request('POST', '/v2/vectordb/collections/drop', {
            'collectionName': collection_name
        })
    
    def describe_collection(self, collection_name: str):
        """Describe a collection."""
        response = self._make_request('POST', '/v2/vectordb/collections/describe', {
            'collectionName': collection_name
        })
        return response.get('data', {})
    
    def get_collection_stats(self, collection_name: str, timeout: Optional[float] = None) -> dict:
        """Fetch collection statistics."""
        logger.debug(f"📊 [DEBUG] get_collection_stats() called for: {collection_name}")
        
        # Use the describe endpoint because stats endpoint is not available
        response = self._make_request('POST', '/v2/vectordb/collections/describe', {
            'collectionName': collection_name
        })
        
        collection_info = response.get('data', {})
        
        # Extract statistics from describe payload
        stats = {
            'collection_name': collection_info.get('collectionName', collection_name),
            'description': collection_info.get('description', ''),
            'num_shards': collection_info.get('shardsNum', 0),
            'num_partitions': collection_info.get('partitionsNum', 0),
            'num_fields': len(collection_info.get('fields', [])),
            'num_indexes': len(collection_info.get('indexes', [])),
            'consistency_level': collection_info.get('consistencyLevel', 'Unknown'),
            'load_state': collection_info.get('load', 'Unknown'),
            'auto_id': collection_info.get('autoId', False),
            'enable_dynamic_field': collection_info.get('enableDynamicField', False),
            'collection_id': collection_info.get('collectionID', 0),
            'fields': collection_info.get('fields', []),
            'indexes': collection_info.get('indexes', [])
        }
        
        logger.debug(f"📊 [DEBUG] Collection stats extracted from describe: {stats}")
        
        return stats
    
    def load_collection(self, collection_name: str):
        """Load a collection into memory."""
        return self._make_request('POST', '/v2/vectordb/collections/load', {
            'collectionName': collection_name
        })
    
    def release_collection(self, collection_name: str):
        """Release a collection from memory."""
        return self._make_request('POST', '/v2/vectordb/collections/release', {
            'collectionName': collection_name
        })
    
    # Data operations
    def insert(self, collection_name: str, data: List[Dict[str, Any]], partition_name: Optional[str] = None):
        """Insert entities into Milvus."""
        request_data = {
            'collectionName': collection_name,
            'data': data
        }
        
        if partition_name:
            request_data['partitionName'] = partition_name
        
        return self._make_request('POST', '/v2/vectordb/entities/insert', request_data)
    
    def upsert(self, collection_name: str, data: list[dict[str, Any]], partition_name: Optional[str] = None):
        """Insert or update entities."""
        request_data: dict[str, Any] = {
            'collectionName': collection_name,
            'data': data
        }
        if partition_name:
            request_data['partitionName'] = partition_name
        return self._make_request('POST', '/v2/vectordb/entities/upsert', request_data)
    
    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        anns_field: str = "vector",
        limit: int = 10,
        output_fields: Optional[list[str]] = None,
        filter: Optional[str] = None,
        search_params: Optional[dict] = None,
        partition_names: Optional[list[str]] = None,
        **kwargs
    ) -> list:
        """Perform vector similarity search."""
        payload: dict[str, Any] = {
            "collectionName": collection_name,
            "data": data,
            "annsField": anns_field,
            "limit": limit
        }

        # Attach optional parameters
        if output_fields:
            payload["outputFields"] = output_fields
        if filter:
            payload["filter"] = filter
        if search_params:
            payload["searchParams"] = search_params
        if partition_names:
            payload["partitionNames"] = partition_names

        response = self._make_request(
            "POST", 
            "/v2/vectordb/entities/search", 
            payload
        )
        return response.get('data', [])
    
    def query(self, collection_name: str, filter: Optional[str] = None, output_fields: Optional[list[str]] = None,
              limit: Optional[int] = None, partition_names: Optional[list[str]] = None):
        """Query entities via filter or IDs."""
        request_data: Dict[str, Any] = {
            'collectionName': collection_name
        }
        
        if filter:
            request_data['filter'] = filter
        
        if output_fields:
            request_data['outputFields'] = output_fields
        
        if limit is not None:
            request_data['limit'] = limit
        
        if partition_names:
            request_data['partitionNames'] = partition_names
        
        response = self._make_request('POST', '/v2/vectordb/entities/query', request_data)
        return response.get('data', [])

    def hybrid_search(self, payload: Dict[str, Any]) -> list:
        """Perform Hybrid Search (V2) with a fully-specified payload."""
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict for hybrid_search")

        if not payload.get("collectionName"):
            raise ValueError("'collectionName' is required in payload for hybrid_search")

        if not payload.get("search") or not isinstance(payload.get("search"), list):
            raise ValueError("'search' array is required in payload for hybrid_search")

        response = self._make_request(
            'POST',
            '/v2/vectordb/entities/hybrid_search',
            payload
        )
        return response.get('data', [])
    
    def get(self, collection_name: str, ids: List[Any], output_fields: Optional[List[str]] = None,
            partition_names: Optional[List[str]] = None, primary_field: Optional[str] = None):
        """Fetch entities by IDs."""
        request_data: Dict[str, Any] = {
            'collectionName': collection_name,
            'id': ids
        }

        sanitized_field = None
        if isinstance(primary_field, str):
            candidate = primary_field.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
                sanitized_field = candidate
        if sanitized_field:
            request_data['primaryField'] = sanitized_field
        
        if output_fields:
            request_data['outputFields'] = output_fields
        
        if partition_names:
            request_data['partitionNames'] = partition_names
        
        response = self._make_request('POST', '/v2/vectordb/entities/get', request_data)
        return response.get('data', [])
    
    def delete(self, collection_name: str, ids: Optional[list[Any]] = None, filter: Optional[str] = None,
               partition_name: Optional[str] = None, primary_field: Optional[str] = None):
        """Delete entities by IDs or filter."""
        # Must provide either a non-empty list of IDs or a non-empty filter expression
        has_valid_ids = isinstance(ids, list) and len(ids) > 0
        has_valid_filter = isinstance(filter, str) and len(filter.strip()) > 0

        if not has_valid_ids and not has_valid_filter:
            raise ValueError("Either a non-empty list of 'ids' or a non-empty 'filter' string must be provided for the delete operation.")

        request_data: dict[str, Any] = {
            'collectionName': collection_name
        }

        # Build filter expression when only IDs are provided
        if has_valid_ids and not has_valid_filter:
            field_name = "id"
            if isinstance(primary_field, str):
                candidate = primary_field.strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
                    field_name = candidate
            if ids and all(isinstance(i, str) for i in ids):
                id_list_str = ", ".join(f'"{i}"' for i in ids)
            elif ids:
                id_list_str = ", ".join(str(i) for i in ids)
            else:
                id_list_str = ""
            request_data['filter'] = f"{field_name} in [{id_list_str}]"
        elif has_valid_filter:
            request_data['filter'] = filter
        
        # If both IDs and filter are supplied, prefer caller filter
        logger.debug(f"🗑️ [DEBUG] Delete operation with filter: {request_data.get('filter')}")

        if partition_name:
            request_data['partitionName'] = partition_name

        return self._make_request('POST', '/v2/vectordb/entities/delete', request_data)
    
    def close(self):
        """Close the underlying requests session."""
        self.session.close() 
