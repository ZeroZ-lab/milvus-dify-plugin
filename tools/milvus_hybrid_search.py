from typing import Any, Dict, List, Optional
from collections.abc import Generator
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .json_utils import parse_json_relaxed

from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusHybridSearchTool(MilvusBaseTool, Tool):
    """Milvus Hybrid Search (V2) tool"""

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            collection_name = tool_parameters.get("collection_name")
            if not collection_name or not self._validate_collection_name(collection_name):
                raise ValueError("Invalid or missing collection name.")

            # searches_json is required and must follow Zilliz Hybrid Search (V2) schema
            searches_json = tool_parameters.get("searches_json")
            if not searches_json:
                raise ValueError("'searches_json' is required and should be a JSON array of search objects.")

            try:
                searches_any = parse_json_relaxed(searches_json, expect_types=(list,))
            except Exception as e:
                if isinstance(searches_json, str):
                    patched = searches_json.strip()
                    if patched:
                        if not patched.startswith("["):
                            patched = "[" + patched
                        if not patched.endswith("]"):
                            patched = patched + "]"
                        try:
                            searches_any = parse_json_relaxed(patched, expect_types=(list,))
                        except Exception as patch_err:
                            raise ValueError(f"'searches_json' must be a valid JSON array: {patch_err}") from patch_err
                    else:
                        raise ValueError("'searches_json' must be a valid JSON array: empty input") from e
                else:
                    raise ValueError(f"'searches_json' must be a valid JSON array: {e}") from e

            searches: List[Dict[str, Any]] = searches_any  # type: ignore[assignment]

            if not isinstance(searches, list) or len(searches) == 0:
                raise ValueError("'searches_json' must be a non-empty JSON array.")

            # Basic validation: every search route must provide annsField, limit, and precomputed data
            for idx, s in enumerate(searches, start=1):
                if not isinstance(s, dict):
                    raise ValueError(f"search[{idx}] must be an object.")
                if "annsField" not in s or not isinstance(s["annsField"], str) or not s["annsField"].strip():
                    raise ValueError(f"search[{idx}].annsField is required and must be a non-empty string.")
                has_data = "data" in s and isinstance(s["data"], list) and len(s["data"]) > 0
                if not has_data:
                    raise ValueError(f"search[{idx}] must provide 'data' as a non-empty array of vectors.")
                # Ensure limit is converted to int
                if "limit" not in s:
                    raise ValueError(f"search[{idx}].limit is required.")
                try:
                    s["limit"] = int(s["limit"])  # type: ignore[index]
                except Exception:
                    raise ValueError(f"search[{idx}].limit must be an integer.")
                if s["limit"] <= 0:
                    raise ValueError(f"search[{idx}].limit must be > 0.")

            # Text embedding is not performed inside the tool; caller must supply vectors

            # Optional rerank configuration
            rerank_strategy = tool_parameters.get("rerank_strategy")
            rerank_params_raw = tool_parameters.get("rerank_params")
            if rerank_strategy:
                rerank_obj: Dict[str, Any] = {"strategy": rerank_strategy}
                params_obj: Optional[Dict[str, Any]] = None
                if rerank_params_raw:
                    if isinstance(rerank_params_raw, dict):
                        params_obj = rerank_params_raw
                    elif isinstance(rerank_params_raw, str):
                        candidate = rerank_params_raw.strip()
                        if not candidate:
                            params_obj = None
                        else:
                            try:
                                params_obj = json.loads(candidate)
                            except Exception:
                                raise ValueError("'rerank_params' must be a valid JSON object.")
                    else:
                        raise ValueError("'rerank_params' must be a JSON object or stringified JSON.")

                if params_obj is not None:
                    try:
                        json.dumps(params_obj)
                    except TypeError as ser_err:
                        raise ValueError(f"'rerank_params' contains non-serializable values: {ser_err}")

                if params_obj is not None:
                    rerank_obj["params"] = params_obj
                # Additional validation: weighted strategy requires numeric weights matching search count
                if rerank_obj.get("strategy") == "weighted":
                    params = rerank_obj.get("params")
                    if not isinstance(params, dict) or "weights" not in params:
                        raise ValueError("When 'rerank_strategy' is 'weighted', 'rerank_params' must include 'weights'.")
                    weights = params.get("weights")
                    if not isinstance(weights, list):
                        raise ValueError("'rerank_params.weights' must be an array.")
                    if len(weights) != len(searches):
                        raise ValueError("'weights' length must equal number of search routes.")
                    normalized_weights = []
                    for i, w in enumerate(weights, start=1):
                        try:
                            fw = float(w)
                        except Exception:
                            raise ValueError(f"'weights[{i}]' must be numeric.")
                        normalized_weights.append(fw)
                    params["weights"] = normalized_weights
                # defer attaching rerank to payload until payload assembly

            # Collect optional top-level parameters before constructing payload
            limit = tool_parameters.get("limit")
            limit_val: Optional[int] = None
            if limit is not None and str(limit) != "":
                try:
                    limit_val = int(limit)
                except Exception:
                    raise ValueError("'limit' must be an integer.")

            raw_output_fields = tool_parameters.get("output_fields")
            top_output_fields: Optional[List[str]] = None
            if isinstance(raw_output_fields, list):
                candidate = [str(item).strip() for item in raw_output_fields if str(item).strip()]
                if candidate:
                    top_output_fields = candidate
            elif isinstance(raw_output_fields, str):
                candidate = [segment.strip() for segment in raw_output_fields.split(',') if segment.strip()]
                if candidate:
                    top_output_fields = candidate
            elif raw_output_fields is not None:
                raise ValueError("'output_fields' must be a string or an array of field names.")

            # Optional: partitionNames, consistencyLevel, offset, grouping options
            raw_partition_names = tool_parameters.get("partition_names")
            parts: Optional[List[str]] = None
            if isinstance(raw_partition_names, list):
                candidate = [str(item).strip() for item in raw_partition_names if str(item).strip()]
                if candidate:
                    parts = candidate
            elif isinstance(raw_partition_names, str):
                candidate = [segment.strip() for segment in raw_partition_names.split(',') if segment.strip()]
                if candidate:
                    parts = candidate
            elif raw_partition_names is not None:
                raise ValueError("'partition_names' must be a string or an array of partition names.")

            consistency_level = tool_parameters.get("consistency_level")

            offset = tool_parameters.get("offset")
            offset_val: Optional[int] = None
            if offset is not None and str(offset) != "":
                try:
                    offset_val = int(offset)
                except Exception:
                    raise ValueError("'offset' must be an integer.")

            grouping_field = tool_parameters.get("grouping_field")

            group_size = tool_parameters.get("group_size")
            group_size_val: Optional[int] = None
            if group_size is not None and str(group_size) != "":
                try:
                    group_size_val = int(group_size)
                except Exception:
                    raise ValueError("'group_size' must be an integer.")

            strict_group_size = tool_parameters.get("strict_group_size")
            strict_group_flag: Optional[bool] = None
            if isinstance(strict_group_size, bool):
                strict_group_flag = strict_group_size
            elif isinstance(strict_group_size, str) and strict_group_size.lower() in ("true", "false"):
                strict_group_flag = strict_group_size.lower() == "true"

            # Enforce API constraint limit + offset < 16384 when limit is specified
            if isinstance(limit_val, int):
                off = int(offset_val or 0)
                if limit_val + off >= 16384:
                    raise ValueError("The sum of 'limit' and 'offset' must be less than 16384.")

            # Optional: functionScore passthrough
            function_score_json = tool_parameters.get("function_score")
            function_score_obj: Optional[Dict[str, Any]] = None
            if function_score_json:
                try:
                    function_score_obj = json.loads(function_score_json)
                except Exception:
                    raise ValueError("'function_score' must be a valid JSON object.")

            with self._get_milvus_client(self.runtime.credentials) as client:
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist.")

                # Retrieve vector field dimensions for validation
                desc = client.describe_collection(collection_name) or {}
                dims_map = self._extract_vector_dims(desc)

                # Validate each search route against collection schema
                for idx, s in enumerate(searches, start=1):
                    anns_field = s.get("annsField")
                    field_dim = dims_map.get(str(anns_field))

                    # Validate vector dimensionality and numeric content
                    if not isinstance(s.get("data"), list) or len(s["data"]) == 0 or not isinstance(s["data"][0], list):
                        raise ValueError(f"search[{idx}].data must be an array of vectors (e.g., [[...]]).")

                    if field_dim is not None:
                        for vi, v in enumerate(s["data"], start=1):
                            if not all(isinstance(x, (int, float)) for x in v):
                                raise ValueError(f"search[{idx}].data[{vi}] must be a numeric vector.")
                            if len(v) != field_dim:
                                raise ValueError(
                                    f"Dimension mismatch for search[{idx}] on field '{anns_field}': expected {field_dim}, got {len(v)}."
                                )

                # Assemble final payload for hybrid search API
                payload: Dict[str, Any] = {
                    "collectionName": collection_name,
                    "search": searches,
                }

                if rerank_strategy:
                    rerank_obj: Dict[str, Any] = {"strategy": rerank_strategy}
                    if params_obj is not None:
                        rerank_obj["params"] = params_obj
                    if rerank_obj.get("strategy") == "weighted":
                        params = rerank_obj.get("params", {})
                        weights = params.get("weights", [])
                        if len(weights) != len(searches):
                            raise ValueError("'weights' length must equal number of search routes.")
                    elif rerank_obj.get("strategy") not in ("rrf", "weighted"):
                        raise ValueError(f"Unsupported rerank strategy: {rerank_strategy}")
                    payload["rerank"] = rerank_obj
                else:
                    rerank_obj = None

                if isinstance(limit_val, int):
                    payload["limit"] = limit_val
                if top_output_fields:
                    payload["outputFields"] = top_output_fields
                if parts:
                    payload["partitionNames"] = parts
                if consistency_level:
                    payload["consistencyLevel"] = consistency_level
                if isinstance(offset_val, int):
                    payload["offset"] = offset_val
                if grouping_field:
                    payload["groupingField"] = grouping_field
                if isinstance(group_size_val, int):
                    payload["groupSize"] = group_size_val
                if isinstance(strict_group_flag, bool):
                    payload["strictGroupSize"] = strict_group_flag
                if function_score_obj:
                    payload["functionScore"] = function_score_obj

                logger.info(f"🔎 [DEBUG] Hybrid searching in collection '{collection_name}'")
                results = client.hybrid_search(payload)

                response_data = {
                    "operation": "hybrid_search",
                    "collection_name": collection_name,
                    "results": results,
                    "result_count": len(results)
                }
                yield from self._create_success_message(response_data)

        except Exception as e:
            yield from self._handle_error(e)

    def _handle_error(self, error: Exception) -> Generator[ToolInvokeMessage, None, None]:
        error_msg = str(error)
        yield self.create_json_message({
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        })

    def _create_success_message(self, data: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        response = {"success": True, **data}
        yield self.create_json_message(response)

    def _extract_vector_dims(self, desc: Dict[str, Any]) -> Dict[str, int]:
        """Extract a mapping of vector field name to dimension from describe response."""
        dims: Dict[str, int] = {}
        fields = (desc or {}).get("fields") or []
        for f in fields:
            try:
                name = f.get("fieldName") or f.get("name")
                dtype = (f.get("dataType") or f.get("type") or "").lower()
                if name and ("vector" in dtype):
                    # Try different sections for dimension metadata
                    dim = (
                        f.get("elementTypeParams", {}).get("dim")
                        or f.get("typeParams", {}).get("dim")
                        or f.get("params", {}).get("dim")
                    )
                    if isinstance(dim, int):
                        dims[str(name)] = dim
                    elif isinstance(dim, str) and dim.isdigit():
                        dims[str(name)] = int(dim)
            except Exception:
                continue
        return dims

    
