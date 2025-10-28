from typing import Any, Dict, List, Optional
from collections.abc import Generator
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusHybridSearchTool(MilvusBaseTool, Tool):
    """Milvus 混合搜索 (Hybrid Search V2) 工具"""

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            collection_name = tool_parameters.get("collection_name")
            if not collection_name or not self._validate_collection_name(collection_name):
                raise ValueError("Invalid or missing collection name.")

            # searches_json 必填，遵循 Zilliz Hybrid Search (V2) 的每项 search 结构
            searches_json = tool_parameters.get("searches_json")
            if not searches_json:
                raise ValueError("'searches_json' is required and should be a JSON array of search objects.")

            try:
                searches: List[Dict[str, Any]] = json.loads(searches_json)
            except Exception:
                raise ValueError("'searches_json' must be a valid JSON array.")

            if not isinstance(searches, list) or len(searches) == 0:
                raise ValueError("'searches_json' must be a non-empty JSON array.")

            # 基础结构校验：每路检索必须包含 data、annsField、limit
            for idx, s in enumerate(searches, start=1):
                if not isinstance(s, dict):
                    raise ValueError(f"search[{idx}] must be an object.")
                if "annsField" not in s or not isinstance(s["annsField"], str) or not s["annsField"].strip():
                    raise ValueError(f"search[{idx}].annsField is required and must be a non-empty string.")
                if "data" not in s or not isinstance(s["data"], list) or len(s["data"]) == 0:
                    raise ValueError(f"search[{idx}].data is required and must be a non-empty array of vectors.")
                # 将 limit 转为 int
                if "limit" not in s:
                    raise ValueError(f"search[{idx}].limit is required.")
                try:
                    s["limit"] = int(s["limit"])  # type: ignore[index]
                except Exception:
                    raise ValueError(f"search[{idx}].limit must be an integer.")
                if s["limit"] <= 0:
                    raise ValueError(f"search[{idx}].limit must be > 0.")

            payload: Dict[str, Any] = {
                "collectionName": collection_name,
                "search": searches,
            }

            # 可选：rerank
            rerank_strategy = tool_parameters.get("rerank_strategy")
            rerank_params_json = tool_parameters.get("rerank_params")
            if rerank_strategy:
                rerank_obj: Dict[str, Any] = {"strategy": rerank_strategy}
                if rerank_params_json:
                    try:
                        rerank_obj["params"] = json.loads(rerank_params_json)
                    except Exception:
                        raise ValueError("'rerank_params' must be a valid JSON object.")
                # 加强校验：weighted 权重长度与路数一致，且为数值
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
                payload["rerank"] = rerank_obj

            # 顶层 limit / outputFields
            limit = tool_parameters.get("limit")
            if limit is not None and str(limit) != "":
                try:
                    payload["limit"] = int(limit)
                except Exception:
                    raise ValueError("'limit' must be an integer.")

            output_fields_str = tool_parameters.get("output_fields")
            if output_fields_str:
                payload["outputFields"] = [f.strip() for f in str(output_fields_str).split(',') if f.strip()]

            # 可选：partitionNames, consistencyLevel, offset, group settings
            partition_names_str = tool_parameters.get("partition_names")
            if partition_names_str:
                parts = [p.strip() for p in str(partition_names_str).split(',') if p.strip()]
                if parts:
                    payload["partitionNames"] = parts

            consistency_level = tool_parameters.get("consistency_level")
            if consistency_level:
                payload["consistencyLevel"] = consistency_level

            offset = tool_parameters.get("offset")
            if offset is not None and str(offset) != "":
                try:
                    payload["offset"] = int(offset)
                except Exception:
                    raise ValueError("'offset' must be an integer.")

            grouping_field = tool_parameters.get("grouping_field")
            if grouping_field:
                payload["groupingField"] = grouping_field

            group_size = tool_parameters.get("group_size")
            if group_size is not None and str(group_size) != "":
                try:
                    payload["groupSize"] = int(group_size)
                except Exception:
                    raise ValueError("'group_size' must be an integer.")

            strict_group_size = tool_parameters.get("strict_group_size")
            if isinstance(strict_group_size, bool):
                payload["strictGroupSize"] = strict_group_size
            elif isinstance(strict_group_size, str) and strict_group_size.lower() in ("true", "false"):
                payload["strictGroupSize"] = strict_group_size.lower() == "true"

            # 顶层 limit + offset < 16384 校验（若提供了 limit）
            top_limit = payload.get("limit")
            top_offset = payload.get("offset", 0)
            if isinstance(top_limit, int):
                try:
                    off = int(top_offset)
                except Exception:
                    raise ValueError("'offset' must be an integer.")
                if top_limit + off >= 16384:
                    raise ValueError("The sum of 'limit' and 'offset' must be less than 16384.")

            # 可选：functionScore 透传
            function_score_json = tool_parameters.get("function_score")
            if function_score_json:
                try:
                    payload["functionScore"] = json.loads(function_score_json)
                except Exception:
                    raise ValueError("'function_score' must be a valid JSON object.")

            with self._get_milvus_client(self.runtime.credentials) as client:
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist.")

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
