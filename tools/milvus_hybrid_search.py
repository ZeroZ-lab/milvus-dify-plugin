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

            # 基础结构校验：每路检索必须包含 annsField、limit，且提供 data（预计算向量）
            for idx, s in enumerate(searches, start=1):
                if not isinstance(s, dict):
                    raise ValueError(f"search[{idx}] must be an object.")
                if "annsField" not in s or not isinstance(s["annsField"], str) or not s["annsField"].strip():
                    raise ValueError(f"search[{idx}].annsField is required and must be a non-empty string.")
                has_data = "data" in s and isinstance(s["data"], list) and len(s["data"]) > 0
                if not has_data:
                    raise ValueError(f"search[{idx}] must provide 'data' as a non-empty array of vectors.")
                # 将 limit 转为 int
                if "limit" not in s:
                    raise ValueError(f"search[{idx}].limit is required.")
                try:
                    s["limit"] = int(s["limit"])  # type: ignore[index]
                except Exception:
                    raise ValueError(f"search[{idx}].limit must be an integer.")
                if s["limit"] <= 0:
                    raise ValueError(f"search[{idx}].limit must be > 0.")

            # 不再支持在工具内部进行文本嵌入，调用方需提供向量

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
                # defer attaching rerank to payload until payload assembly

            # 顶层参数收集（在构建 payload 之前临时存储）
            limit = tool_parameters.get("limit")
            limit_val: Optional[int] = None
            if limit is not None and str(limit) != "":
                try:
                    limit_val = int(limit)
                except Exception:
                    raise ValueError("'limit' must be an integer.")

            output_fields_str = tool_parameters.get("output_fields")
            top_output_fields: Optional[List[str]] = None
            if output_fields_str:
                top_output_fields = [f.strip() for f in str(output_fields_str).split(',') if f.strip()]

            # 可选：partitionNames, consistencyLevel, offset, group settings
            partition_names_str = tool_parameters.get("partition_names")
            parts: Optional[List[str]] = None
            if partition_names_str:
                parts = [p.strip() for p in str(partition_names_str).split(',') if p.strip()]

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

            # 顶层 limit + offset < 16384 校验（若提供了 limit）
            if isinstance(limit_val, int):
                off = int(offset_val or 0)
                if limit_val + off >= 16384:
                    raise ValueError("The sum of 'limit' and 'offset' must be less than 16384.")

            # 可选：functionScore 透传
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

                # 取集合字段维度信息
                desc = client.describe_collection(collection_name) or {}
                dims_map = self._extract_vector_dims(desc)

                # 维度一致性校验（仅支持直接传入 data 向量）
                for idx, s in enumerate(searches, start=1):
                    anns_field = s.get("annsField")
                    field_dim = dims_map.get(str(anns_field))

                    # 校验 data 维度
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

                # 组装最终 payload
                payload: Dict[str, Any] = {
                    "collectionName": collection_name,
                    "search": searches,
                }

                if rerank_strategy:
                    rerank_obj: Dict[str, Any] = {"strategy": rerank_strategy}
                    if tool_parameters.get("rerank_params"):
                        rerank_obj["params"] = json.loads(tool_parameters["rerank_params"])  # 类型在前面已校验
                    if rerank_obj.get("strategy") == "weighted":
                        params = rerank_obj.get("params", {})
                        weights = params.get("weights", [])
                        if len(weights) != len(searches):
                            raise ValueError("'weights' length must equal number of search routes.")
                    payload["rerank"] = rerank_obj

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
        """从 describe 响应中提取向量字段维度映射 {fieldName: dim}."""
        dims: Dict[str, int] = {}
        fields = (desc or {}).get("fields") or []
        for f in fields:
            try:
                name = f.get("fieldName") or f.get("name")
                dtype = (f.get("dataType") or f.get("type") or "").lower()
                if name and ("vector" in dtype):
                    # 尝试不同位置读取 dim
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

    
