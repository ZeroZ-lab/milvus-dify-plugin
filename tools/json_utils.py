import ast
import json
import json5  # type: ignore
import logging
from typing import Any, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_json_relaxed(
    data: Any,
    expect_types: Optional[Iterable[type]] = None,
    max_nested_decodes: int = 2,
    try_json5: bool = True,
) -> Any:
    """Best-effort JSON parsing helper.

    - Accepts already-parsed Python objects (list/dict) and returns them directly if type matches.
    - Tries standard json.loads first.
    - If the result is still a string (nested-escaped JSON), decodes up to `max_nested_decodes` times.
    - Optionally tries `json5` (if available) to accept single quotes, trailing commas, etc.
    - Falls back to `ast.literal_eval` for Python-literal style inputs.
    - As a final attempt, applies `unicode_escape` decoding then json.loads.

    Args:
        data: Input data (str, bytes, list, dict, ...)
        expect_types: Optional iterable of acceptable Python types (e.g., (list, dict)). If provided, the final parsed
                      result must be an instance of one of these types, otherwise a ValueError is raised.
        max_nested_decodes: Maximum times to decode nested-escaped JSON strings.
        try_json5: Whether to attempt `json5.loads` if available.

    Returns:
        Parsed Python object.

    Raises:
        ValueError: If parsing fails or the result type doesn't match `expect_types` (when provided).
    """

    expect_tuple: Optional[Tuple[type, ...]] = tuple(expect_types) if expect_types else None

    def _ok_type(x: Any) -> bool:
        if expect_tuple is None:
            return True
        return isinstance(x, expect_tuple)

    # 1) Already parsed object
    if _ok_type(data):
        if isinstance(data, dict) and any(not isinstance(k, str) for k in data.keys()):
            raise ValueError("Dictionary keys must be strings for relaxed JSON parsing")
        return data

    # 2) Normalize to string
    txt: Optional[str] = None
    if isinstance(data, bytes):
        try:
            txt = data.decode("utf-8", errors="strict")
        except Exception:
            txt = data.decode("utf-8", errors="ignore")
    elif isinstance(data, str):
        txt = data
    else:
        # Not a string-like; if no expected type, return as-is
        if not expect_types:
            return data
        raise ValueError("Input is neither expected type nor a JSON string")

    txt = txt.strip()

    # 3) Standard json.loads
    try:
        parsed: Any = json.loads(txt)
        # decode nested strings if necessary
        for _ in range(max_nested_decodes):
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    break
            else:
                break
        if _ok_type(parsed):
            return parsed
    except json.JSONDecodeError as e:
        logger.debug(f"parse_json_relaxed: json.loads failed: {e}")

    # 4) Try json5 (required dependency)
    if try_json5:
        try:
            parsed = json5.loads(txt)
            if _ok_type(parsed):
                return parsed
        except Exception as e:
            logger.debug(f"parse_json_relaxed: json5 loads failed: {e}")

    # 5) Try ast.literal_eval for Python-style literals (single quotes, etc.)
    try:
        parsed = ast.literal_eval(txt)
        if _ok_type(parsed):
            if isinstance(parsed, dict) and any(not isinstance(k, str) for k in parsed.keys()):
                raise ValueError("Dictionary keys must be strings")
            return parsed
    except Exception as e:
        logger.debug(f"parse_json_relaxed: ast.literal_eval failed: {e}")

    # 6) unicode_escape then json.loads
    try:
        unescaped = bytes(txt, "utf-8").decode("unicode_escape")
        parsed = json.loads(unescaped)
        if _ok_type(parsed):
            if isinstance(parsed, dict) and any(not isinstance(k, str) for k in parsed.keys()):
                raise ValueError("Dictionary keys must be strings")
            return parsed
    except Exception as e:
        logger.debug(f"parse_json_relaxed: unicode_escape path failed: {e}")

    # Final failure
    exp_types_str = (
        ", expect one of: " + ", ".join(t.__name__ for t in expect_types)
        if expect_types
        else ""
    )
    raise ValueError(f"Failed to parse input as JSON{exp_types_str}")
