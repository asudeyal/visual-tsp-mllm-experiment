"""Deney süreleri, Gemini kullanımı ve hata kayıtları."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any


USAGE_FIELDS = (
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "cached_content_token_count",
    "total_token_count",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_timer() -> float:
    return perf_counter()


def elapsed_seconds(started_at: float) -> float:
    return perf_counter() - started_at


def usage_metadata(response: object) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None)
    return {
        field: (
            int(value)
            if (value := getattr(usage, field, None)) is not None
            else None
        )
        for field in USAGE_FIELDS
    }


def api_call_record(
    *,
    phase: str,
    model: str,
    temperature: float,
    started_at_utc: str,
    wall_seconds: float,
    success: bool,
    input_image_count: int,
    input_image_bytes: int,
    usage: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "model": model,
        "temperature": temperature,
        "success": success,
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now_iso(),
        "api_call_wall_seconds": wall_seconds,
        "input_image_count": input_image_count,
        "input_image_bytes": input_image_bytes,
        "usage": usage or {field: None for field in USAGE_FIELDS},
    }


def error_record(
    exc: Exception,
    *,
    phase: str,
    iteration: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "phase": phase,
    }
    if iteration is not None:
        result["iteration"] = iteration
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        result["status_code"] = status_code
    api_call = getattr(exc, "gemini_call_record", None)
    if isinstance(api_call, dict):
        result["api_call"] = api_call
    return result


def summarize_api_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    known_tokens = [
        int(value)
        for call in calls
        if (value := call.get("usage", {}).get("total_token_count")) is not None
    ]
    return {
        "api_call_count": len(calls),
        "successful_api_call_count": sum(bool(call.get("success")) for call in calls),
        "failed_api_call_count": sum(not bool(call.get("success")) for call in calls),
        "total_api_call_wall_seconds": sum(
            float(call.get("api_call_wall_seconds", 0.0)) for call in calls
        ),
        "total_token_count": sum(known_tokens) if known_tokens else None,
    }
