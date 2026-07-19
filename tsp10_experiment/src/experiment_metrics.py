"""Deneylerde süre, API kullanımı ve hata kayıtları için ortak yardımcılar."""

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
    """Sıralanabilir UTC zaman damgası döndürür."""

    return datetime.now(timezone.utc).isoformat()


def start_timer() -> float:
    """Monoton ve yüksek çözünürlüklü bir süre ölçümü başlatır."""

    return perf_counter()


def elapsed_seconds(started_at: float) -> float:
    """Başlangıçtan beri geçen süreyi saniye olarak döndürür."""

    return perf_counter() - started_at


def extract_usage_metadata(response: object) -> dict[str, int | None]:
    """Gemini yanıtındaki token kullanımını SDK sürümünden bağımsız okur."""

    usage = getattr(response, "usage_metadata", None)
    return {
        field: (
            int(value)
            if (value := getattr(usage, field, None)) is not None
            else None
        )
        for field in USAGE_FIELDS
    }


def build_api_call_record(
    *,
    phase: str,
    model: str,
    temperature: float,
    started_at_utc: str,
    finished_at_utc: str,
    wall_seconds: float,
    success: bool,
    input_image_count: int,
    input_image_bytes: int,
    usage: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """JSON'a doğrudan yazılabilen ortak Gemini çağrı kaydı üretir."""

    return {
        "phase": phase,
        "model": model,
        "temperature": temperature,
        "success": success,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "api_call_wall_seconds": wall_seconds,
        "input_image_count": input_image_count,
        "input_image_bytes": input_image_bytes,
        "usage": usage or {field: None for field in USAGE_FIELDS},
    }


def error_record(
    exc: Exception,
    *,
    iteration: int | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """Hata ve varsa başarısız Gemini çağrısının ölçümünü kaydeder."""

    record: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if iteration is not None:
        record["iteration"] = iteration
    if phase is not None:
        record["phase"] = phase

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        record["status_code"] = status_code
    api_call = getattr(exc, "gemini_call_record", None)
    if isinstance(api_call, dict):
        record["api_call"] = api_call
    return record


def summarize_api_calls(api_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Bir deney içindeki API süre ve token toplamlarını özetler."""

    successful = [call for call in api_calls if call.get("success")]
    failed = [call for call in api_calls if not call.get("success")]
    total_tokens = 0
    token_count_known = False
    for call in api_calls:
        value = call.get("usage", {}).get("total_token_count")
        if value is not None:
            total_tokens += int(value)
            token_count_known = True
    return {
        "api_call_count": len(api_calls),
        "successful_api_call_count": len(successful),
        "failed_api_call_count": len(failed),
        "total_api_call_wall_seconds": sum(
            float(call.get("api_call_wall_seconds", 0.0)) for call in api_calls
        ),
        "total_request_preparation_seconds": sum(
            float(call.get("request_preparation_seconds", 0.0))
            for call in api_calls
        ),
        "total_gemini_request_wall_seconds": sum(
            float(
                call.get(
                    "request_total_wall_seconds",
                    call.get("api_call_wall_seconds", 0.0),
                )
            )
            for call in api_calls
        ),
        "total_token_count": total_tokens if token_count_known else None,
    }
