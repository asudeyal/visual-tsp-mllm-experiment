"""OpenRouter üzerinden görsel TSP rota isteği gönderme araçları."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.metrics import (
    elapsed_seconds,
    start_timer,
    utc_now_iso,
)


OPENROUTER_API_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

OPENROUTER_MODELS: dict[str, str] = {
    "gemma-4-26b-a4b-it": (
        "google/gemma-4-26b-a4b-it:free"
    ),
    "gemma-4-31b-it": "google/gemma-4-31b-it:free",
    "nemotron-3-nano-omni": (
        "nvidia/"
        "nemotron-3-nano-omni-30b-a3b-reasoning:free"
    ),
    "nemotron-nano-12b-v2-vl": (
        "nvidia/nemotron-nano-12b-v2-vl:free"
    ),
}


class OpenRouterAPIError(RuntimeError):
    """OpenRouter HTTP veya yanıt biçimi hatası."""


@dataclass(frozen=True)
class OpenRouterTextResult:
    text: str
    api_call: dict[str, Any]


@dataclass(frozen=True)
class OpenRouterCandidatesResult:
    texts: list[str]
    api_call: dict[str, Any]
    api_calls: list[dict[str, Any]]


def resolve_model_alias(alias: str) -> str:
    """Kısa deney adını sabit OpenRouter model kimliğine çevirir."""

    try:
        return OPENROUTER_MODELS[alias]
    except KeyError as exc:
        expected = ", ".join(OPENROUTER_MODELS)
        raise ValueError(
            f"Bilinmeyen OpenRouter model adı: {alias}. "
            f"Beklenenlerden biri: {expected}"
        ) from exc


def _mime_type(path: Path) -> str:
    value = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower())
    if value is None:
        raise ValueError(
            "OpenRouter için yalnız PNG/JPEG görseller desteklenir."
        )
    return value


def _data_url(path: Path) -> tuple[str, int]:
    image_data = path.read_bytes()
    encoded = base64.b64encode(image_data).decode("ascii")
    return (
        f"data:{_mime_type(path)};base64,{encoded}",
        len(image_data),
    )


def build_route_payload(
    *,
    model: str,
    prompt: str,
    image_data_url: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Modeller arasında ortak tutulan OpenRouter isteğini oluşturur."""

    return build_multimodal_payload(
        model=model,
        prompt=prompt,
        image_data_urls=[image_data_url],
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )


def build_multimodal_payload(
    *,
    model: str,
    prompt: str,
    image_data_urls: list[str],
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    candidate_count: int = 1,
    image_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Tek/çok görselli ve tek/çok adaylı ortak istek gövdesini oluşturur."""

    if candidate_count < 1:
        raise ValueError("candidate_count en az 1 olmalıdır.")
    if image_ids is not None and len(image_ids) != len(
        image_data_urls
    ):
        raise ValueError(
            "Görsel kimlikleri ile görsellerin sayısı eşit olmalıdır."
        )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]
    for index, image_data_url in enumerate(image_data_urls):
        if image_ids is not None:
            content.append(
                {
                    "type": "text",
                    "text": f"Image {image_ids[index]}:",
                }
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url,
                },
            }
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {
            "effort": reasoning_effort,
            "exclude": True,
        },
        "stream": False,
    }
    if candidate_count > 1:
        payload["n"] = candidate_count
    return payload


def normalize_usage(
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """OpenRouter kullanım alanlarını ortak deney şemasına dönüştürür."""

    raw = usage or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = (
        raw.get("completion_tokens_details") or {}
    )
    return {
        "prompt_token_count": raw.get("prompt_tokens"),
        "candidates_token_count": raw.get(
            "completion_tokens"
        ),
        "thoughts_token_count": completion_details.get(
            "reasoning_tokens"
        ),
        "cached_content_token_count": prompt_details.get(
            "cached_tokens"
        ),
        "total_token_count": raw.get("total_tokens"),
        "cost": raw.get("cost"),
        "is_byok": raw.get("is_byok"),
        "raw": raw,
    }


def _response_texts(payload: dict[str, Any]) -> list[str]:
    response_error = payload.get("error")
    if response_error is not None:
        serialized = json.dumps(
            response_error,
            ensure_ascii=False,
        )[:4000]
        raise OpenRouterAPIError(
            "OpenRouter yanıt gövdesi hata içeriyor: "
            f"{serialized}"
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        response_keys = ", ".join(
            sorted(str(key) for key in payload)
        )
        raise OpenRouterAPIError(
            "OpenRouter cevabında choices alanı bulunamadı. "
            f"Yanıt alanları: {response_keys or '(boş yanıt)'}"
        )
    texts: list[str] = []
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
                and item.get("type") in {
                    "text",
                    "output_text",
                }
            ).strip()
        else:
            text = ""
        if text:
            texts.append(text)
    if not texts:
        raise OpenRouterAPIError(
            "OpenRouter boş rota cevabı döndürdü."
        )
    return texts


def _response_text(payload: dict[str, Any]) -> str:
    return _response_texts(payload)[0]


def _safe_error_body(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")[:4000]


def _request_multimodal(
    image_paths: list[Path],
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    phase: str,
    candidate_count: int = 1,
    image_ids: list[int] | None = None,
    timeout_seconds: float = 180.0,
) -> OpenRouterCandidatesResult:
    """Ortak OpenRouter çoklu görsel/çoklu seçim isteğini yürütür."""

    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("En az bir görsel gereklidir.")
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Görsel bulunamadı: {missing[0]}"
        )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY ortam değişkeni tanımlı değil."
        )
    encoded = [_data_url(path) for path in paths]
    image_data_urls = [item[0] for item in encoded]
    image_bytes = sum(item[1] for item in encoded)
    payload = build_multimodal_payload(
        model=model,
        prompt=prompt,
        image_data_urls=image_data_urls,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        candidate_count=candidate_count,
        image_ids=image_ids,
    )
    request = urllib.request.Request(
        OPENROUTER_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": (
                "https://github.com/asudeyal/"
                "visual-tsp-mllm-experiment"
            ),
            "X-OpenRouter-Title": (
                "Dynamic Visual TSP Experiment"
            ),
        },
        method="POST",
    )
    started_at = utc_now_iso()
    timer = start_timer()
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            response_payload = json.loads(
                response.read().decode("utf-8")
            )
    except urllib.error.HTTPError as exc:
        wall_seconds = elapsed_seconds(timer)
        body = _safe_error_body(exc.read())
        error = OpenRouterAPIError(
            f"OpenRouter HTTP {exc.code}: {body}"
        )
        error.status_code = exc.code
        error.openrouter_call_record = {
            "phase": phase,
            "provider": "openrouter",
            "model": model,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "success": False,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "api_call_wall_seconds": wall_seconds,
            "input_image_count": len(paths),
            "input_image_bytes": image_bytes,
            "requested_candidate_count": candidate_count,
            "returned_candidate_count": 0,
            "usage": normalize_usage(None),
        }
        error.gemini_call_record = error.openrouter_call_record
        raise error from exc
    except urllib.error.URLError as exc:
        wall_seconds = elapsed_seconds(timer)
        error = OpenRouterAPIError(
            f"OpenRouter bağlantı hatası: {exc.reason}"
        )
        error.openrouter_call_record = {
            "phase": phase,
            "provider": "openrouter",
            "model": model,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "success": False,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "api_call_wall_seconds": wall_seconds,
            "input_image_count": len(paths),
            "input_image_bytes": image_bytes,
            "requested_candidate_count": candidate_count,
            "returned_candidate_count": 0,
            "usage": normalize_usage(None),
        }
        error.gemini_call_record = error.openrouter_call_record
        raise error from exc

    wall_seconds = elapsed_seconds(timer)
    choices = response_payload.get("choices") or []
    call = {
        "phase": phase,
        "provider": "openrouter",
        "model": model,
        "response_model": response_payload.get("model"),
        "routed_provider": response_payload.get("provider"),
        "generation_id": response_payload.get("id"),
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "success": True,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "api_call_wall_seconds": wall_seconds,
        "input_image_count": len(paths),
        "input_image_bytes": image_bytes,
        "requested_candidate_count": candidate_count,
        "returned_candidate_count": len(choices),
        "finish_reason": (
            (choices or [{}])[0].get("finish_reason")
        ),
        "finish_reasons": [
            choice.get("finish_reason")
            for choice in choices
        ],
        "usage": normalize_usage(
            response_payload.get("usage")
        ),
    }
    try:
        texts = _response_texts(response_payload)
    except Exception as exc:
        call["success"] = False
        if response_payload.get("error") is not None:
            call["response_error"] = response_payload.get(
                "error"
            )
        exc.openrouter_call_record = call
        exc.gemini_call_record = call
        raise
    return OpenRouterCandidatesResult(
        texts=texts,
        api_call=call,
        api_calls=[call],
    )


def request_route(
    image_path: Path,
    *,
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    reasoning_effort: str = "none",
    phase: str = "route_generation",
    timeout_seconds: float = 180.0,
) -> OpenRouterTextResult:
    """Bir yerel görselden tek rota ister."""

    response = _request_multimodal(
        [image_path],
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        phase=phase,
        timeout_seconds=timeout_seconds,
    )
    return OpenRouterTextResult(
        text=response.texts[0],
        api_call=response.api_call,
    )


def request_candidates(
    image_path: Path,
    *,
    candidate_count: int,
    model: str,
    prompt: str | None = None,
    problem: Any = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    reasoning_effort: str = "none",
    timeout_seconds: float = 180.0,
    strategy: str = "independent_calls",
) -> OpenRouterCandidatesResult:
    """Gemini candidate_count davranışına karşılık gelen adayları üretir."""

    if not 1 <= candidate_count <= 7:
        raise ValueError(
            "candidate-count 1 ile 7 arasında olmalıdır."
        )
    if prompt is None:
        if problem is None:
            raise ValueError(
                "Critic isteği için prompt veya problem gereklidir."
            )
        from src.gemini import critic_prompt

        prompt = critic_prompt(problem)
    if strategy == "native_multiple_choices":
        return _request_multimodal(
            [image_path],
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            phase="critic_candidate_generation",
            candidate_count=candidate_count,
            timeout_seconds=timeout_seconds,
        )
    if strategy != "independent_calls":
        raise ValueError(
            "Bilinmeyen critic aday stratejisi: "
            f"{strategy}"
        )

    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for candidate_id in range(1, candidate_count + 1):
        try:
            response = _request_multimodal(
                [image_path],
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                phase=(
                    "critic_candidate_generation_"
                    f"{candidate_id:02d}"
                ),
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            failed_call = getattr(
                exc,
                "openrouter_call_record",
                None,
            )
            recorded = list(calls)
            if isinstance(failed_call, dict):
                recorded.append(failed_call)
            try:
                setattr(
                    exc,
                    "openrouter_call_records",
                    recorded,
                )
            except Exception:
                pass
            raise
        texts.append(response.texts[0])
        calls.extend(response.api_calls)

    usage_fields = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "total_token_count",
        "cost",
    )
    usage: dict[str, Any] = {}
    for field in usage_fields:
        values = [
            (call.get("usage") or {}).get(field)
            for call in calls
        ]
        known = [
            float(value)
            for value in values
            if value is not None
        ]
        usage[field] = sum(known) if known else None
    aggregate = {
        "phase": "critic_candidate_generation",
        "provider": "openrouter",
        "model": model,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "success": True,
        "strategy": strategy,
        "http_request_count": len(calls),
        "started_at_utc": calls[0]["started_at_utc"],
        "finished_at_utc": calls[-1]["finished_at_utc"],
        "api_call_wall_seconds": sum(
            float(call["api_call_wall_seconds"])
            for call in calls
        ),
        "input_image_count": sum(
            int(call.get("input_image_count") or 0)
            for call in calls
        ),
        "input_image_bytes": sum(
            int(call.get("input_image_bytes") or 0)
            for call in calls
        ),
        "requested_candidate_count": candidate_count,
        "returned_candidate_count": len(texts),
        "response_models": sorted(
            {
                str(call["response_model"])
                for call in calls
                if call.get("response_model")
            }
        ),
        "routed_providers": sorted(
            {
                str(call["routed_provider"])
                for call in calls
                if call.get("routed_provider")
            }
        ),
        "finish_reasons": [
            call.get("finish_reason")
            for call in calls
        ],
        "usage": usage,
    }
    return OpenRouterCandidatesResult(
        texts=texts,
        api_call=aggregate,
        api_calls=calls,
    )


def request_scorer(
    image_paths: list[Path],
    *,
    image_ids: list[int],
    model: str,
    prompt: str | None = None,
    problem: Any = None,
    max_tokens: int = 4096,
    reasoning_effort: str = "none",
    timeout_seconds: float = 180.0,
) -> OpenRouterTextResult:
    """Geçerli aday görsellerini tek bir görsel scorer çağrısına gönderir."""

    if prompt is None:
        if problem is None:
            raise ValueError(
                "Scorer isteği için prompt veya problem gereklidir."
            )
        from src.gemini import scorer_prompt

        prompt = scorer_prompt(problem, image_ids)
    response = _request_multimodal(
        image_paths,
        prompt=prompt,
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        phase="visual_scorer",
        image_ids=image_ids,
        timeout_seconds=timeout_seconds,
    )
    return OpenRouterTextResult(
        text=response.texts[0],
        api_call=response.api_call,
    )
