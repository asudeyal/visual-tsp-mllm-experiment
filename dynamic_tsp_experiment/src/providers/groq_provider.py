"""Groq OpenAI-uyumlu vision API adaptörü."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from src.gemini import critic_prompt, scorer_prompt
from src.metrics import (
    elapsed_seconds,
    start_timer,
    summarize_api_calls,
    utc_now_iso,
)
from src.problem_instance import ProblemInstance
from src.providers.base import (
    ProviderAdapter,
    ProviderCandidatesResult,
    ProviderCapabilities,
    ProviderTextResult,
)


GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqAPIError(RuntimeError):
    """Groq HTTP veya cevap biçimi hatası."""


def _mime_type(path: Path) -> str:
    value = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower())
    if value is None:
        raise ValueError("Groq için yalnız PNG/JPEG desteklenir.")
    return value


def _data_url(path: Path) -> tuple[str, int]:
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    return (
        f"data:{_mime_type(path)};base64,{encoded}",
        len(content),
    )


def _usage(value: dict[str, Any] | None) -> dict[str, Any]:
    raw = value or {}
    return {
        "prompt_token_count": raw.get("prompt_tokens"),
        "candidates_token_count": raw.get("completion_tokens"),
        "thoughts_token_count": None,
        "cached_content_token_count": None,
        "total_token_count": raw.get("total_tokens"),
        "raw": raw,
    }


def _request(
    image_paths: Sequence[Path],
    *,
    image_ids: Sequence[int] | None,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    phase: str,
    timeout_seconds: float = 180.0,
) -> ProviderTextResult:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY ortam değişkeni tanımlı değil."
        )
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("En az bir görsel gereklidir.")
    ids = (
        [int(value) for value in image_ids]
        if image_ids is not None
        else None
    )
    if ids is not None and len(ids) != len(paths):
        raise ValueError(
            "Görseller ile görsel kimliklerinin sayısı eşit olmalıdır."
        )

    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt}
    ]
    total_bytes = 0
    for index, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(f"Görsel bulunamadı: {path}")
        data_url, byte_count = _data_url(path)
        total_bytes += byte_count
        if ids is not None:
            content.append(
                {
                    "type": "text",
                    "text": f"Image {ids[index]}:",
                }
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": (
            temperature if temperature > 0 else 1e-8
        ),
        "max_completion_tokens": max_tokens,
        "stream": False,
        "n": 1,
    }
    request = urllib.request.Request(
        GROQ_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started_at = utc_now_iso()
    timer = start_timer()
    call: dict[str, Any]
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        error = parsed.get("error")
        if error:
            raise GroqAPIError(
                "Groq yanıt gövdesi hata içeriyor: "
                + json.dumps(error, ensure_ascii=False)[:4000]
            )
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GroqAPIError(
                "Groq cevabında choices alanı bulunamadı."
            )
        text = str(
            (choices[0].get("message") or {}).get("content") or ""
        ).strip()
        if not text:
            raise GroqAPIError("Groq boş cevap döndürdü.")
        wall = elapsed_seconds(timer)
        call = {
            "phase": phase,
            "provider": "groq",
            "model": model,
            "response_model": parsed.get("model"),
            "temperature": temperature,
            "success": True,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "api_call_wall_seconds": wall,
            "input_image_count": len(paths),
            "input_image_bytes": total_bytes,
            "finish_reason": choices[0].get("finish_reason"),
            "usage": _usage(parsed.get("usage")),
        }
        return ProviderTextResult(text, call)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        error = GroqAPIError(
            f"Groq HTTP {exc.code}: {body[:4000]}"
        )
        error.status_code = exc.code
        caught: Exception = error
    except Exception as exc:
        caught = exc

    call = {
        "phase": phase,
        "provider": "groq",
        "model": model,
        "temperature": temperature,
        "success": False,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "api_call_wall_seconds": elapsed_seconds(timer),
        "input_image_count": len(paths),
        "input_image_bytes": total_bytes,
        "usage": _usage(None),
    }
    try:
        setattr(caught, "provider_call_record", call)
        setattr(caught, "groq_call_record", call)
    except Exception:
        pass
    raise caught


class GroqProvider(ProviderAdapter):
    provider_id = "groq"
    default_candidate_strategy = "independent_calls"
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=False,
        max_images_per_request=5,
        max_native_choices=1,
    )

    def __init__(self, model: str) -> None:
        self.model_alias = model
        self.resolved_model = model

    def request_route(
        self,
        image_path: Path,
        *,
        prompt: str,
        temperature: float,
        phase: str,
    ) -> ProviderTextResult:
        return _request(
            [image_path],
            image_ids=None,
            prompt=prompt,
            model=self.resolved_model,
            temperature=temperature,
            max_tokens=8192,
            phase=phase,
        )

    def request_candidates(
        self,
        image_path: Path,
        *,
        problem: ProblemInstance,
        candidate_count: int,
        temperature: float,
        strategy: str,
    ) -> ProviderCandidatesResult:
        self.validate_candidate_count(candidate_count)
        if strategy == "auto":
            strategy = "independent_calls"
        if strategy != "independent_calls":
            raise ValueError(
                "Groq Chat Completions API yalnız n=1 desteklediği "
                "için independent_calls kullanılmalıdır."
            )
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        for candidate_id in range(1, candidate_count + 1):
            try:
                response = _request(
                    [image_path],
                    image_ids=None,
                    prompt=critic_prompt(problem),
                    model=self.resolved_model,
                    temperature=temperature,
                    max_tokens=8192,
                    phase=(
                        "critic_candidate_generation_"
                        f"{candidate_id:02d}"
                    ),
                )
            except Exception as exc:
                try:
                    setattr(
                        exc,
                        "provider_call_records",
                        [
                            *calls,
                            *(
                                [exc.provider_call_record]
                                if isinstance(
                                    getattr(
                                        exc,
                                        "provider_call_record",
                                        None,
                                    ),
                                    dict,
                                )
                                else []
                            ),
                        ],
                    )
                except Exception:
                    pass
                raise
            texts.append(response.text)
            calls.append(response.api_call)
        summary = summarize_api_calls(calls)
        aggregate = {
            "phase": "critic_candidate_generation",
            "provider": self.provider_id,
            "model": self.resolved_model,
            "temperature": temperature,
            "success": True,
            "strategy": strategy,
            "http_request_count": len(calls),
            "api_call_wall_seconds": summary[
                "total_api_call_wall_seconds"
            ],
            "requested_candidate_count": candidate_count,
            "returned_candidate_count": len(texts),
            "usage": {
                "total_token_count": summary["total_token_count"],
            },
        }
        return ProviderCandidatesResult(texts, aggregate, calls)

    def request_scorer(
        self,
        image_paths: Sequence[Path],
        *,
        problem: ProblemInstance,
        image_ids: Sequence[int],
    ) -> ProviderTextResult:
        self.validate_candidate_count(len(image_paths))
        return _request(
            image_paths,
            image_ids=image_ids,
            prompt=scorer_prompt(problem, image_ids),
            model=self.resolved_model,
            temperature=0.0,
            max_tokens=4096,
            phase="visual_scorer",
        )
