"""Mistral Chat Completions vision API adaptörü."""

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


MISTRAL_CHAT_URL = (
    "https://api.mistral.ai/v1/chat/completions"
)
MISTRAL_ROUTE_MAX_TOKENS = 4096
MISTRAL_SCORER_MAX_TOKENS = 2048


class MistralAPIError(RuntimeError):
    """Mistral HTTP veya cevap biçimi hatası."""


def _request_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _mime_type(path: Path) -> str:
    value = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower())

    if value is None:
        raise ValueError(
            "Mistral için yalnız PNG/JPEG desteklenir."
        )

    return value


def _data_url(path: Path) -> tuple[str, int]:
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode("ascii")

    return (
        f"data:{_mime_type(path)};base64,{encoded}",
        len(content),
    )


def _usage(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = value or {}

    return {
        "prompt_token_count": raw.get(
            "prompt_tokens"
        ),
        "candidates_token_count": raw.get(
            "completion_tokens"
        ),
        "thoughts_token_count": None,
        "cached_content_token_count": None,
        "total_token_count": raw.get(
            "total_tokens"
        ),
        "raw": raw,
    }


def _response_text(
    choice: dict[str, Any],
) -> str:
    message = choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if not isinstance(item, dict):
                continue

            text = item.get("text")

            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

        return "\n".join(parts).strip()

    return ""


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
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY ortam değişkeni "
            "tanımlı değil."
        )

    paths = [Path(path) for path in image_paths]

    if not paths:
        raise ValueError(
            "En az bir görsel gereklidir."
        )

    ids = (
        [int(value) for value in image_ids]
        if image_ids is not None
        else None
    )

    if ids is not None and len(ids) != len(paths):
        raise ValueError(
            "Görseller ile görsel kimliklerinin "
            "sayısı eşit olmalıdır."
        )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    total_bytes = 0

    for index, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(
                f"Görsel bulunamadı: {path}"
            )

        image_url, image_bytes = _data_url(path)
        total_bytes += image_bytes

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
                "image_url": image_url,
            }
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    request = urllib.request.Request(
        MISTRAL_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=_request_headers(api_key),
        method="POST",
    )

    started_at = utc_now_iso()
    timer = start_timer()

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            body = response.read().decode("utf-8")

        parsed = json.loads(body)

        response_error = parsed.get("error")

        if response_error:
            raise MistralAPIError(
                "Mistral yanıt gövdesi hata içeriyor: "
                + json.dumps(
                    response_error,
                    ensure_ascii=False,
                )[:4000]
            )

        choices = parsed.get("choices")

        if (
            not isinstance(choices, list)
            or not choices
        ):
            raise MistralAPIError(
                "Mistral cevabında choices alanı "
                "bulunamadı."
            )

        text = _response_text(choices[0])

        if not text:
            raise MistralAPIError(
                "Mistral boş cevap döndürdü."
            )

        call = {
            "phase": phase,
            "provider": "mistral",
            "model": model,
            "response_model": parsed.get("model"),
            "temperature": temperature,
            "success": True,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "api_call_wall_seconds": (
                elapsed_seconds(timer)
            ),
            "input_image_count": len(paths),
            "input_image_bytes": total_bytes,
            "max_tokens": max_tokens,
            "finish_reason": choices[0].get(
                "finish_reason"
            ),
            "usage": _usage(parsed.get("usage")),
        }

        return ProviderTextResult(text, call)

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        caught: Exception = MistralAPIError(
            f"Mistral HTTP {exc.code}: "
            f"{body[:4000]}"
        )
        caught.status_code = exc.code

    except urllib.error.URLError as exc:
        caught = MistralAPIError(
            f"Mistral bağlantı hatası: {exc.reason}"
        )

    except Exception as exc:
        caught = exc

    call = {
        "phase": phase,
        "provider": "mistral",
        "model": model,
        "temperature": temperature,
        "success": False,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "api_call_wall_seconds": (
            elapsed_seconds(timer)
        ),
        "input_image_count": len(paths),
        "input_image_bytes": total_bytes,
        "max_tokens": max_tokens,
        "usage": _usage(None),
    }

    try:
        setattr(
            caught,
            "provider_call_record",
            call,
        )
        setattr(
            caught,
            "mistral_call_record",
            call,
        )
    except Exception:
        pass

    raise caught


class MistralProvider(ProviderAdapter):
    provider_id = "mistral"
    default_candidate_strategy = (
        "independent_calls"
    )
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

    @property
    def model_metadata(self) -> dict[str, Any]:
        metadata = super().model_metadata

        metadata["inference_settings"] = {
            "route_max_tokens": (
                MISTRAL_ROUTE_MAX_TOKENS
            ),
            "scorer_max_tokens": (
                MISTRAL_SCORER_MAX_TOKENS
            ),
            "candidate_strategy": (
                self.default_candidate_strategy
            ),
        }

        return metadata

    def request_route(
        self,
        image_path: Path,
        *,
        prompt: str,
        temperature: float,
        phase: str,
    ) -> ProviderTextResult:
        return self._execute_request(
            lambda: _request(
                [image_path],
                image_ids=None,
                prompt=prompt,
                model=self.resolved_model,
                temperature=temperature,
                max_tokens=(
                    MISTRAL_ROUTE_MAX_TOKENS
                ),
                phase=phase,
            ),
            label=f"mistral:{phase}",
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
        self.validate_candidate_count(
            candidate_count
        )

        if strategy == "auto":
            strategy = "independent_calls"

        if strategy != "independent_calls":
            raise ValueError(
                "Mistral provider için "
                "independent_calls kullanılmalıdır."
            )

        texts: list[str] = []
        calls: list[dict[str, Any]] = []

        for candidate_id in range(
            1,
            candidate_count + 1,
        ):
            phase = (
                "critic_candidate_generation_"
                f"{candidate_id:02d}"
            )

            try:
                response = self._execute_request(
                    lambda: _request(
                        [image_path],
                        image_ids=None,
                        prompt=critic_prompt(problem),
                        model=self.resolved_model,
                        temperature=temperature,
                        max_tokens=(
                            MISTRAL_ROUTE_MAX_TOKENS
                        ),
                        phase=phase,
                    ),
                    label=f"mistral:{phase}",
                )

            except Exception as exc:
                failed_records = getattr(
                    exc,
                    "provider_call_records",
                    None,
                )

                if not isinstance(
                    failed_records,
                    list,
                ):
                    single_record = getattr(
                        exc,
                        "provider_call_record",
                        None,
                    )

                    failed_records = (
                        [single_record]
                        if isinstance(
                            single_record,
                            dict,
                        )
                        else []
                    )

                try:
                    setattr(
                        exc,
                        "provider_call_records",
                        [
                            *calls,
                            *failed_records,
                        ],
                    )
                except Exception:
                    pass

                raise

            texts.append(response.text)
            calls.append(response.api_call)

        summary = summarize_api_calls(calls)

        aggregate = {
            "phase": (
                "critic_candidate_generation"
            ),
            "provider": self.provider_id,
            "model": self.resolved_model,
            "temperature": temperature,
            "success": True,
            "strategy": strategy,
            "http_request_count": len(calls),
            "api_call_wall_seconds": summary[
                "total_api_call_wall_seconds"
            ],
            "requested_candidate_count": (
                candidate_count
            ),
            "returned_candidate_count": len(texts),
            "usage": {
                "total_token_count": summary[
                    "total_token_count"
                ],
            },
        }

        return ProviderCandidatesResult(
            texts,
            aggregate,
            calls,
        )

    def request_scorer(
        self,
        image_paths: Sequence[Path],
        *,
        problem: ProblemInstance,
        image_ids: Sequence[int],
    ) -> ProviderTextResult:
        self.validate_candidate_count(
            len(image_paths)
        )

        return self._execute_request(
            lambda: _request(
                image_paths,
                image_ids=image_ids,
                prompt=scorer_prompt(
                    problem,
                    image_ids,
                ),
                model=self.resolved_model,
                temperature=0.0,
                max_tokens=(
                    MISTRAL_SCORER_MAX_TOKENS
                ),
                phase="visual_scorer",
            ),
            label="mistral:visual_scorer",
        )