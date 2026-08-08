"""Sağlayıcı seçimi, varsayılan modeller ve çıktı yolları."""

from __future__ import annotations

import re
from pathlib import Path

from src.gemini import GEMINI_MODEL
from src.providers.base import ProviderAdapter
from src.providers.gemini_provider import GeminiProvider
from src.providers.openrouter_provider import OpenRouterProvider
from src.providers.mistral_provider import (
    MistralProvider,
)

DEFAULT_MODELS = {
    "gemini": GEMINI_MODEL,
    "groq": "qwen/qwen3.6-27b",
    "mistral": "mistral-small-latest",
}


def supported_providers() -> tuple[str, ...]:
    return (
        "gemini",
        "openrouter",
        "groq",
        "mistral",
    )


def model_slug(model: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", model).strip("-")
    if not value:
        raise ValueError("Model adı geçerli bir klasör adına dönüşmedi.")
    return value.lower()


def provider_model_root(
    run_dir: Path,
    provider_id: str,
    model_alias: str,
) -> Path:
    return (
        run_dir
        / "providers"
        / provider_id
        / model_slug(model_alias)
    )


def zero_shot_result_candidates(
    run_dir: Path,
    provider_id: str,
    model_alias: str,
) -> tuple[Path, ...]:
    """Yeni yolun ardından güvenli tarihsel initializer yollarını döndürür."""

    paths = [
        (
            provider_model_root(
                run_dir,
                provider_id,
                model_alias,
            )
            / "zero_shot"
            / "zero_shot_results.json"
        )
    ]
    if provider_id == "gemini":
        paths.append(
            run_dir / "zero_shot" / "zero_shot_results.json"
        )
    elif provider_id == "openrouter":
        paths.append(
            run_dir
            / "model_comparisons"
            / "openrouter"
            / model_alias
            / "zero_shot_results.json"
        )
    return tuple(paths)


def create_provider(
    provider_id: str,
    model: str | None,
) -> ProviderAdapter:
    provider_id = provider_id.lower()
    resolved = model or DEFAULT_MODELS.get(provider_id)

    if not resolved:
        raise ValueError(
            f"{provider_id} için --model belirtilmelidir."
        )

    if provider_id == "gemini":
        return GeminiProvider(resolved)

    if provider_id == "openrouter":
        return OpenRouterProvider(resolved)

    if provider_id == "mistral":
        return MistralProvider(resolved)

    if provider_id == "groq":
        from src.providers.groq_provider import (
            GroqProvider,
        )

        return GroqProvider(resolved)

    expected = ", ".join(supported_providers())

    raise ValueError(
        f"Bilinmeyen provider: {provider_id}. "
        f"Beklenen: {expected}"
    )
