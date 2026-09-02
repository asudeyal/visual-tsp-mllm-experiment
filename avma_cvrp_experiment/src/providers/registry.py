"""Provider registry. Imports are lazy so API SDKs are only required when used."""

from __future__ import annotations

from ..config import ProviderConfig
from .base import ProviderAdapter


def create_provider(config: ProviderConfig) -> ProviderAdapter:
    name = config.name.strip().lower()
    kwargs = {
        "model": config.model,
        "timeout_seconds": config.timeout_seconds,
        "request_retries": config.request_retries,
    }
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(**kwargs)
    if name == "groq":
        from .groq import GroqProvider
        return GroqProvider(**kwargs)
    if name == "mistral":
        from .mistral import MistralProvider
        return MistralProvider(**kwargs)
    if name == "openrouter":
        from .openrouter import OpenRouterProvider
        return OpenRouterProvider(**kwargs)
    if name == "cohere":
        from .cohere import CohereProvider
        return CohereProvider(**kwargs)
    raise ValueError(f"Desteklenmeyen provider: {config.name}")


def supported_providers() -> tuple[str, ...]:
    return ("gemini", "groq", "mistral", "openrouter", "cohere")
