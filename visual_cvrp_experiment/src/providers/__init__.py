from __future__ import annotations

from src.providers.base import ModelResponse, VisionProvider
from src.providers.gemini import GeminiVisionProvider
from src.providers.groq import GroqVisionProvider
from src.providers.mistral import MistralVisionProvider
from src.providers.openrouter import OpenRouterVisionProvider
from src.providers.dashscope import DashScopeVisionProvider

def get_provider(provider_name: str, **kwargs) -> VisionProvider:
    """İsimden bağımsız olarak istenen sağlayıcı sınıfını döndürür."""
    name = provider_name.lower().strip()
    
    if name == "gemini":
        return GeminiVisionProvider(**kwargs)
    elif name == "groq":
        return GroqVisionProvider(**kwargs)
    elif name == "mistral":
        return MistralVisionProvider(**kwargs)
    elif name == "openrouter":
        return OpenRouterVisionProvider(**kwargs)
    elif name == "dashscope":
        return DashScopeVisionProvider(**kwargs)
    else:
        raise ValueError(f"Bilinmeyen model sağlayıcısı: {provider_name}")