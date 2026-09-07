"""OpenRouter multimodal adapter. The selected routed model must support vision."""

from .http_openai_compatible import OpenAICompatibleVisionProvider


class OpenRouterProvider(OpenAICompatibleVisionProvider):
    provider_id = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1/chat/completions"
