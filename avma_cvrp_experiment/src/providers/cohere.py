"""Cohere multimodal adapter via the OpenAI-compatible API."""

from .http_openai_compatible import OpenAICompatibleVisionProvider


class CohereProvider(OpenAICompatibleVisionProvider):
    provider_id = "cohere"
    api_key_env = "COHERE_API_KEY"
    base_url = "https://api.cohere.ai/compatibility/v1/chat/completions"
