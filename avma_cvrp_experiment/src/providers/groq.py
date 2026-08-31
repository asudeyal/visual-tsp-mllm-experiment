"""Groq multimodal adapter. The selected model must support vision."""

from .http_openai_compatible import OpenAICompatibleVisionProvider


class GroqProvider(OpenAICompatibleVisionProvider):
    provider_id = "groq"
    api_key_env = "GROQ_API_KEY"
    base_url = "https://api.groq.com/openai/v1/chat/completions"
