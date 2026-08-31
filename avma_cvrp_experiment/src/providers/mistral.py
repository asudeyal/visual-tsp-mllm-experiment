"""Mistral multimodal adapter. The selected model must support vision."""

from .http_openai_compatible import OpenAICompatibleVisionProvider


class MistralProvider(OpenAICompatibleVisionProvider):
    provider_id = "mistral"
    api_key_env = "MISTRAL_API_KEY"
    base_url = "https://api.mistral.ai/v1/chat/completions"
