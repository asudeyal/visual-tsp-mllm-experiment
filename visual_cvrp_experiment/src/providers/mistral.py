"""Mistral Vision (Pixtral vb.) modelleri için sağlayıcı (Direct HTTP)."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from src.providers.base import ModelResponse

DEFAULT_MISTRAL_MODEL = "pixtral-12b-2409"


class MistralVisionProvider:
    """Mistral Vision (Pixtral vb.) modelleri için HTTP tabanlı sağlayıcı."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MISTRAL_MODEL,
        temperature: float = 0.0,
        api_key: str | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.model = model.strip()
        self.temperature = temperature
        self._clock = clock
        
        # API anahtarını al
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise ValueError("Mistral API anahtarı (MISTRAL_API_KEY) bulunamadı.")
            
        self._api_key = key

    def generate(
        self,
        *,
        prompt: str,
        image_path: Path | str,
    ) -> ModelResponse:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Görsel bulunamadı: {path}")

        # Görseli Base64 formatına çevir
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        base64_image = f"data:image/png;base64,{encoded_string}"

        # Mistral API için istek paketini (payload) hazırla
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

        # HTTP başlıklarını (headers) ayarla
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        request = urllib.request.Request(
            "https://api.mistral.ai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        started_at = self._clock()
        
        try:
            # İnternet üzerinden API çağrısını doğrudan Python ile yapıyoruz
            with urllib.request.urlopen(request) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            raise RuntimeError(f"Mistral API çağrısı başarısız oldu: {error}") from error

        elapsed_seconds = max(0.0, self._clock() - started_at)
        
        # Gelen yanıtı ayıkla
        response_text = response_data["choices"][0]["message"]["content"]
        usage = response_data.get("usage", {})

        return ModelResponse(
            model=self.model,
            text=response_text.strip(),
            elapsed_seconds=elapsed_seconds,
            prompt_token_count=usage.get("prompt_tokens"),
            output_token_count=usage.get("completion_tokens"),
            thoughts_token_count=None,
            total_token_count=usage.get("total_tokens"),
        )