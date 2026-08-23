from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from google import genai
from google.genai import types

from src.providers.base import ModelResponse

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


class GeminiClientError(RuntimeError):
    """Gemini çağrısı veya yanıtı kullanılamadı."""


def _optional_integer_attribute(value: Any, attribute_name: str) -> int | None:
    attribute = getattr(value, attribute_name, None)
    if attribute is None:
        return None
    return int(attribute)


class GeminiVisionProvider:
    """Tek görsel ve prompt ile Gemini çağrısı yapar."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.0,
        api_key: str | None = None,
        client: Any | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not model.strip():
            raise ValueError("Gemini model adı boş olamaz.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature 0 ile 2 arasında olmalıdır.")

        self.model = model.strip()
        self.temperature = temperature
        self._clock = clock

        if client is not None:
            self._client = client
        elif api_key is not None:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()

    def generate(
        self,
        *,
        prompt: str,
        image_path: Path | str,
    ) -> ModelResponse:
        """Görsel CVRP problemini Gemini'ye gönder."""
        if not prompt.strip():
            raise ValueError("Gemini prompt'u boş olamaz.")

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Gemini görseli bulunamadı: {path}")
        if path.suffix.lower() != ".png":
            raise ValueError("Gemini giriş görseli PNG olmalıdır.")

        image_part = types.Part.from_bytes(
            data=path.read_bytes(),
            mime_type="image/png",
        )

        started_at = self._clock()

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
        except Exception as error:
            raise GeminiClientError(
                f"Gemini API çağrısı başarısız oldu: {type(error).__name__}: {error}"
            ) from error

        elapsed_seconds = max(0.0, self._clock() - started_at)

        try:
            response_text = response.text
        except Exception as error:
            raise GeminiClientError(
                f"Gemini yanıt metni okunamadı: {type(error).__name__}: {error}"
            ) from error

        if not isinstance(response_text, str) or not response_text.strip():
            raise GeminiClientError("Gemini boş bir metin yanıtı döndürdü.")

        usage_metadata = getattr(response, "usage_metadata", None)

        return ModelResponse(
            model=self.model,
            text=response_text.strip(),
            elapsed_seconds=elapsed_seconds,
            prompt_token_count=_optional_integer_attribute(usage_metadata, "prompt_token_count"),
            output_token_count=_optional_integer_attribute(usage_metadata, "candidates_token_count"),
            thoughts_token_count=_optional_integer_attribute(usage_metadata, "thoughts_token_count"),
            total_token_count=_optional_integer_attribute(usage_metadata, "total_token_count"),
        )