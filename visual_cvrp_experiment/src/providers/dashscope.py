from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from src.providers.base import ModelResponse

DEFAULT_DASHSCOPE_MODEL = "qwen-vl-max"


class DashScopeVisionProvider:
    """DashScope (Qwen-VL vb.) modelleri için sağlayıcı."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_DASHSCOPE_MODEL,
        temperature: float = 0.0,
        api_key: str | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.model = model.strip()
        self.temperature = temperature
        self._clock = clock
        # Buraya DashScope istemcisi başlatma kodları eklenebilir

    def generate(
        self,
        *,
        prompt: str,
        image_path: Path | str,
    ) -> ModelResponse:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Görsel bulunamadı: {path}")

        started_at = self._clock()
        
        # --- BURAYA DASHSCOPE API ÇAĞRISI GELECEK ---
        response_text = '{"routes": []}' 
        # --------------------------------------------

        elapsed_seconds = max(0.0, self._clock() - started_at)

        return ModelResponse(
            model=self.model,
            text=response_text,
            elapsed_seconds=elapsed_seconds,
            prompt_token_count=None,
            output_token_count=None,
            thoughts_token_count=None,
            total_token_count=None,
        )