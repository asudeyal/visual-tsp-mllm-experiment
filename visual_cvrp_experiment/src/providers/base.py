from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Tüm model sağlayıcıları için ortak gözlemlenebilir sonuç yapısı."""

    model: str
    text: str
    elapsed_seconds: float
    prompt_token_count: int | None
    output_token_count: int | None
    thoughts_token_count: int | None
    total_token_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "text": self.text,
            "elapsed_seconds": self.elapsed_seconds,
            "usage": {
                "prompt_token_count": self.prompt_token_count,
                "output_token_count": self.output_token_count,
                "thoughts_token_count": self.thoughts_token_count,
                "total_token_count": self.total_token_count,
            },
        }


class VisionProvider(Protocol):
    """Tüm sağlayıcıların (Gemini, Groq vb.) uygulaması gereken arayüz."""

    def generate(self, *, prompt: str, image_path: Path | str) -> ModelResponse:
        ...