"""Gemini görsel istemcisinin çevrimdışı testleri."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.providers.gemini import (
    GeminiClientError,
    GeminiVisionProvider,
)


class FakeUsageMetadata:
    prompt_token_count = 120
    candidates_token_count = 30
    thoughts_token_count = 10
    total_token_count = 160


class FakeResponse:
    def __init__(
        self,
        text: str | None,
    ) -> None:
        self.text = text
        self.usage_metadata = FakeUsageMetadata()


class FakeModels:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self,
        **kwargs: Any,
    ) -> FakeResponse:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        assert self.response is not None
        return self.response


class FakeClient:
    def __init__(
        self,
        models: FakeModels,
    ) -> None:
        self.models = models


class FakeClock:
    def __init__(
        self,
        values: list[float],
    ) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def create_png(path: Path) -> None:
    Image.new(
        "RGB",
        (32, 32),
        color="white",
    ).save(
        path,
        format="PNG",
    )


def test_generate_returns_text_and_usage(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "problem.png"
    create_png(image_path)

    fake_models = FakeModels(
        response=FakeResponse(
            '{"routes": [[0, 1, 0]]}'
        )
    )
    client = GeminiVisionProvider(
        model="gemini-test-model",
        temperature=0.0,
        client=FakeClient(fake_models),
        clock=FakeClock([10.0, 12.5]),
    )

    response = client.generate(
        prompt="Solve the problem.",
        image_path=image_path,
    )

    assert response.model == "gemini-test-model"
    assert response.text == (
        '{"routes": [[0, 1, 0]]}'
    )
    assert response.elapsed_seconds == pytest.approx(
        2.5
    )
    assert response.prompt_token_count == 120
    assert response.output_token_count == 30
    assert response.thoughts_token_count == 10
    assert response.total_token_count == 160

    assert len(fake_models.calls) == 1
    call = fake_models.calls[0]

    assert call["model"] == "gemini-test-model"
    assert call["contents"][0] == (
        "Solve the problem."
    )
    assert len(call["contents"]) == 2


def test_missing_image_is_rejected(
    tmp_path: Path,
) -> None:
    fake_models = FakeModels(
        response=FakeResponse("{}")
    )
    client = GeminiVisionProvider(
        client=FakeClient(fake_models)
    )

    with pytest.raises(FileNotFoundError):
        client.generate(
            prompt="Solve.",
            image_path=tmp_path / "missing.png",
        )

    assert fake_models.calls == []


def test_empty_response_is_rejected(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "problem.png"
    create_png(image_path)

    fake_models = FakeModels(
        response=FakeResponse("   ")
    )
    client = GeminiVisionProvider(
        client=FakeClient(fake_models),
        clock=FakeClock([1.0, 2.0]),
    )

    with pytest.raises(
        GeminiClientError,
        match="boş",
    ):
        client.generate(
            prompt="Solve.",
            image_path=image_path,
        )


def test_api_error_is_wrapped(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "problem.png"
    create_png(image_path)

    fake_models = FakeModels(
        error=RuntimeError("quota exceeded")
    )
    client = GeminiVisionProvider(
        client=FakeClient(fake_models),
        clock=FakeClock([1.0]),
    )

    with pytest.raises(
        GeminiClientError,
        match="quota exceeded",
    ):
        client.generate(
            prompt="Solve.",
            image_path=image_path,
        )


def test_invalid_temperature_is_rejected() -> None:
    fake_models = FakeModels(
        response=FakeResponse("{}")
    )

    with pytest.raises(
        ValueError,
        match="Temperature",
    ):
        GeminiVisionProvider(
            temperature=2.5,
            client=FakeClient(fake_models),
        )