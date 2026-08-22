"""Tek çağrılık görsel CVRP runner testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_experiment import execute_experiment
from src.gemini_client import (
    GeminiClientError,
    GeminiModelResponse,
)


EXACT_RESPONSE = """
{
  "routes": [
    [0, 9, 2, 1, 0],
    [0, 8, 5, 3, 0],
    [0, 7, 6, 4, 0]
  ]
}
"""


class FakeGeminiClient:
    def __init__(
        self,
        *,
        text: str = EXACT_RESPONSE,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        prompt: str,
        image_path: Path | str,
    ) -> GeminiModelResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "image_path": Path(image_path),
            }
        )

        if self.error is not None:
            raise self.error

        return GeminiModelResponse(
            model="gemini-test",
            text=self.text,
            elapsed_seconds=1.25,
            prompt_token_count=100,
            output_token_count=20,
            thoughts_token_count=5,
            total_token_count=125,
        )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_validate_only_creates_artifacts_without_api(
    tmp_path: Path,
) -> None:
    fake_client = FakeGeminiClient()

    result, result_path = execute_experiment(
        run_id="validation_run",
        output_dir=tmp_path,
        model="gemini-test",
        validate_only=True,
        client=fake_client,
    )

    run_dir = tmp_path / "runs" / "validation_run"

    assert result["status"] == "validated_only"
    assert result["api_call_performed"] is False
    assert fake_client.calls == []
    assert result_path.is_file()
    assert (
        run_dir
        / "inputs"
        / "problem_numeric.png"
    ).is_file()
    assert (
        run_dir
        / "baseline"
        / "exact_results.json"
    ).is_file()
    assert read_json(result_path)["status"] == (
        "validated_only"
    )


def test_size_validate_only_creates_size_artifacts(
    tmp_path: Path,
) -> None:
    fake_client = FakeGeminiClient()

    result, result_path = execute_experiment(
        run_id="size_validation_run",
        output_dir=tmp_path,
        model="gemini-test",
        encoding="size",
        validate_only=True,
        client=fake_client,
    )

    run_dir = (
        tmp_path / "runs" / "size_validation_run"
    )

    assert result["encoding"] == "size"
    assert result["api_call_performed"] is False
    assert fake_client.calls == []
    assert (
        run_dir / "inputs" / "problem_size.png"
    ).is_file()
    assert result_path == (
        run_dir
        / "providers"
        / "gemini"
        / "gemini-test"
        / "size"
        / "single_call_results.json"
    )
    assert "circle diameter" in (
        run_dir
        / "providers"
        / "gemini"
        / "gemini-test"
        / "size"
        / "prompt.txt"
    ).read_text(encoding="utf-8")


def test_color_intensity_validate_only_creates_artifacts(
    tmp_path: Path,
) -> None:
    fake_client = FakeGeminiClient()

    result, result_path = execute_experiment(
        run_id="color_intensity_validation_run",
        output_dir=tmp_path,
        model="gemini-test",
        encoding="color_intensity",
        validate_only=True,
        client=fake_client,
    )

    run_dir = (
        tmp_path
        / "runs"
        / "color_intensity_validation_run"
    )

    assert result["encoding"] == "color_intensity"
    assert result["api_call_performed"] is False
    assert fake_client.calls == []
    assert (
        run_dir
        / "inputs"
        / "problem_color_intensity.png"
    ).is_file()
    assert result_path == (
        run_dir
        / "providers"
        / "gemini"
        / "gemini-test"
        / "color_intensity"
        / "single_call_results.json"
    )
    prompt = (
        run_dir
        / "providers"
        / "gemini"
        / "gemini-test"
        / "color_intensity"
        / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert "blue fill intensity" in prompt
    assert "lightest blue at zero demand" in prompt
    assert "darkest blue at vehicle capacity Q" in (
        prompt
    )


def test_bar_length_validate_only_creates_artifacts(
    tmp_path: Path,
) -> None:
    fake_client = FakeGeminiClient()

    result, result_path = execute_experiment(
        run_id="bar_length_validation_run",
        output_dir=tmp_path,
        model="gemini-test",
        encoding="bar_length",
        validate_only=True,
        client=fake_client,
    )

    run_dir = (
        tmp_path
        / "runs"
        / "bar_length_validation_run"
    )

    assert result["encoding"] == "bar_length"
    assert result["api_call_performed"] is False
    assert fake_client.calls == []
    assert (
        run_dir
        / "inputs"
        / "problem_bar_length.png"
    ).is_file()
    assert result_path == (
        run_dir
        / "providers"
        / "gemini"
        / "gemini-test"
        / "bar_length"
        / "single_call_results.json"
    )
    prompt = (
        run_dir
        / "providers"
        / "gemini"
        / "gemini-test"
        / "bar_length"
        / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert "fixed-width bar below" in prompt
    assert "filled fraction is demand divided by Q" in (
        prompt
    )


def test_completed_response_is_validated_and_scored(
    tmp_path: Path,
) -> None:
    fake_client = FakeGeminiClient()

    result, result_path = execute_experiment(
        run_id="completed_run",
        output_dir=tmp_path,
        model="gemini-test",
        client=fake_client,
    )

    assert result["status"] == "completed"
    assert result["api_call_performed"] is True
    assert result["validation"]["valid"] is True
    assert result["optimality_gap_percent"] == (
        pytest.approx(0.0)
    )
    assert len(fake_client.calls) == 1
    assert read_json(result_path)["status"] == (
        "completed"
    )


def test_invalid_solution_is_not_repaired(
    tmp_path: Path,
) -> None:
    raw_response = (
        '{"routes": [[0, 2, 4, 8, 0], '
        '[0, 1, 3, 5, 6, 0], [0, 7, 9, 0]]}'
    )
    fake_client = FakeGeminiClient(
        text=raw_response
    )

    result, _ = execute_experiment(
        run_id="invalid_run",
        output_dir=tmp_path,
        model="gemini-test",
        client=fake_client,
    )

    assert result["status"] == "completed"
    assert result["validation"]["valid"] is False
    assert result["validation"][
        "total_capacity_excess"
    ] == 3
    assert result["optimality_gap_percent"] is None
    assert result["parsed_solution"]["routes"] == [
        [0, 2, 4, 8, 0],
        [0, 1, 3, 5, 6, 0],
        [0, 7, 9, 0],
    ]


def test_parse_failure_is_persisted(
    tmp_path: Path,
) -> None:
    fake_client = FakeGeminiClient(
        text="not json"
    )

    result, result_path = execute_experiment(
        run_id="parse_failure_run",
        output_dir=tmp_path,
        model="gemini-test",
        client=fake_client,
    )

    assert result["status"] == "parse_failed"
    assert result["model_response"] is not None
    assert result["parsed_solution"] is None
    assert read_json(result_path)["error"][
        "type"
    ] == "ModelResponseParseError"


def test_request_failure_is_persisted(
    tmp_path: Path,
) -> None:
    fake_client = FakeGeminiClient(
        error=GeminiClientError("quota exceeded")
    )

    result, result_path = execute_experiment(
        run_id="request_failure_run",
        output_dir=tmp_path,
        model="gemini-test",
        client=fake_client,
    )

    assert result["status"] == "request_failed"
    assert result["api_call_performed"] is True
    assert result["model_response"] is None
    assert read_json(result_path)["error"][
        "message"
    ] == "quota exceeded"
