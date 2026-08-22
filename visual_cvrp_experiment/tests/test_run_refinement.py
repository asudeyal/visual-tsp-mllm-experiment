"""Geri bildirimli CVRP iyileştirme runner testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_refinement import (
    build_refinement_prompt,
    execute_refinement,
)
from src.gemini_client import (
    GeminiClientError,
    GeminiModelResponse,
)


EXACT_RESPONSE = """
{"routes":[
  [0,9,2,1,0],
  [0,8,5,3,0],
  [0,7,6,4,0]
]}
"""

VALID_SUBOPTIMAL_RESPONSE = """
{"routes":[
  [0,2,3,5,0],
  [0,4,1,9,0],
  [0,6,7,8,0]
]}
"""

INVALID_RESPONSE = """
{"routes":[
  [0,8,5,7,0],
  [0,4,1,6,0],
  [0,2,9,3,0]
]}
"""


class SequencedGeminiClient:
    def __init__(
        self,
        responses: dict[str, list[str | Exception]],
    ) -> None:
        self.responses = {
            key: list(value)
            for key, value in responses.items()
        }
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        prompt: str,
        image_path: Path | str,
    ) -> GeminiModelResponse:
        path = Path(image_path)
        encoding = path.stem.removeprefix("problem_")
        self.calls.append(
            {
                "encoding": encoding,
                "prompt": prompt,
                "image_path": path,
            }
        )
        item = self.responses[encoding].pop(0)
        if isinstance(item, Exception):
            raise item
        return GeminiModelResponse(
            model="gemini-test",
            text=item,
            elapsed_seconds=1.5,
            prompt_token_count=100,
            output_token_count=25,
            thoughts_token_count=50,
            total_token_count=175,
        )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_validate_only_prepares_both_methods_without_api(
    tmp_path: Path,
) -> None:
    client = SequencedGeminiClient(
        {
            "bar_length": [EXACT_RESPONSE],
            "color_intensity": [EXACT_RESPONSE],
        }
    )

    manifest, manifest_path = execute_refinement(
        run_id="refinement_validation",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        validate_only=True,
        client=client,
    )

    run_dir = tmp_path / "runs" / "refinement_validation"
    assert manifest["status"] == "validated_only"
    assert manifest["maximum_total_calls"] == 8
    assert client.calls == []
    assert manifest_path.is_file()
    assert (
        run_dir / "inputs" / "problem_bar_length.png"
    ).is_file()
    assert (
        run_dir
        / "inputs"
        / "problem_color_intensity.png"
    ).is_file()
    assert not list(
        run_dir.glob(
            "providers/*/*/*/iteration_*"
        )
    )


def test_fresh_calls_refine_in_rounds_and_stop_at_exact(
    tmp_path: Path,
) -> None:
    client = SequencedGeminiClient(
        {
            "bar_length": [
                VALID_SUBOPTIMAL_RESPONSE,
                EXACT_RESPONSE,
            ],
            "color_intensity": [
                INVALID_RESPONSE,
                VALID_SUBOPTIMAL_RESPONSE,
                EXACT_RESPONSE,
            ],
        }
    )

    manifest, _ = execute_refinement(
        run_id="fresh_refinement",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        client=client,
    )

    assert manifest["status"] == "completed"
    assert len(client.calls) == 5
    assert [
        call["encoding"]
        for call in client.calls
    ] == [
        "bar_length",
        "color_intensity",
        "color_intensity",
        "bar_length",
        "color_intensity",
    ]

    bar_summary = next(
        item
        for item in manifest["methods"]
        if item["encoding"] == "bar_length"
    )
    intensity_summary = next(
        item
        for item in manifest["methods"]
        if item["encoding"] == "color_intensity"
    )
    assert bar_summary["status"] == "early_stopped"
    assert bar_summary["final_iteration"] == 2
    assert intensity_summary["status"] == "early_stopped"
    assert intensity_summary["final_iteration"] == 3

    initial_prompts = [
        str(call["prompt"])
        for call in client.calls
        if "Refinement iteration" not in str(call["prompt"])
    ]
    assert len(initial_prompts) == 2
    refinement_prompts = [
        str(call["prompt"])
        for call in client.calls
        if "Refinement iteration" in str(call["prompt"])
    ]
    assert len(refinement_prompts) == 3
    assert any(
        "Total capacity excess: 1" in prompt
        for prompt in refinement_prompts
    )
    assert any(
        "Total route distance:" in prompt
        for prompt in refinement_prompts
    )
    assert all(
        "419.7278467775164" not in prompt
        for prompt in refinement_prompts
    )
    assert all(
        "node 1 demand" not in prompt.lower()
        for prompt in refinement_prompts
    )

    invalid_path = (
        tmp_path
        / "runs"
        / "fresh_refinement"
        / "providers"
        / "gemini"
        / "gemini-test"
        / "color_intensity"
        / "iteration_01"
        / "iteration_results.json"
    )
    invalid_result = read_json(invalid_path)
    assert invalid_result["parsed_solution"]["routes"] == [
        [0, 8, 5, 7, 0],
        [0, 4, 1, 6, 0],
        [0, 2, 9, 3, 0],
    ]
    assert invalid_result["validation"]["valid"] is False


def test_refinement_prompt_rejects_unvalidated_input() -> None:
    with pytest.raises(ValueError):
        build_refinement_prompt(
            encoding="bar_length",
            previous_result={"status": "request_failed"},
            iteration=2,
        )


def test_request_failure_is_preserved_when_resumed(
    tmp_path: Path,
) -> None:
    failed_client = SequencedGeminiClient(
        {
            "bar_length": [
                GeminiClientError("temporary failure")
            ]
        }
    )
    first_manifest, _ = execute_refinement(
        run_id="resume_run",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        encodings=["bar_length"],
        max_refinement_iterations=1,
        client=failed_client,
    )
    assert first_manifest["status"] == "paused"

    success_client = SequencedGeminiClient(
        {"bar_length": [EXACT_RESPONSE]}
    )
    second_manifest, _ = execute_refinement(
        run_id="resume_run",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        encodings=["bar_length"],
        max_refinement_iterations=1,
        resume=True,
        client=success_client,
    )

    iteration_dir = (
        tmp_path
        / "runs"
        / "resume_run"
        / "providers"
        / "gemini"
        / "gemini-test"
        / "bar_length"
        / "iteration_01"
    )
    assert second_manifest["status"] == "completed"
    assert (
        iteration_dir / "request_failure_01.json"
    ).is_file()
    assert read_json(
        iteration_dir / "request_failure_01.json"
    )["error"]["message"] == "temporary failure"
    assert read_json(
        iteration_dir / "iteration_results.json"
    )["status"] == "completed"


def test_existing_results_require_resume(
    tmp_path: Path,
) -> None:
    client = SequencedGeminiClient(
        {"bar_length": [EXACT_RESPONSE]}
    )
    execute_refinement(
        run_id="protected_run",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        encodings=["bar_length"],
        client=client,
    )

    with pytest.raises(FileExistsError):
        execute_refinement(
            run_id="protected_run",
            historical_run_id="pilot_run",
            output_dir=tmp_path,
            model="gemini-test",
            encodings=["bar_length"],
            client=SequencedGeminiClient(
                {"bar_length": [EXACT_RESPONSE]}
            ),
        )


def test_existing_run_can_be_extended_with_size(
    tmp_path: Path,
) -> None:
    initial_client = SequencedGeminiClient(
        {
            "bar_length": [EXACT_RESPONSE],
            "color_intensity": [EXACT_RESPONSE],
        }
    )
    execute_refinement(
        run_id="extended_run",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        client=initial_client,
    )

    size_client = SequencedGeminiClient(
        {
            "bar_length": [],
            "color_intensity": [],
            "size": [EXACT_RESPONSE],
        }
    )
    manifest, _ = execute_refinement(
        run_id="extended_run",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        encodings=[
            "bar_length",
            "color_intensity",
            "size",
        ],
        resume=True,
        extend_encodings=True,
        client=size_client,
    )

    assert [
        call["encoding"]
        for call in size_client.calls
    ] == ["size"]
    assert manifest["encodings"] == [
        "bar_length",
        "color_intensity",
        "size",
    ]
    assert manifest["maximum_total_calls"] == 12
    assert manifest["actual_api_calls"] == 3
    size_summary = next(
        item
        for item in manifest["methods"]
        if item["encoding"] == "size"
    )
    assert size_summary["status"] == "early_stopped"
    assert size_summary["final_iteration"] == 1


def test_encoding_extension_requires_resume(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="yalnızca --resume",
    ):
        execute_refinement(
            run_id="extension_without_resume",
            historical_run_id="pilot_run",
            output_dir=tmp_path,
            model="gemini-test",
            encodings=[
                "bar_length",
                "color_intensity",
                "size",
            ],
            extend_encodings=True,
            client=SequencedGeminiClient(
                {"size": [EXACT_RESPONSE]}
            ),
        )


def test_existing_run_iteration_limit_can_be_extended(
    tmp_path: Path,
) -> None:
    initial_client = SequencedGeminiClient(
        {
            "bar_length": [
                VALID_SUBOPTIMAL_RESPONSE,
                VALID_SUBOPTIMAL_RESPONSE,
            ]
        }
    )
    execute_refinement(
        run_id="iteration_extension",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        encodings=["bar_length"],
        max_refinement_iterations=1,
        client=initial_client,
    )

    extension_client = SequencedGeminiClient(
        {
            "bar_length": [
                VALID_SUBOPTIMAL_RESPONSE,
                EXACT_RESPONSE,
            ]
        }
    )
    manifest, _ = execute_refinement(
        run_id="iteration_extension",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        encodings=["bar_length"],
        max_refinement_iterations=3,
        resume=True,
        client=extension_client,
    )

    assert len(extension_client.calls) == 2
    assert manifest["maximum_iterations_per_encoding"] == 4
    assert manifest["maximum_total_calls"] == 4
    assert manifest["actual_api_calls"] == 4
    method = manifest["methods"][0]
    assert method["status"] == "early_stopped"
    assert method["final_iteration"] == 4


def test_resume_cannot_reduce_iteration_limit(
    tmp_path: Path,
) -> None:
    execute_refinement(
        run_id="iteration_reduction",
        historical_run_id="pilot_run",
        output_dir=tmp_path,
        model="gemini-test",
        encodings=["bar_length"],
        max_refinement_iterations=3,
        client=SequencedGeminiClient(
            {
                "bar_length": [
                    VALID_SUBOPTIMAL_RESPONSE,
                    VALID_SUBOPTIMAL_RESPONSE,
                    VALID_SUBOPTIMAL_RESPONSE,
                    VALID_SUBOPTIMAL_RESPONSE,
                ]
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="maximum_iterations_per_encoding",
    ):
        execute_refinement(
            run_id="iteration_reduction",
            historical_run_id="pilot_run",
            output_dir=tmp_path,
            model="gemini-test",
            encodings=["bar_length"],
            max_refinement_iterations=1,
            resume=True,
            client=SequencedGeminiClient(
                {"bar_length": [EXACT_RESPONSE]}
            ),
        )
