"""Tek çağrılık görsel CVRP deneyini çalıştır."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.exact_solver import solve_exact_cvrp

from src.providers import get_provider
from src.providers.gemini import DEFAULT_GEMINI_MODEL, GeminiClientError

from src.instances import build_capacity_demo_10
from src.model_contract import (
    ModelResponseParseError,
    build_solver_prompt,
    parse_model_response,
)
from src.rendering import DemandEncoding, render_problem
from src.validation import evaluate_solution


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_RUN_ID = "capacity_demo_10_01"


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
    )
    parser.add_argument(
        "--encoding",
        choices=[
            encoding.value
            for encoding in DemandEncoding
        ],
        default=DemandEncoding.NUMERIC.value,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Görseli, prompt'u ve baseline'ı hazırlar; "
            "API çağrısı yapmaz."
        ),
    )
    return parser.parse_args(argv)


def _normalize_path_component(
    value: str,
    *,
    field_name: str,
) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(
            f"{field_name} boş olamaz."
        )

    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        stripped,
    ).strip("-.")

    if not normalized:
        raise ValueError(
            f"{field_name} geçerli bir dosya adı üretmedi."
        )

    return normalized


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _relative(
    path: Path,
    run_dir: Path,
) -> str:
    return path.resolve().relative_to(
        run_dir.resolve()
    ).as_posix()


def _gap_percent(
    *,
    candidate_distance: float | None,
    reference_distance: float,
    candidate_valid: bool,
) -> float | None:
    if (
        not candidate_valid
        or candidate_distance is None
    ):
        return None

    return (
        (
            candidate_distance
            - reference_distance
        )
        / reference_distance
        * 100.0
    )


def execute_experiment(
    *,
    run_id: str,
    output_dir: Path | str,
    model: str = DEFAULT_GEMINI_MODEL,
    encoding: DemandEncoding | str = (
        DemandEncoding.NUMERIC
    ),
    validate_only: bool = False,
    client: Any | None = None,
) -> tuple[dict[str, Any], Path]:
    """Deney girdilerini hazırla ve gerekirse tek API çağrısı yap."""

    normalized_run_id = _normalize_path_component(
        run_id,
        field_name="Run ID",
    )
    normalized_model = _normalize_path_component(
        model,
        field_name="Model adı",
    )
    normalized_encoding = DemandEncoding(encoding)

    run_dir = (
        Path(output_dir)
        / "runs"
        / normalized_run_id
    )
    input_dir = run_dir / "inputs"
    baseline_dir = run_dir / "baseline"
    method_dir = (
        run_dir
        / "providers"
        / "gemini"
        / normalized_model
        / normalized_encoding.value
    )

    problem_path = input_dir / "problem.json"
    image_path = (
        input_dir
        / f"problem_{normalized_encoding.value}.png"
    )
    baseline_path = (
        baseline_dir
        / "exact_results.json"
    )
    prompt_path = method_dir / "prompt.txt"
    result_path = (
        method_dir
        / "single_call_results.json"
    )

    problem = build_capacity_demo_10()
    exact_solution = solve_exact_cvrp(problem)
    prompt = build_solver_prompt(
        encoding=normalized_encoding
    )

    render_problem(
        problem,
        image_path,
        encoding=normalized_encoding,
    )
    _write_json(
        problem_path,
        problem.to_dict(),
    )
    _write_json(
        baseline_path,
        exact_solution.to_dict(),
    )
    prompt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    prompt_path.write_text(
        prompt + "\n",
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": "visual_cvrp_single_call",
        "run_id": normalized_run_id,
        "status": (
            "validated_only"
            if validate_only
            else "pending"
        ),
        "provider": "gemini",
        "model": model.strip(),
        "encoding": normalized_encoding.value,
        "api_call_performed": False,
        "problem": problem.to_dict(),
        "exact_baseline": exact_solution.to_dict(),
        "prompt": prompt,
        "artifacts": {
            "problem": _relative(
                problem_path,
                run_dir,
            ),
            "problem_image": _relative(
                image_path,
                run_dir,
            ),
            "exact_baseline": _relative(
                baseline_path,
                run_dir,
            ),
            "prompt": _relative(
                prompt_path,
                run_dir,
            ),
        },
        "model_response": None,
        "parsed_solution": None,
        "validation": None,
        "optimality_gap_percent": None,
        "error": None,
    }

    if validate_only:
        _write_json(result_path, result)
        return result, result_path

    gemini_client = (
        client
        if client is not None
        else get_provider("gemini", model=model)
    )

    try:
        model_response = gemini_client.generate(
            prompt=prompt,
            image_path=image_path,
        )
    except GeminiClientError as error:
        result["status"] = "request_failed"
        result["api_call_performed"] = True
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _write_json(result_path, result)
        return result, result_path

    result["api_call_performed"] = True
    result["model_response"] = (
        model_response.to_dict()
    )

    try:
        parsed_solution = parse_model_response(
            model_response.text
        )
    except ModelResponseParseError as error:
        result["status"] = "parse_failed"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _write_json(result_path, result)
        return result, result_path

    evaluation = evaluate_solution(
        problem,
        parsed_solution.routes,
    )
    result["status"] = "completed"
    result["parsed_solution"] = (
        parsed_solution.to_dict()
    )
    result["validation"] = evaluation.to_dict()
    result["optimality_gap_percent"] = (
        _gap_percent(
            candidate_distance=(
                evaluation.total_distance
            ),
            reference_distance=(
                exact_solution.total_distance
            ),
            candidate_valid=evaluation.valid,
        )
    )

    _write_json(result_path, result)
    return result, result_path


def main(
    argv: list[str] | None = None,
) -> None:
    args = parse_args(argv)

    result, result_path = execute_experiment(
        run_id=args.run_id,
        output_dir=args.output_dir,
        model=args.model,
        encoding=args.encoding,
        validate_only=args.validate_only,
    )

    print("Görsel CVRP deney hazırlığı tamamlandı.")
    print(f"Run ID: {result['run_id']}")
    print(f"Problem: {result['problem']['name']}")
    print(f"Kodlama: {result['encoding']}")
    print(f"Model: {result['model']}")
    print(
        "Kesin optimum mesafe: "
        f"{result['exact_baseline']['total_distance']}"
    )

    if args.validate_only:
        print(
            "API çağrısı yapılmadı ve kota kullanılmadı."
        )
    elif result["status"] == "completed":
        validation = result["validation"]
        print(
            "Model çözümü geçerli: "
            f"{validation['valid']}"
        )
        print(
            "Model mesafesi: "
            f"{validation['total_distance']}"
        )
        print(
            "Optimumluk gap yüzdesi: "
            f"{result['optimality_gap_percent']}"
        )
    else:
        print(
            "Deney tamamlanamadı: "
            f"{result['error']['message']}"
        )

    print(f"Sonuç dosyası: {result_path}")

    if result["status"] in {
        "request_failed",
        "parse_failed",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
