"""Sabit OpenRouter vision modellerini aynı TSP görselinde karşılaştırır."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

from src.core import (
    evaluate_route,
    normalize_run_id,
    plot_route,
    read_json,
    write_json,
)
from src.gemini import initializer_prompt, parse_route
from src.metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
    utc_now_iso,
)
from src.openrouter import (
    OPENROUTER_MODELS,
    request_route,
    resolve_model_alias,
)
from src.run_manifest import load_run_problem


ROOT = Path(__file__).resolve().parent
COMPARISON_SCHEMA_VERSION = "1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
    )
    parser.add_argument("--run-id", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        action="append",
        choices=tuple(OPENROUTER_MODELS),
        help=(
            "Çalıştırılacak kısa model adı. Birden fazla kez "
            "verilebilir."
        ),
    )
    selection.add_argument(
        "--all-models",
        action="store_true",
        help="Kayıtlı dört OpenRouter modelini sırayla çalıştırır.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="API çağrısı yapmadan manifest ve istekleri doğrular.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="API çağrısı yapmadan mevcut sonuçları özetler.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan model sonucunun üzerine yeni API sonucunu yazar.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
    )
    return parser.parse_args()


def _relative(path: Path, run_dir: Path) -> str:
    return path.resolve().relative_to(
        run_dir.resolve()
    ).as_posix()


def _model_dir(
    run_dir: Path,
    alias: str,
) -> Path:
    return (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / alias
    )


def _comparison_path(run_dir: Path) -> Path:
    return (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / "openrouter_model_comparison.json"
    )


def _status(result: dict[str, Any] | None) -> str:
    if result is None:
        return "missing"
    if result.get("errors"):
        return "failed"
    if result.get("validation", {}).get("is_valid") is True:
        return "valid"
    if "raw_response" in result:
        return "invalid"
    return "failed"


def output_format_compliant(raw_response: Any) -> bool | None:
    """Yanıtın yalnız istenen start/end rota bloğundan oluştuğunu ölçer."""

    if not isinstance(raw_response, str):
        return None
    return (
        re.fullmatch(
            (
                r"\s*<<start>>\s*"
                r"Salesman\s*1\s*:\s*[^\r\n]+"
                r"\s*<<end>>\s*"
            ),
            raw_response,
            flags=re.I,
        )
        is not None
    )


def is_ascending_node_id_route(
    route: Any,
    *,
    node_ids: list[int],
    depot_id: int,
) -> bool | None:
    """Rotanın görsel seçim yapmadan artan kimlik sırası olup olmadığını ölçer."""

    if not isinstance(route, list):
        return None
    visits = sorted(
        node_id
        for node_id in node_ids
        if node_id != depot_id
    )
    return route == [depot_id, *visits, depot_id]


def build_comparison(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Uzun model sonuçlarından kompakt karşılaştırma üretir."""

    rows: list[dict[str, Any]] = []
    problem = manifest["problem"]
    fingerprint = problem["fingerprint_sha256"]
    coordinates = problem.get("coordinates")
    node_ids = (
        [
            int(item["node_id"])
            for item in coordinates
        ]
        if isinstance(coordinates, list)
        else None
    )
    for alias, model in OPENROUTER_MODELS.items():
        path = (
            _model_dir(run_dir, alias)
            / "zero_shot_results.json"
        )
        result = read_json(path) if path.exists() else None
        if result is not None:
            result_fingerprint = (
                result.get("problem") or {}
            ).get("fingerprint_sha256")
            if result_fingerprint != fingerprint:
                raise ValueError(
                    f"Model sonucu farklı probleme ait: {path}"
                )
        call = (
            (result.get("api_calls") or [None])[0]
            if result is not None
            else None
        ) or {}
        usage = call.get("usage") or {}
        validation = (
            result.get("validation") or {}
            if result is not None
            else {}
        )
        errors = (
            result.get("errors") or []
            if result is not None
            else []
        )
        rows.append(
            {
                "alias": alias,
                "model": model,
                "status": _status(result),
                "output_format_compliant": (
                    output_format_compliant(
                        result.get("raw_response")
                    )
                    if result is not None
                    else None
                ),
                "is_valid": validation.get("is_valid"),
                "is_ascending_node_id_route": (
                    is_ascending_node_id_route(
                        result.get("route"),
                        node_ids=node_ids,
                        depot_id=int(problem["depot_id"]),
                    )
                    if result is not None
                    and node_ids is not None
                    else None
                ),
                "distance": (
                    result.get("distance")
                    if result is not None
                    else None
                ),
                "gap_to_reference_percent": (
                    result.get("gap_to_reference_percent")
                    if result is not None
                    else None
                ),
                "missing_nodes": validation.get(
                    "missing_nodes"
                ),
                "repeated_nodes": validation.get(
                    "repeated_nodes"
                ),
                "unexpected_nodes": validation.get(
                    "unexpected_nodes"
                ),
                "api_call_wall_seconds": call.get(
                    "api_call_wall_seconds"
                ),
                "total_token_count": usage.get(
                    "total_token_count"
                ),
                "cost": usage.get("cost"),
                "finish_reason": call.get("finish_reason"),
                "error": (
                    {
                        "type": errors[-1].get("type"),
                        "phase": errors[-1].get("phase"),
                        "message": errors[-1].get("message"),
                    }
                    if errors
                    else None
                ),
            }
        )
    valid_ranking = sorted(
        [
            {
                "alias": row["alias"],
                "model": row["model"],
                "distance": row["distance"],
                "gap_to_reference_percent": row[
                    "gap_to_reference_percent"
                ],
            }
            for row in rows
            if row["is_valid"] is True
            and row["distance"] is not None
        ],
        key=lambda row: row["distance"],
    )
    best_valid_models = (
        [
            row
            for row in valid_ranking
            if math.isclose(
                float(row["distance"]),
                float(valid_ranking[0]["distance"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]
        if valid_ranking
        else []
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "experiment": (
            "dynamic_openrouter_zero_shot_model_comparison"
        ),
        "generated_at_utc": utc_now_iso(),
        "run_id": manifest["run_id"],
        "problem": {
            "name": manifest["problem"]["name"],
            "dimension": manifest["problem"]["dimension"],
            "depot_id": manifest["problem"]["depot_id"],
            "fingerprint_sha256": fingerprint,
            "reference": manifest["problem"].get("reference"),
        },
        "common_configuration": {
            "method": "zero_shot",
            "temperature": 0.0,
            "reasoning_effort": "none",
            "coordinates_sent_to_model": False,
            "distance_matrix_sent_to_model": False,
        },
        "counts": {
            "registered_model_count": len(rows),
            "completed_result_count": sum(
                row["status"] != "missing"
                for row in rows
            ),
            "valid_route_count": sum(
                row["is_valid"] is True
                for row in rows
            ),
            "failed_result_count": sum(
                row["status"] == "failed"
                for row in rows
            ),
        },
        "models": rows,
        "ranking_by_valid_distance": valid_ranking,
        "best_valid_model": (
            valid_ranking[0]
            if valid_ranking
            else None
        ),
        "best_valid_models_at_same_distance": (
            best_valid_models
        ),
        "best_distance_is_tied": len(
            best_valid_models
        ) > 1,
    }


def _value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_comparison(comparison: dict[str, Any]) -> None:
    print("\nOpenRouter zero-shot model karşılaştırması")
    print(
        "Model                         Durum     "
        "Geçerli  Mesafe      Gap (%)    API (sn)"
    )
    print("-" * 86)
    for row in comparison["models"]:
        print(
            f"{row['alias']:<29} "
            f"{row['status']:<9} "
            f"{_value(row['is_valid']):<8} "
            f"{_value(row['distance']):<11} "
            f"{_value(row['gap_to_reference_percent']):<10} "
            f"{_value(row['api_call_wall_seconds'])}"
        )
    best_models = comparison[
        "best_valid_models_at_same_distance"
    ]
    if not best_models:
        print("\nHenüz geçerli OpenRouter rotası yok.")
    elif len(best_models) > 1:
        aliases = ", ".join(
            item["alias"]
            for item in best_models
        )
        print(
            "\nEn iyi geçerli modeller eşit mesafede: "
            f"{aliases} / mesafe={best_models[0]['distance']} / "
            "gap="
            f"{_value(best_models[0]['gap_to_reference_percent'])}%"
        )
    else:
        best = best_models[0]
        print(
            "\nEn iyi geçerli model: "
            f"{best['alias']} / mesafe={best['distance']} / "
            "gap="
            f"{_value(best['gap_to_reference_percent'])}%"
        )


def _total_cost(calls: list[dict[str, Any]]) -> float | None:
    costs = [
        float(value)
        for call in calls
        if (value := (call.get("usage") or {}).get("cost"))
        is not None
    ]
    return sum(costs) if costs else None


def run_model(
    *,
    alias: str,
    run_dir: Path,
    manifest: dict[str, Any],
    problem: Any,
    points_image: Path,
    prompt: str,
    max_tokens: int,
    overwrite: bool,
) -> str:
    model = resolve_model_alias(alias)
    output = _model_dir(run_dir, alias)
    result_path = output / "zero_shot_results.json"
    if result_path.exists() and not overwrite:
        print(
            f"{alias}: mevcut sonuç atlandı. Yeniden çalıştırmak "
            "için --overwrite kullanın."
        )
        return "skipped"

    total_timer = start_timer()
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": "2.0",
        "experiment": "dynamic_openrouter_zero_shot_tsp",
        "run_id": manifest["run_id"],
        "method": "zero_shot",
        "problem": {
            "name": problem.name,
            "dimension": problem.dimension,
            "depot_id": problem.depot_id,
            "source_type": problem.source_type.value,
            "fingerprint_sha256": manifest["problem"][
                "fingerprint_sha256"
            ],
            "reference_type": (
                problem.reference.reference_type.value
                if problem.reference is not None
                else None
            ),
            "reference_is_proven_optimal": (
                problem.reference.is_proven_optimal
                if problem.reference is not None
                else None
            ),
        },
        "model": {
            "provider": "openrouter",
            "alias": alias,
            "requested_name": model,
            "temperature": 0.0,
            "reasoning_effort": "none",
            "max_tokens": max_tokens,
        },
        "model_input": {
            "image": _relative(points_image, run_dir),
            "coordinates_sent_to_model": False,
            "distance_matrix_sent_to_model": False,
            "prompt": prompt,
        },
        "errors": errors,
    }
    phase = "route_generation"
    try:
        request = request_route(
            points_image,
            prompt=prompt,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            reasoning_effort="none",
            phase=phase,
        )
        calls.append(request.api_call)
        result["raw_response"] = request.text
        result["output_format_compliant"] = (
            output_format_compliant(request.text)
        )
        result["model"]["response_name"] = request.api_call.get(
            "response_model"
        )
        result["model"]["routed_provider"] = (
            request.api_call.get("routed_provider")
        )

        phase = "response_parsing"
        parsing_timer = start_timer()
        route = parse_route(
            request.text,
            depot_id=problem.depot_id,
        )
        parsing_seconds = elapsed_seconds(parsing_timer)

        phase = "validation_and_metrics"
        evaluation_timer = start_timer()
        evaluation = evaluate_route(problem, route)
        evaluation_seconds = elapsed_seconds(evaluation_timer)

        phase = "route_rendering"
        rendering_timer = start_timer()
        route_image = output / "images" / "route.png"
        plot_route(
            problem,
            route,
            route_image,
            title=(
                f"OpenRouter {alias} — {problem.name} — "
                f"distance {evaluation['distance']}"
            ),
        )
        rendering_seconds = elapsed_seconds(rendering_timer)
        result.update(
            {
                **evaluation,
                "artifacts": {
                    "route_image": _relative(
                        route_image,
                        run_dir,
                    )
                },
                "api_calls": calls,
                "timing": {
                    "api_call_wall_seconds": request.api_call[
                        "api_call_wall_seconds"
                    ],
                    "response_parsing_seconds": parsing_seconds,
                    "validation_and_metrics_seconds": (
                        evaluation_seconds
                    ),
                    "route_rendering_seconds": (
                        rendering_seconds
                    ),
                    "total_wall_seconds_before_result_write": (
                        elapsed_seconds(total_timer)
                    ),
                },
            }
        )
    except Exception as exc:
        record = error_record(exc, phase=phase)
        openrouter_call = getattr(
            exc,
            "openrouter_call_record",
            None,
        )
        if isinstance(openrouter_call, dict):
            record["api_call"] = openrouter_call
            calls.append(openrouter_call)
        errors.append(record)
        result.update(
            {
                "api_calls": calls,
                "timing": {
                    "total_wall_seconds_before_result_write": (
                        elapsed_seconds(total_timer)
                    )
                },
            }
        )

    result["run_summary"] = summarize_api_calls(calls)
    result["run_summary"]["total_cost"] = _total_cost(calls)
    write_json(result_path, result)

    if errors:
        print(
            f"{alias}: başarısız — "
            f"{errors[-1]['type']}: {errors[-1]['message']}"
        )
        return "failed"
    print(
        f"{alias}: geçerli="
        f"{result['validation']['is_valid']}, "
        f"mesafe={result['distance']}, "
        "referans gap="
        + (
            "hesaplanamadı"
            if result["gap_to_reference_percent"] is None
            else (
                f"%{result['gap_to_reference_percent']:.4f}"
            )
        )
    )
    return "completed"


def main() -> None:
    args = parse_args()
    if args.max_tokens < 128:
        raise SystemExit("--max-tokens en az 128 olmalıdır.")
    if (
        not args.summary_only
        and not args.all_models
        and not args.model
    ):
        raise SystemExit(
            "En az bir --model, --all-models veya "
            "--summary-only seçilmelidir."
        )

    run_id = normalize_run_id(args.run_id)
    run_dir = Path(args.output_dir) / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            "Run manifesti bulunamadı. Önce aynı --run-id ile "
            "run_baseline.py çalıştırılmalıdır."
        )
    manifest, problem = load_run_problem(manifest_path)
    points_image = (
        run_dir / "baseline" / "images" / "points.png"
    )
    if not points_image.exists():
        raise SystemExit(
            f"Baseline problem görseli bulunamadı: {points_image}"
        )
    prompt = initializer_prompt(problem)

    if args.summary_only:
        comparison = build_comparison(
            run_dir=run_dir,
            manifest=manifest,
        )
        comparison_path = _comparison_path(run_dir)
        write_json(comparison_path, comparison)
        print_comparison(comparison)
        print(f"\nKarşılaştırma dosyası: {comparison_path}")
        return

    aliases = (
        list(OPENROUTER_MODELS)
        if args.all_models
        else list(dict.fromkeys(args.model or []))
    )
    if args.validate_only:
        print("OpenRouter zero-shot çevrimdışı doğrulaması başarılı.")
        print(f"Run ID: {run_id}")
        print(
            f"Problem: {problem.name} "
            f"({problem.dimension} düğüm)"
        )
        print(
            "Problem fingerprint: "
            f"{manifest['problem']['fingerprint_sha256']}"
        )
        for alias in aliases:
            print(f"- {alias}: {resolve_model_alias(alias)}")
        print(f"Prompt karakter sayısı: {len(prompt)}")
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY ortam değişkeni tanımlı değil. "
            "Hiçbir sonuç dosyası yazılmadı."
        )

    print(
        f"OpenRouter zero-shot taraması: {problem.name} / "
        f"{len(aliases)} model"
    )
    for alias in aliases:
        run_model(
            alias=alias,
            run_dir=run_dir,
            manifest=manifest,
            problem=problem,
            points_image=points_image,
            prompt=prompt,
            max_tokens=args.max_tokens,
            overwrite=args.overwrite,
        )

    comparison = build_comparison(
        run_dir=run_dir,
        manifest=manifest,
    )
    comparison_path = _comparison_path(run_dir)
    write_json(comparison_path, comparison)
    print_comparison(comparison)
    print(f"\nKarşılaştırma dosyası: {comparison_path}")


if __name__ == "__main__":
    main()
