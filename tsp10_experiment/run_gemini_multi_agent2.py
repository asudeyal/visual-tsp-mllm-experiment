"""Gemini Initializer + Critic ile makaledeki Multi-Agent 2 uyarlaması."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.experiment_metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
    utc_now_iso,
)
from src.llm_routes import (
    GEMINI_MODEL,
    parse_single_salesman_route,
    request_gemini_critic_route_detailed,
)
from src.output_paths import build_experiment_paths
from src.tsp_core import (
    TSPSolution,
    percentage_gap,
    plot_solution,
    route_distance,
    validate_tsp_route,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Varsayılan: seçilen run-id içindeki baseline sonucu.",
    )
    parser.add_argument(
        "--zero-shot",
        type=Path,
        default=None,
        help="Varsayılan: seçilen run-id içindeki zero-shot sonucu.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--run-id",
        help=(
            "Aynı deney çalıştırmasını adlandırır. Verilirse sonuçlar "
            "output/runs/<run-id>/multi_agent2 klasörüne yazılır."
        ),
    )
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="İlk doğrulamada 1; daha sonra makaledeki gibi 10 kullanılır.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Checkpoint'teki başarılı iterasyonlardan devam eder; daha önce "
            "tamamlanan API çağrılarını tekrarlamaz."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Mevcut Multi-Agent 2 sonuç JSON'undan kısa özet üretir; Gemini "
            "API çağrısı yapmaz."
        ),
    )
    return parser.parse_args()


def evaluate_route(
    locations: list[tuple[float, float]],
    route: list[int],
    *,
    or_tools_distance: float,
    exact_distance: float,
) -> dict[str, Any]:
    """Bir model rotasını geçerlilik, mesafe ve gap bakımından değerlendirir."""

    validation = validate_tsp_route(route, num_locations=len(locations))
    legal_node_ids = all(0 <= node < len(locations) for node in route)
    distance = route_distance(locations, route) if legal_node_ids else None
    return {
        "route": route,
        "validation": validation.to_dict(),
        "legal_node_ids": legal_node_ids,
        "distance": distance,
        "gap_to_or_tools_percent": (
            percentage_gap(distance, or_tools_distance) if distance is not None else None
        ),
        "gap_to_exact_percent": (
            percentage_gap(distance, exact_distance) if distance is not None else None
        ),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _error_status_code(error: dict[str, Any]) -> int | None:
    status_code = error.get("status_code")
    if status_code is not None:
        return int(status_code)
    prefix = str(error.get("message", "")).split(maxsplit=1)
    return int(prefix[0]) if prefix and prefix[0].isdigit() else None


def build_multi_agent2_summary(
    result: dict[str, Any],
    *,
    source_results: Path,
) -> dict[str, Any]:
    """Ayrıntılı Multi-Agent 2 sonucundan iterasyon bazlı kısa özet üretir."""

    iterations = list(result.get("critic_iterations", []))
    valid_iterations = [
        item
        for item in iterations
        if item.get("validation", {}).get("is_valid")
    ]
    optimal_iterations = [
        item
        for item in valid_iterations
        if item.get("gap_to_exact_percent") is not None
        and abs(float(item["gap_to_exact_percent"])) < 1e-9
    ]
    requested = int(result.get("requested_iterations", 0))
    completed = int(result.get("completed_iterations", len(iterations)))
    errors = list(result.get("errors", []))
    if completed >= requested:
        status = "completed"
    elif completed == 0 and errors:
        status = "failed"
    else:
        status = "partial"

    iteration_summaries = []
    for item in iterations:
        api_call = item.get("api_call", {})
        iteration_summaries.append(
            {
                "iteration": item.get("iteration"),
                "iteration_type": item.get("iteration_type"),
                "route": item.get("route"),
                "is_valid": item.get("validation", {}).get("is_valid"),
                "distance": item.get("distance"),
                "gap_to_exact_percent": item.get("gap_to_exact_percent"),
                "api_call_wall_seconds": api_call.get(
                    "api_call_wall_seconds"
                ),
                "request_total_wall_seconds": api_call.get(
                    "request_total_wall_seconds"
                ),
                "total_token_count": api_call.get("usage", {}).get(
                    "total_token_count"
                ),
                "iteration_total_wall_seconds": item.get("timing", {}).get(
                    "iteration_total_wall_seconds"
                ),
            }
        )

    def compact_solution(solution: dict[str, Any] | None) -> dict[str, Any] | None:
        if solution is None:
            return None
        return {
            "source": solution.get("source"),
            "iteration": solution.get("iteration"),
            "route": solution.get("route"),
            "is_valid": solution.get("validation", {}).get("is_valid"),
            "distance": solution.get("distance"),
            "gap_to_exact_percent": solution.get("gap_to_exact_percent"),
        }

    return {
        "experiment": result.get("experiment"),
        "summary_type": "compact",
        "source_results": str(source_results),
        "status": status,
        "run_id": result.get("run_id"),
        "model": result.get("model"),
        "requested_iterations": requested,
        "completed_iterations": completed,
        "quality_summary": {
            "valid_iterations": len(valid_iterations),
            "valid_iteration_rate_percent": _percent(
                len(valid_iterations), len(iterations)
            ),
            "optimal_iterations": len(optimal_iterations),
            "optimal_iteration_rate_percent": _percent(
                len(optimal_iterations), len(iterations)
            ),
        },
        "initializer": compact_solution(result.get("initializer")),
        "final_solution": compact_solution(result.get("final_solution")),
        "best_valid_solution": compact_solution(
            result.get("best_valid_solution")
        ),
        "api_summary": result.get("run_summary", {}),
        "iterations": iteration_summaries,
        "errors": [
            {
                "iteration": error.get("iteration"),
                "phase": error.get("phase"),
                "type": error.get("type"),
                "status_code": _error_status_code(error),
                "api_call_wall_seconds": error.get("api_call", {}).get(
                    "api_call_wall_seconds"
                ),
            }
            for error in errors
        ],
    }


def main() -> None:
    args = parse_args()
    run_started_at_utc = utc_now_iso()
    run_timer = start_timer()
    try:
        paths = build_experiment_paths(args.output_dir, args.run_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    baseline_path = args.baseline or paths.baseline / "baseline_results.json"
    zero_shot_path = (
        args.zero_shot or paths.zero_shot / "gemini_zero_shot_results.json"
    )
    if args.baseline is None and args.run_id is None and not baseline_path.exists():
        legacy_baseline = args.output_dir / "baseline_results.json"
        if legacy_baseline.exists():
            baseline_path = legacy_baseline
    if args.zero_shot is None and args.run_id is None and not zero_shot_path.exists():
        legacy_zero_shot = args.output_dir / "gemini_zero_shot_results.json"
        if legacy_zero_shot.exists():
            zero_shot_path = legacy_zero_shot
    method_output_dir = paths.multi_agent2
    method_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = method_output_dir / "gemini_multi_agent2_checkpoint.json"
    result_path = method_output_dir / "gemini_multi_agent2_results.json"
    summary_path = method_output_dir / "gemini_multi_agent2_summary.json"

    if args.summary_only:
        if not result_path.exists():
            raise SystemExit(
                f"Özetlenecek Multi-Agent 2 sonucu bulunamadı: {result_path}"
            )
        existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        write_json(
            summary_path,
            build_multi_agent2_summary(
                existing_result,
                source_results=result_path,
            ),
        )
        print("Multi-Agent 2 kısa özeti oluşturuldu; API çağrısı yapılmadı.")
        print(f"Özet dosyası: {summary_path}")
        return

    image_dir = method_output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.iterations < 1:
        raise SystemExit("--iterations en az 1 olmalıdır.")
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY ortam değişkeni tanımlı değil.")
    if not baseline_path.exists() or not zero_shot_path.exists():
        raise SystemExit(
            "Baseline veya Gemini zero-shot sonucu bulunamadı. Önce "
            "run_baseline.py ve run_gemini_zero_shot.py çalıştırılmalıdır."
        )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    zero_shot = json.loads(zero_shot_path.read_text(encoding="utf-8"))
    locations = [tuple(point) for point in baseline["locations"]]
    or_tools_distance = float(baseline["solutions"]["or_tools"]["distance"])
    exact_distance = float(baseline["solutions"]["exact"]["distance"])

    initial_route = [int(node) for node in zero_shot["route"]]
    initial_evaluation = evaluate_route(
        locations,
        initial_route,
        or_tools_distance=or_tools_distance,
        exact_distance=exact_distance,
    )
    if not initial_evaluation["legal_node_ids"]:
        raise SystemExit("Zero-shot rotasında görselleştirilemeyen düğüm numarası var.")

    current_image = paths.zero_shot / "images" / "gemini_zero_shot_route.png"
    if not current_image.exists():
        old_method_image = paths.zero_shot / "gemini_zero_shot_route.png"
        legacy_flat_image = args.output_dir / "gemini_zero_shot_route.png"
        if old_method_image.exists():
            current_image = old_method_image
        elif args.run_id is None and legacy_flat_image.exists():
            current_image = legacy_flat_image
    if not current_image.exists():
        initial_solution = TSPSolution(
            method=f"{args.model}_zero_shot",
            route=initial_route,
            distance=float(initial_evaluation["distance"]),
            validation=validate_tsp_route(initial_route, len(locations)),
        )
        plot_solution(locations, initial_solution, current_image)

    best_valid: dict[str, Any] | None = None
    if initial_evaluation["validation"]["is_valid"]:
        best_valid = {
            "source": "zero_shot",
            "iteration": 0,
            **initial_evaluation,
        }

    iterations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    final_evaluation = initial_evaluation
    start_iteration = 1

    if args.resume:
        resume_candidates = [
            path for path in (checkpoint_path, result_path) if path.exists()
        ]
        if not resume_candidates:
            raise SystemExit(
                "--resume istendi ancak checkpoint veya sonuç dosyası bulunamadı."
            )
        resume_path = max(resume_candidates, key=lambda path: path.stat().st_mtime_ns)

        previous = json.loads(resume_path.read_text(encoding="utf-8"))
        if previous.get("model") != args.model:
            raise SystemExit(
                "Checkpoint modeli ile seçilen model farklı; güvenli biçimde "
                "devam edilemedi."
            )

        iterations = list(previous.get("critic_iterations", []))
        errors = list(previous.get("errors", []))
        final_evaluation = previous.get("final_solution", initial_evaluation)
        best_valid = previous.get("best_valid_solution", best_valid)
        start_iteration = len(iterations) + 1

        if len(iterations) >= args.iterations:
            summary_source = previous
            if result_path.exists():
                summary_source = json.loads(result_path.read_text(encoding="utf-8"))
            write_json(
                summary_path,
                build_multi_agent2_summary(
                    summary_source,
                    source_results=result_path,
                ),
            )
            print(
                f"Checkpoint zaten {len(iterations)} iterasyon içeriyor; "
                "yeni API çağrısı yapılmadı."
            )
            print(f"Kısa özet dosyası: {summary_path}")
            return

        current_image = image_dir / f"iteration_{len(iterations):02d}.png"
        if not current_image.exists():
            old_iteration_image = method_output_dir / (
                f"gemini_ma2_iteration_{len(iterations):02d}.png"
            )
            if old_iteration_image.exists():
                current_image = old_iteration_image
        if not current_image.exists():
            raise SystemExit(
                f"Devam görseli bulunamadı: {current_image}. "
                "Checkpoint'ten devam edilemedi."
            )
        print(
            f"Checkpoint yüklendi: {len(iterations)} iterasyon tamamlanmış. "
            f"İterasyon {start_iteration}'den devam ediliyor."
        )

    for iteration in range(start_iteration, args.iterations + 1):
        iteration_timer = start_timer()
        try:
            gemini_result = request_gemini_critic_route_detailed(
                current_image,
                model=args.model,
                temperature=0.7,
            )
        except Exception as exc:
            failure = error_record(exc, iteration=iteration, phase="critic")
            failure["iteration_wall_seconds"] = elapsed_seconds(iteration_timer)
            errors.append(failure)
            print(f"\nCritic iterasyon {iteration} tamamlanamadı.")
            print(f"Hata türü: {failure['type']}")
            print(f"Hata: {failure['message']}")
            print("Başarılı iterasyonlar sonuç dosyasına kaydedilecek.")
            break

        raw_response = gemini_result.text
        parse_timer = start_timer()
        route = parse_single_salesman_route(raw_response)
        response_parsing_seconds = elapsed_seconds(parse_timer)

        evaluation_timer = start_timer()
        evaluation = evaluate_route(
            locations,
            route,
            or_tools_distance=or_tools_distance,
            exact_distance=exact_distance,
        )
        validation_and_metrics_seconds = elapsed_seconds(evaluation_timer)
        record = {
            "iteration": iteration,
            "iteration_type": "critic_route_revision",
            "temperature": 0.7,
            "input_image": str(current_image),
            "raw_response": raw_response,
            "api_call": gemini_result.api_call,
            "timing": {
                "request_preparation_seconds": gemini_result.api_call.get(
                    "request_preparation_seconds", 0.0
                ),
                "api_call_wall_seconds": gemini_result.api_call[
                    "api_call_wall_seconds"
                ],
                "response_parsing_seconds": response_parsing_seconds,
                "validation_and_metrics_seconds": validation_and_metrics_seconds,
                "route_rendering_seconds": 0.0,
                "checkpoint_write_seconds": 0.0,
            },
            **evaluation,
        }
        iterations.append(record)
        final_evaluation = evaluation

        print(f"\n--- Critic iterasyon {iteration} ---")
        print(raw_response)
        print(f"Rota: {route}")
        print(f"Geçerli mi? {evaluation['validation']['is_valid']}")
        print(f"Mesafe: {evaluation['distance']}")
        print(f"Kesin optimum gap: {evaluation['gap_to_exact_percent']}")

        if not evaluation["legal_node_ids"] or len(route) < 2:
            record["timing"]["iteration_total_wall_seconds"] = elapsed_seconds(
                iteration_timer
            )
            print("Rota görselleştirilemedi; iterasyonlar güvenli biçimde durduruldu.")
            break

        iteration_image = image_dir / f"iteration_{iteration:02d}.png"
        render_timer = start_timer()
        iteration_solution = TSPSolution(
            method=f"{args.model}_ma2_critic_{iteration}",
            route=route,
            distance=float(evaluation["distance"]),
            validation=validate_tsp_route(route, len(locations)),
        )
        plot_solution(locations, iteration_solution, iteration_image)
        record["timing"]["route_rendering_seconds"] = elapsed_seconds(render_timer)
        current_image = iteration_image

        if evaluation["validation"]["is_valid"] and (
            best_valid is None
            or float(evaluation["distance"]) < float(best_valid["distance"])
        ):
            best_valid = {
                "source": "critic",
                "iteration": iteration,
                **evaluation,
            }

        # Her başarılı çağrıdan sonra ara kayıt alınır. Böylece kota veya ağ
        # hatasında o ana kadarki ücretli/limitli API çağrıları kaybolmaz.
        checkpoint = {
            "experiment": "gemini_visual_multi_agent_2_tsp_checkpoint",
            "run_id": args.run_id,
            "model": args.model,
            "artificial_delay_enabled": False,
            "requested_iterations": args.iterations,
            "completed_iterations": len(iterations),
            "initializer": {
                "source": str(zero_shot_path),
                **initial_evaluation,
            },
            "critic_iterations": iterations,
            "final_solution": final_evaluation,
            "best_valid_solution": best_valid,
            "errors": errors,
        }
        checkpoint_timer = start_timer()
        write_json(checkpoint_path, checkpoint)
        record["timing"]["checkpoint_write_seconds"] = elapsed_seconds(
            checkpoint_timer
        )
        record["timing"]["iteration_total_wall_seconds"] = elapsed_seconds(
            iteration_timer
        )

    result = {
        "experiment": "gemini_visual_multi_agent_2_tsp",
        "run_id": args.run_id,
        "model": args.model,
        "num_locations_including_depot": len(locations),
        "num_salesmen": 1,
        "requested_iterations": args.iterations,
        "completed_iterations": len(iterations),
        "artificial_delay_enabled": False,
        "initializer": {
            "source": str(zero_shot_path),
            **initial_evaluation,
        },
        "critic_iterations": iterations,
        "final_solution": final_evaluation,
        "best_valid_solution": best_valid,
        "errors": errors,
    }
    api_calls = [
        record["api_call"]
        for record in iterations
        if isinstance(record.get("api_call"), dict)
    ] + [
        failure["api_call"]
        for failure in errors
        if isinstance(failure.get("api_call"), dict)
    ]
    result["run_summary"] = {
        "started_at_utc": run_started_at_utc,
        "finished_at_utc_before_result_write": utc_now_iso(),
        "total_wall_seconds_before_result_write": elapsed_seconds(run_timer),
        **summarize_api_calls(api_calls),
    }
    write_json(result_path, result)
    write_json(
        summary_path,
        build_multi_agent2_summary(result, source_results=result_path),
    )

    print("\nGemini Multi-Agent 2 deneyi tamamlandı.")
    print(f"Tamamlanan iterasyon: {len(iterations)}")
    if errors:
        print(f"Tamamlanamayan iterasyon: {errors[0]['iteration']}")
    print(f"Son çözüm mesafesi: {final_evaluation['distance']}")
    if best_valid is not None:
        print(
            "En iyi geçerli çözüm: "
            f"{best_valid['source']} / iterasyon {best_valid['iteration']} / "
            f"mesafe {best_valid['distance']}"
        )
    print(f"Sonuç dosyası: {result_path}")
    print(f"Kısa özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
