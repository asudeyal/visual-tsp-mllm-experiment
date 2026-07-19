"""Gemini Initializer + 7 Critic aday + Scorer ile Multi-Agent 1 uyarlaması."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.experiment_metrics import (
    elapsed_seconds,
    error_record as build_error_record,
    start_timer,
    summarize_api_calls,
    utc_now_iso,
)
from src.llm_routes import (
    GEMINI_MODEL,
    parse_scorer_response,
    parse_single_salesman_route,
    request_gemini_critic_candidates_detailed,
    request_gemini_scorer_detailed,
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
            "output/runs/<run-id>/multi_agent1 klasörüne yazılır."
        ),
    )
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="İlk API doğrulamasında 1; final deneyinde makaledeki gibi 10.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=7,
        help=(
            "Her critic çağrısında üretilecek self-ensemble adayı. Makaledeki "
            "değer 7'dir; küçük smoke test için azaltılabilir."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Checkpoint'ten devam eder. Critic tamamlanıp scorer yarıda "
            "kaldıysa yedi critic adayını yeniden üretmez."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Girdileri ve deney planını API çağrısı yapmadan doğrular.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Mevcut sonuç JSON'undan kısa özet üretir; API anahtarı istemez "
            "ve Gemini çağrısı yapmaz."
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
    """Bir aday rotayı yapısal geçerlilik, mesafe ve gap ile değerlendirir."""

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


def better_valid_solution(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    """Aday geçerli ve daha kısaysa en iyi çözüm kaydını günceller."""

    if not candidate["validation"]["is_valid"] or candidate["distance"] is None:
        return current
    if current is None or float(candidate["distance"]) < float(current["distance"]):
        return candidate
    return current


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_state(
    *,
    experiment: str,
    args: argparse.Namespace,
    num_locations: int,
    initializer: dict[str, Any],
    iterations: list[dict[str, Any]],
    pending_iteration: dict[str, Any] | None,
    final_solution: dict[str, Any],
    best_valid_solution: dict[str, Any] | None,
    best_candidate_oracle: dict[str, Any] | None,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "run_id": args.run_id,
        "model": args.model,
        "num_locations_including_depot": num_locations,
        "num_salesmen": 1,
        "requested_iterations": args.iterations,
        "completed_iterations": len(iterations),
        "critic_candidates_per_iteration": args.candidate_count,
        "critic_temperature": 0.7,
        "scorer_temperature": 0.0,
        "artificial_delay_enabled": False,
        "initializer": initializer,
        "iterations": iterations,
        "pending_iteration": pending_iteration,
        "final_solution": final_solution,
        "best_valid_solution": best_valid_solution,
        "best_critic_candidate_oracle": best_candidate_oracle,
        "errors": errors,
    }


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _known_token_sum(api_calls: list[dict[str, Any]]) -> int | None:
    values = [
        call.get("usage", {}).get("total_token_count") for call in api_calls
    ]
    known = [int(value) for value in values if value is not None]
    return sum(known) if known else None


def build_multi_agent1_summary(
    result: dict[str, Any],
    *,
    source_results: Path,
) -> dict[str, Any]:
    """Ayrıntılı Multi-Agent 1 sonucundan insan okunur kısa özet üretir."""

    iterations = list(result.get("iterations", []))
    all_candidates = [
        candidate
        for iteration in iterations
        for candidate in iteration.get("critic", {}).get("candidates", [])
    ]
    pending = result.get("pending_iteration")
    if pending is not None:
        all_candidates.extend(pending.get("critic", {}).get("candidates", []))

    valid_candidates = [
        candidate
        for candidate in all_candidates
        if candidate.get("validation", {}).get("is_valid")
    ]
    optimal_candidates = [
        candidate
        for candidate in valid_candidates
        if candidate.get("gap_to_exact_percent") is not None
        and abs(float(candidate["gap_to_exact_percent"])) < 1e-9
    ]
    selected_solutions = [
        iteration.get("selected_solution", {}) for iteration in iterations
    ]
    optimal_selections = [
        selected
        for selected in selected_solutions
        if selected.get("validation", {}).get("is_valid")
        and selected.get("gap_to_exact_percent") is not None
        and abs(float(selected["gap_to_exact_percent"])) < 1e-9
    ]

    iteration_summaries: list[dict[str, Any]] = []
    for iteration in iterations:
        candidates = list(iteration.get("critic", {}).get("candidates", []))
        valid = [
            candidate
            for candidate in candidates
            if candidate.get("validation", {}).get("is_valid")
        ]
        optimal = [
            candidate
            for candidate in valid
            if candidate.get("gap_to_exact_percent") is not None
            and abs(float(candidate["gap_to_exact_percent"])) < 1e-9
        ]
        nonoptimal_ids = [
            int(candidate["candidate_id"])
            for candidate in valid
            if candidate.get("gap_to_exact_percent") is not None
            and abs(float(candidate["gap_to_exact_percent"])) >= 1e-9
        ]
        critic = iteration.get("critic", {})
        critic_call = critic.get("api_call", {})
        scorer = iteration.get("scorer", {})
        scorer_calls = [
            call for call in scorer.get("api_calls", []) if isinstance(call, dict)
        ]
        selected = iteration.get("selected_solution", {})
        iteration_summaries.append(
            {
                "iteration": int(iteration["iteration"]),
                "critic": {
                    "candidate_count": len(candidates),
                    "valid_candidate_count": len(valid),
                    "optimal_candidate_count": len(optimal),
                    "nonoptimal_candidate_ids": nonoptimal_ids,
                    "api_call_wall_seconds": critic_call.get(
                        "api_call_wall_seconds"
                    ),
                    "request_total_wall_seconds": critic_call.get(
                        "request_total_wall_seconds"
                    ),
                    "total_token_count": critic_call.get("usage", {}).get(
                        "total_token_count"
                    ),
                },
                "scorer": {
                    "selected_candidate_id": scorer.get("best_candidate_id"),
                    "selected_is_oracle_best": scorer.get(
                        "selected_is_oracle_best_after_evaluation"
                    ),
                    "selection_regret_percent": scorer.get(
                        "selection_regret_percent_after_evaluation"
                    ),
                    "api_call_wall_seconds": sum(
                        float(call.get("api_call_wall_seconds", 0.0))
                        for call in scorer_calls
                    ),
                    "total_token_count": _known_token_sum(scorer_calls),
                },
                "selected_solution": {
                    "route": selected.get("route"),
                    "distance": selected.get("distance"),
                    "gap_to_exact_percent": selected.get(
                        "gap_to_exact_percent"
                    ),
                },
                "logical_iteration_measured_seconds": iteration.get(
                    "timing", {}
                ).get("logical_iteration_measured_seconds"),
            }
        )

    error_summaries: list[dict[str, Any]] = []
    for error in result.get("errors", []):
        api_call = error.get("api_call", {})
        status_code = error.get("status_code")
        if status_code is None:
            message_prefix = str(error.get("message", "")).split(maxsplit=1)
            if message_prefix and message_prefix[0].isdigit():
                status_code = int(message_prefix[0])
        error_summaries.append(
            {
                "iteration": error.get("iteration"),
                "phase": error.get("phase"),
                "type": error.get("type"),
                "status_code": status_code,
                "api_call_wall_seconds": api_call.get("api_call_wall_seconds"),
            }
        )

    requested = int(result.get("requested_iterations", 0))
    completed = int(result.get("completed_iterations", len(iterations)))
    return {
        "experiment": result.get("experiment"),
        "summary_type": "compact",
        "source_results": str(source_results),
        "run_id": result.get("run_id"),
        "model": result.get("model"),
        "status": (
            "completed"
            if completed >= requested and pending is None
            else "partial"
        ),
        "requested_iterations": requested,
        "completed_iterations": completed,
        "pending_iteration": (
            {
                "iteration": pending.get("iteration"),
                "phase": "scorer",
                "critic_candidate_count": len(
                    pending.get("critic", {}).get("candidates", [])
                ),
            }
            if pending is not None
            else None
        ),
        "quality_summary": {
            "total_critic_candidates": len(all_candidates),
            "valid_critic_candidates": len(valid_candidates),
            "valid_candidate_rate_percent": _percent(
                len(valid_candidates), len(all_candidates)
            ),
            "optimal_critic_candidates": len(optimal_candidates),
            "optimal_candidate_rate_percent": _percent(
                len(optimal_candidates), len(all_candidates)
            ),
            "scorer_selections": len(selected_solutions),
            "optimal_scorer_selections": len(optimal_selections),
            "optimal_scorer_selection_rate_percent": _percent(
                len(optimal_selections), len(selected_solutions)
            ),
        },
        "api_summary": result.get("run_summary", {}),
        "iterations": iteration_summaries,
        "errors": error_summaries,
    }


def error_record(iteration: int, phase: str, exc: Exception) -> dict[str, Any]:
    return build_error_record(exc, iteration=iteration, phase=phase)


def validate_resume_compatibility(
    previous: dict[str, Any], args: argparse.Namespace
) -> None:
    if previous.get("model") != args.model:
        raise SystemExit(
            "Checkpoint modeli ile seçilen model farklı; güvenli biçimde "
            "devam edilemedi."
        )
    previous_count = previous.get("critic_candidates_per_iteration")
    if previous_count != args.candidate_count:
        raise SystemExit(
            "Checkpoint candidate-count değeri ile komut satırı farklı; "
            "güvenli biçimde devam edilemedi."
        )


def plot_evaluation(
    locations: list[tuple[float, float]],
    evaluation: dict[str, Any],
    *,
    method: str,
    output_path: Path,
) -> None:
    if not evaluation["legal_node_ids"] or len(evaluation["route"]) < 2:
        raise ValueError("Yasal olmayan düğüm kimlikleri içeren rota çizilemez.")
    solution = TSPSolution(
        method=method,
        route=[int(node) for node in evaluation["route"]],
        distance=float(evaluation["distance"]),
        validation=validate_tsp_route(evaluation["route"], len(locations)),
    )
    plot_solution(locations, solution, output_path)


def finalize_pending_iteration(
    *,
    pending: dict[str, Any],
    locations: list[tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hazır critic adaylarını scorer'a verir ve seçilen çözümü döndürür."""

    scorer_phase_timer = start_timer()
    scorer_parse_seconds = 0.0
    candidates = list(pending["critic"]["candidates"])
    image_ids = [int(candidate["candidate_id"]) for candidate in candidates]
    image_paths = [Path(candidate["image"]) for candidate in candidates]
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(
                f"Checkpoint'teki critic aday görseli bulunamadı: {image_path}"
            )

    scorer_attempts = pending.setdefault("scorer_attempts", [])
    raw_scorer_response: str | None = None
    scores: dict[int, float] | None = None
    best_candidate_id: int | None = None

    # Yeni parser eski bir ham cevabı anlayabiliyorsa aynı API çağrısını tekrar
    # harcamadan checkpoint'teki cevabı yeniden kullanır.
    for attempt in reversed(scorer_attempts):
        stored_response = attempt.get("raw_response")
        if not stored_response:
            continue
        try:
            parse_timer = start_timer()
            stored_scores, stored_best = parse_scorer_response(
                stored_response,
                expected_image_ids=image_ids,
            )
            scorer_parse_seconds += elapsed_seconds(parse_timer)
        except Exception:
            scorer_parse_seconds += elapsed_seconds(parse_timer)
            continue
        raw_scorer_response = str(stored_response)
        scores = stored_scores
        best_candidate_id = stored_best
        attempt["reparsed_successfully"] = True
        print("Checkpoint'teki ham scorer cevabı yeniden ayrıştırıldı; API çağrısı yapılmadı.")
        break

    if scores is None or best_candidate_id is None:
        gemini_result = request_gemini_scorer_detailed(
            image_paths,
            image_ids=image_ids,
            model=args.model,
            temperature=0.0,
        )
        raw_scorer_response = gemini_result.text
        attempt_record: dict[str, Any] = {
            "raw_response": raw_scorer_response,
            "api_call": gemini_result.api_call,
        }
        scorer_attempts.append(attempt_record)
        try:
            parse_timer = start_timer()
            scores, best_candidate_id = parse_scorer_response(
                raw_scorer_response,
                expected_image_ids=image_ids,
            )
            parse_duration = elapsed_seconds(parse_timer)
            scorer_parse_seconds += parse_duration
            attempt_record["response_parsing_seconds"] = parse_duration
        except Exception as exc:
            parse_duration = elapsed_seconds(parse_timer)
            scorer_parse_seconds += parse_duration
            attempt_record["response_parsing_seconds"] = parse_duration
            attempt_record["parse_error_type"] = type(exc).__name__
            attempt_record["parse_error"] = str(exc)
            raise

    selected_candidate = next(
        candidate
        for candidate in candidates
        if int(candidate["candidate_id"]) == best_candidate_id
    )

    iteration = int(pending["iteration"])
    iteration_image_dir = args.output_dir / f"iteration_{iteration:02d}"
    iteration_image_dir.mkdir(parents=True, exist_ok=True)
    selected_image = iteration_image_dir / "selected.png"
    render_timer = start_timer()
    plot_evaluation(
        locations,
        selected_candidate,
        method=f"{args.model}_ma1_selected_{iteration}",
        output_path=selected_image,
    )
    selected_route_rendering_seconds = elapsed_seconds(render_timer)

    evaluation_timer = start_timer()
    selected_solution = {
        "source": "scorer",
        "iteration": iteration,
        "candidate_id": best_candidate_id,
        "scorer_score": scores[best_candidate_id],
        "image": str(selected_image),
        **{
            key: selected_candidate[key]
            for key in (
                "route",
                "validation",
                "legal_node_ids",
                "distance",
                "gap_to_or_tools_percent",
                "gap_to_exact_percent",
            )
        },
    }
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["validation"]["is_valid"] and candidate["distance"] is not None
    ]
    best_candidate_distance = (
        min(float(candidate["distance"]) for candidate in valid_candidates)
        if valid_candidates
        else None
    )
    selected_is_valid = bool(selected_solution["validation"]["is_valid"])
    scorer_regret_percent = None
    selected_is_oracle_best = False
    if selected_is_valid and best_candidate_distance is not None:
        scorer_regret_percent = percentage_gap(
            float(selected_solution["distance"]),
            best_candidate_distance,
        )
        selected_is_oracle_best = abs(scorer_regret_percent) < 1e-9
    scorer_evaluation_seconds = elapsed_seconds(evaluation_timer)

    scorer_api_calls = [
        attempt["api_call"]
        for attempt in scorer_attempts
        if isinstance(attempt.get("api_call"), dict)
    ]
    scorer_api_seconds = sum(
        float(call.get("api_call_wall_seconds", 0.0)) for call in scorer_api_calls
    )

    completed = {
        "iteration": iteration,
        "iteration_type": "critic_candidates_and_visual_scorer",
        "critic": pending["critic"],
        "scorer": {
            "temperature": 0.0,
            "input_images": [str(path) for path in image_paths],
            "raw_response": raw_scorer_response,
            "attempts": scorer_attempts,
            "scores": {str(image_id): score for image_id, score in scores.items()},
            "best_candidate_id": best_candidate_id,
            "best_valid_candidate_distance_after_evaluation": best_candidate_distance,
            "selected_is_oracle_best_after_evaluation": selected_is_oracle_best,
            "selection_regret_percent_after_evaluation": scorer_regret_percent,
            "api_calls": scorer_api_calls,
            "timing": {
                "request_preparation_seconds": sum(
                    float(call.get("request_preparation_seconds", 0.0))
                    for call in scorer_api_calls
                ),
                "api_call_wall_seconds": scorer_api_seconds,
                "response_parsing_seconds": scorer_parse_seconds,
                "selected_route_rendering_seconds": (
                    selected_route_rendering_seconds
                ),
                "selection_evaluation_seconds": scorer_evaluation_seconds,
                "phase_wall_seconds_this_session": elapsed_seconds(
                    scorer_phase_timer
                ),
            },
        },
        "selected_solution": selected_solution,
    }
    critic_timing = pending["critic"].get("timing", {})
    critic_total = float(
        critic_timing.get(
            "phase_total_wall_seconds",
            critic_timing.get("phase_wall_seconds_before_checkpoint", 0.0),
        )
    )
    scorer_total = elapsed_seconds(scorer_phase_timer)
    completed["timing"] = {
        "critic_phase_measured_seconds": critic_total,
        "scorer_phase_measured_seconds": scorer_total,
        "logical_iteration_measured_seconds": critic_total + scorer_total,
    }
    return completed, selected_solution


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
    method_output_dir = paths.multi_agent1
    method_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = method_output_dir / "gemini_multi_agent1_checkpoint.json"
    result_path = method_output_dir / "gemini_multi_agent1_results.json"
    summary_path = method_output_dir / "gemini_multi_agent1_summary.json"

    if args.summary_only:
        if not result_path.exists():
            raise SystemExit(
                f"Özetlenecek Multi-Agent 1 sonuç dosyası bulunamadı: {result_path}"
            )
        existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        summary = build_multi_agent1_summary(
            existing_result,
            source_results=result_path,
        )
        write_json(summary_path, summary)
        print("Multi-Agent 1 kısa özeti oluşturuldu; API çağrısı yapılmadı.")
        print(f"Özet dosyası: {summary_path}")
        return

    image_dir = method_output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.iterations < 1:
        raise SystemExit("--iterations en az 1 olmalıdır.")
    if not 1 <= args.candidate_count <= 7:
        raise SystemExit("--candidate-count 1 ile 7 arasında olmalıdır.")
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
    args.output_dir = image_dir
    if not current_image.exists():
        plot_evaluation(
            locations,
            initial_evaluation,
            method=f"{args.model}_zero_shot",
            output_path=current_image,
        )

    initializer = {
        "source": str(zero_shot_path),
        "coordinates_sent_to_model": False,
        **initial_evaluation,
    }
    final_solution = {
        "source": "zero_shot",
        "iteration": 0,
        "image": str(current_image),
        **initial_evaluation,
    }
    best_valid_solution: dict[str, Any] | None = None
    if initial_evaluation["validation"]["is_valid"]:
        best_valid_solution = dict(final_solution)
    best_candidate_oracle: dict[str, Any] | None = None
    iterations: list[dict[str, Any]] = []
    pending_iteration: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = []

    if args.resume:
        resume_candidates = [
            path for path in (checkpoint_path, result_path) if path.exists()
        ]
        if not resume_candidates:
            raise SystemExit(
                "--resume istendi ancak Multi-Agent 1 checkpoint veya sonuç "
                "dosyası bulunamadı."
            )
        resume_path = max(resume_candidates, key=lambda path: path.stat().st_mtime_ns)
        previous = json.loads(resume_path.read_text(encoding="utf-8"))
        validate_resume_compatibility(previous, args)
        iterations = list(previous.get("iterations", []))
        pending_iteration = previous.get("pending_iteration")
        final_solution = previous.get("final_solution", final_solution)
        best_valid_solution = previous.get(
            "best_valid_solution", best_valid_solution
        )
        best_candidate_oracle = previous.get("best_critic_candidate_oracle")
        errors = list(previous.get("errors", []))
        if pending_iteration is None and iterations:
            current_image = Path(iterations[-1]["selected_solution"]["image"])
        print(
            f"Checkpoint yüklendi: {len(iterations)} tam iterasyon, "
            f"bekleyen scorer aşaması={'var' if pending_iteration else 'yok'}."
        )

    if args.validate_only:
        completed = len(iterations)
        remaining_iterations = max(args.iterations - completed, 0)
        estimated_requests = remaining_iterations * 2
        if pending_iteration is not None:
            estimated_requests -= 1
        print("\nMulti-Agent 1 çevrimdışı doğrulaması başarılı.")
        print(f"Model: {args.model}")
        print(f"Nokta sayısı: {len(locations)}")
        print(f"Hedef iterasyon: {args.iterations}")
        print(f"Critic adayı / iterasyon: {args.candidate_count}")
        print(f"Tahmini kalan Gemini isteği: {max(estimated_requests, 0)}")
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
        return

    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY ortam değişkeni tanımlı değil.")
    if len(iterations) >= args.iterations and pending_iteration is None:
        print(
            f"Checkpoint zaten {len(iterations)} tam iterasyon içeriyor; "
            "yeni API çağrısı yapılmadı."
        )
        return

    def save_checkpoint() -> float:
        state = build_state(
            experiment="gemini_visual_multi_agent_1_tsp_checkpoint",
            args=args,
            num_locations=len(locations),
            initializer=initializer,
            iterations=iterations,
            pending_iteration=pending_iteration,
            final_solution=final_solution,
            best_valid_solution=best_valid_solution,
            best_candidate_oracle=best_candidate_oracle,
            errors=errors,
        )
        checkpoint_timer = start_timer()
        write_json(checkpoint_path, state)
        return elapsed_seconds(checkpoint_timer)

    if pending_iteration is not None:
        pending_number = int(pending_iteration["iteration"])
        try:
            print(
                f"\nİterasyon {pending_number}: kayıtlı critic adayları "
                "yeniden kullanılacak; yalnız scorer çağrılacak."
            )
            completed, selected = finalize_pending_iteration(
                pending=pending_iteration,
                locations=locations,
                args=args,
            )
            iterations.append(completed)
            final_solution = selected
            best_valid_solution = better_valid_solution(
                best_valid_solution,
                selected,
            )
            current_image = Path(selected["image"])
            pending_iteration = None
            save_checkpoint()
        except Exception as exc:
            record = error_record(pending_number, "scorer", exc)
            errors.append(record)
            save_checkpoint()
            print(f"\nScorer iterasyon {pending_number} tamamlanamadı.")
            print(f"Hata türü: {record['type']}")
            print(f"Hata: {record['message']}")

    start_iteration = len(iterations) + 1
    for iteration in range(start_iteration, args.iterations + 1):
        if pending_iteration is not None:
            break
        try:
            print(
                f"\n=== Multi-Agent 1 iterasyon {iteration}: "
                f"{args.candidate_count} critic adayı ==="
            )
            critic_phase_timer = start_timer()
            gemini_result = request_gemini_critic_candidates_detailed(
                current_image,
                candidate_count=args.candidate_count,
                model=args.model,
                temperature=0.7,
            )
            raw_candidates = gemini_result.texts
            if len(raw_candidates) != args.candidate_count:
                print(
                    "UYARI: Gemini istenen aday sayısından farklı sayıda "
                    f"cevap döndürdü; istenen={args.candidate_count}, "
                    f"alınan={len(raw_candidates)}. Mevcut adaylarla scorer "
                    "aşamasına devam ediliyor ve gerçek sayı JSON'a yazılıyor."
                )

            candidate_records: list[dict[str, Any]] = []
            iteration_image_dir = args.output_dir / f"iteration_{iteration:02d}"
            iteration_image_dir.mkdir(parents=True, exist_ok=True)
            candidate_parsing_seconds = 0.0
            candidate_evaluation_seconds = 0.0
            candidate_rendering_seconds = 0.0
            for candidate_id, raw_response in enumerate(raw_candidates, start=1):
                try:
                    parse_timer = start_timer()
                    route = parse_single_salesman_route(raw_response)
                    parsing_seconds = elapsed_seconds(parse_timer)
                    candidate_parsing_seconds += parsing_seconds
                except Exception as exc:
                    candidate_parsing_seconds += elapsed_seconds(parse_timer)
                    raise ValueError(
                        f"Critic adayı {candidate_id} ayrıştırılamadı. "
                        f"Ham cevap={raw_response!r}"
                    ) from exc
                evaluation_timer = start_timer()
                evaluation = evaluate_route(
                    locations,
                    route,
                    or_tools_distance=or_tools_distance,
                    exact_distance=exact_distance,
                )
                evaluation_seconds = elapsed_seconds(evaluation_timer)
                candidate_evaluation_seconds += evaluation_seconds
                if not evaluation["legal_node_ids"] or len(route) < 2:
                    raise ValueError(
                        f"Critic adayı {candidate_id} çizilemeyen düğüm içeriyor."
                    )
                candidate_image = (
                    iteration_image_dir / f"candidate_{candidate_id:02d}.png"
                )
                render_timer = start_timer()
                plot_evaluation(
                    locations,
                    evaluation,
                    method=(
                        f"{args.model}_ma1_critic_{iteration}_"
                        f"candidate_{candidate_id}"
                    ),
                    output_path=candidate_image,
                )
                rendering_seconds = elapsed_seconds(render_timer)
                candidate_rendering_seconds += rendering_seconds
                candidate_record = {
                    "candidate_id": candidate_id,
                    "raw_response": raw_response,
                    "image": str(candidate_image),
                    "timing": {
                        "response_parsing_seconds": parsing_seconds,
                        "validation_and_metrics_seconds": evaluation_seconds,
                        "route_rendering_seconds": rendering_seconds,
                    },
                    **evaluation,
                }
                candidate_records.append(candidate_record)
                oracle_record = {
                    "source": "critic_candidate_oracle",
                    "iteration": iteration,
                    **candidate_record,
                }
                best_candidate_oracle = better_valid_solution(
                    best_candidate_oracle,
                    oracle_record,
                )
                print(
                    f"Aday {candidate_id}: geçerli="
                    f"{evaluation['validation']['is_valid']}, "
                    f"mesafe={evaluation['distance']}, "
                    f"gap={evaluation['gap_to_exact_percent']}"
                )

            pending_iteration = {
                "iteration": iteration,
                "critic": {
                    "temperature": 0.7,
                    "requested_candidate_count": args.candidate_count,
                    "returned_candidate_count": len(candidate_records),
                    "input_image": str(current_image),
                    "api_call": gemini_result.api_call,
                    "timing": {
                        "request_preparation_seconds": gemini_result.api_call.get(
                            "request_preparation_seconds", 0.0
                        ),
                        "api_call_wall_seconds": gemini_result.api_call[
                            "api_call_wall_seconds"
                        ],
                        "candidate_parsing_seconds": candidate_parsing_seconds,
                        "candidate_evaluation_seconds": (
                            candidate_evaluation_seconds
                        ),
                        "candidate_rendering_seconds": candidate_rendering_seconds,
                        "checkpoint_write_seconds": 0.0,
                        "phase_wall_seconds_before_checkpoint": elapsed_seconds(
                            critic_phase_timer
                        ),
                    },
                    "candidates": candidate_records,
                },
            }
            checkpoint_seconds = save_checkpoint()
            pending_iteration["critic"]["timing"][
                "checkpoint_write_seconds"
            ] = checkpoint_seconds
            pending_iteration["critic"]["timing"][
                "phase_total_wall_seconds"
            ] = elapsed_seconds(critic_phase_timer)
        except Exception as exc:
            record = error_record(iteration, "critic", exc)
            errors.append(record)
            save_checkpoint()
            print(f"\nCritic iterasyon {iteration} tamamlanamadı.")
            print(f"Hata türü: {record['type']}")
            print(f"Hata: {record['message']}")
            break

        try:
            completed, selected = finalize_pending_iteration(
                pending=pending_iteration,
                locations=locations,
                args=args,
            )
            iterations.append(completed)
            final_solution = selected
            best_valid_solution = better_valid_solution(
                best_valid_solution,
                selected,
            )
            current_image = Path(selected["image"])
            pending_iteration = None
            save_checkpoint()

            print(f"Scorer seçimi: aday {selected['candidate_id']}")
            print(f"Seçilen rota: {selected['route']}")
            print(f"Geçerli mi? {selected['validation']['is_valid']}")
            print(f"Mesafe: {selected['distance']}")
            print(f"Kesin optimum gap: {selected['gap_to_exact_percent']}")
        except Exception as exc:
            record = error_record(iteration, "scorer", exc)
            errors.append(record)
            save_checkpoint()
            print(f"\nScorer iterasyon {iteration} tamamlanamadı.")
            print(f"Hata türü: {record['type']}")
            print(f"Hata: {record['message']}")
            print(
                "Critic adayları checkpoint'e kaydedildi. --resume kullanıldığında "
                "critic çağrısı tekrarlanmayacak."
            )
            break

    result = build_state(
        experiment="gemini_visual_multi_agent_1_tsp",
        args=args,
        num_locations=len(locations),
        initializer=initializer,
        iterations=iterations,
        pending_iteration=pending_iteration,
        final_solution=final_solution,
        best_valid_solution=best_valid_solution,
        best_candidate_oracle=best_candidate_oracle,
        errors=errors,
    )
    api_calls: list[dict[str, Any]] = []
    for completed_iteration in iterations:
        critic_call = completed_iteration.get("critic", {}).get("api_call")
        if isinstance(critic_call, dict):
            api_calls.append(critic_call)
        api_calls.extend(
            call
            for call in completed_iteration.get("scorer", {}).get("api_calls", [])
            if isinstance(call, dict)
        )
    if pending_iteration is not None:
        critic_call = pending_iteration.get("critic", {}).get("api_call")
        if isinstance(critic_call, dict):
            api_calls.append(critic_call)
        api_calls.extend(
            attempt["api_call"]
            for attempt in pending_iteration.get("scorer_attempts", [])
            if isinstance(attempt.get("api_call"), dict)
        )
    api_calls.extend(
        failure["api_call"]
        for failure in errors
        if isinstance(failure.get("api_call"), dict)
    )
    result["run_summary"] = {
        "started_at_utc": run_started_at_utc,
        "finished_at_utc_before_result_write": utc_now_iso(),
        "session_wall_seconds_before_result_write": elapsed_seconds(run_timer),
        **summarize_api_calls(api_calls),
    }
    write_json(result_path, result)
    summary = build_multi_agent1_summary(result, source_results=result_path)
    write_json(summary_path, summary)

    print("\nGemini Multi-Agent 1 deneyi durumu kaydedildi.")
    print(f"Tamamlanan tam iterasyon: {len(iterations)}")
    if pending_iteration is not None:
        print(
            f"Scorer bekleyen iterasyon: {pending_iteration['iteration']} "
            "(--resume ile devam edilebilir)"
        )
    print(f"Son seçilen çözüm mesafesi: {final_solution['distance']}")
    if best_valid_solution is not None:
        print(
            "En iyi sistem çözümü: "
            f"{best_valid_solution['source']} / "
            f"iterasyon {best_valid_solution['iteration']} / "
            f"mesafe {best_valid_solution['distance']}"
        )
    if best_candidate_oracle is not None:
        print(
            "En iyi critic adayı (oracle analiz): "
            f"iterasyon {best_candidate_oracle['iteration']} / "
            f"aday {best_candidate_oracle['candidate_id']} / "
            f"mesafe {best_candidate_oracle['distance']}"
        )
    print(f"Sonuç dosyası: {result_path}")
    print(f"Kısa özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
