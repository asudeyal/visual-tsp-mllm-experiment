"""Gemini Initializer + 7 Critic aday + Scorer ile Multi-Agent 1 uyarlaması."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from src.llm_routes import (
    GEMINI_MODEL,
    parse_scorer_response,
    parse_single_salesman_route,
    request_gemini_critic_candidates,
    request_gemini_scorer,
)
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
        default=Path("output/baseline_results.json"),
    )
    parser.add_argument(
        "--zero-shot",
        type=Path,
        default=Path("output/gemini_zero_shot_results.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
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
        "--delay-seconds",
        type=float,
        default=13.0,
        help="Ardışık Gemini API istekleri arasındaki en az bekleme süresi.",
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


class RequestPacer:
    """Aynı süreç içindeki Gemini istekleri arasında minimum aralık uygular."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_request_at: float | None = None

    def before_request(self) -> None:
        if self.last_request_at is not None and self.delay_seconds > 0:
            elapsed = time.monotonic() - self.last_request_at
            remaining = self.delay_seconds - elapsed
            if remaining > 0:
                print(
                    f"\nKota sınırını aşmamak için {remaining:.1f} saniye "
                    "bekleniyor..."
                )
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


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
        "model": args.model,
        "num_locations_including_depot": num_locations,
        "num_salesmen": 1,
        "requested_iterations": args.iterations,
        "completed_iterations": len(iterations),
        "critic_candidates_per_iteration": args.candidate_count,
        "critic_temperature": 0.7,
        "scorer_temperature": 0.0,
        "delay_seconds_between_requests": args.delay_seconds,
        "initializer": initializer,
        "iterations": iterations,
        "pending_iteration": pending_iteration,
        "final_solution": final_solution,
        "best_valid_solution": best_valid_solution,
        "best_critic_candidate_oracle": best_candidate_oracle,
        "errors": errors,
    }


def error_record(iteration: int, phase: str, exc: Exception) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
    }


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
    pacer: RequestPacer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hazır critic adaylarını scorer'a verir ve seçilen çözümü döndürür."""

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
            stored_scores, stored_best = parse_scorer_response(
                stored_response,
                expected_image_ids=image_ids,
            )
        except Exception:
            continue
        raw_scorer_response = str(stored_response)
        scores = stored_scores
        best_candidate_id = stored_best
        attempt["reparsed_successfully"] = True
        print("Checkpoint'teki ham scorer cevabı yeniden ayrıştırıldı; API çağrısı yapılmadı.")
        break

    if scores is None or best_candidate_id is None:
        pacer.before_request()
        raw_scorer_response = request_gemini_scorer(
            image_paths,
            image_ids=image_ids,
            model=args.model,
            temperature=0.0,
        )
        attempt_record: dict[str, Any] = {
            "raw_response": raw_scorer_response,
        }
        scorer_attempts.append(attempt_record)
        try:
            scores, best_candidate_id = parse_scorer_response(
                raw_scorer_response,
                expected_image_ids=image_ids,
            )
        except Exception as exc:
            attempt_record["parse_error_type"] = type(exc).__name__
            attempt_record["parse_error"] = str(exc)
            raise

    selected_candidate = next(
        candidate
        for candidate in candidates
        if int(candidate["candidate_id"]) == best_candidate_id
    )

    iteration = int(pending["iteration"])
    selected_image = (
        args.output_dir / f"gemini_ma1_iteration_{iteration:02d}_selected.png"
    )
    plot_evaluation(
        locations,
        selected_candidate,
        method=f"{args.model}_ma1_selected_{iteration}",
        output_path=selected_image,
    )
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

    completed = {
        "iteration": iteration,
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
        },
        "selected_solution": selected_solution,
    }
    return completed, selected_solution


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations en az 1 olmalıdır.")
    if not 1 <= args.candidate_count <= 7:
        raise SystemExit("--candidate-count 1 ile 7 arasında olmalıdır.")
    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds negatif olamaz.")
    if not args.baseline.exists() or not args.zero_shot.exists():
        raise SystemExit(
            "Baseline veya Gemini zero-shot sonucu bulunamadı. Önce "
            "run_baseline.py ve run_gemini_zero_shot.py çalıştırılmalıdır."
        )

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    zero_shot = json.loads(args.zero_shot.read_text(encoding="utf-8"))
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gemini_multi_agent1_checkpoint.json"
    result_path = args.output_dir / "gemini_multi_agent1_results.json"
    current_image = args.output_dir / "gemini_zero_shot_route.png"
    if not current_image.exists():
        plot_evaluation(
            locations,
            initial_evaluation,
            method=f"{args.model}_zero_shot",
            output_path=current_image,
        )

    initializer = {
        "source": str(args.zero_shot),
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
        resume_path = checkpoint_path if checkpoint_path.exists() else result_path
        if not resume_path.exists():
            raise SystemExit(
                "--resume istendi ancak Multi-Agent 1 checkpoint veya sonuç "
                "dosyası bulunamadı."
            )
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

    pacer = RequestPacer(args.delay_seconds)

    def save_checkpoint() -> None:
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
        write_json(checkpoint_path, state)

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
                pacer=pacer,
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
            pacer.before_request()
            raw_candidates = request_gemini_critic_candidates(
                current_image,
                candidate_count=args.candidate_count,
                model=args.model,
                temperature=0.7,
            )
            if len(raw_candidates) != args.candidate_count:
                print(
                    "UYARI: Gemini istenen aday sayısından farklı sayıda "
                    f"cevap döndürdü; istenen={args.candidate_count}, "
                    f"alınan={len(raw_candidates)}. Mevcut adaylarla scorer "
                    "aşamasına devam ediliyor ve gerçek sayı JSON'a yazılıyor."
                )

            candidate_records: list[dict[str, Any]] = []
            for candidate_id, raw_response in enumerate(raw_candidates, start=1):
                try:
                    route = parse_single_salesman_route(raw_response)
                except Exception as exc:
                    raise ValueError(
                        f"Critic adayı {candidate_id} ayrıştırılamadı. "
                        f"Ham cevap={raw_response!r}"
                    ) from exc
                evaluation = evaluate_route(
                    locations,
                    route,
                    or_tools_distance=or_tools_distance,
                    exact_distance=exact_distance,
                )
                if not evaluation["legal_node_ids"] or len(route) < 2:
                    raise ValueError(
                        f"Critic adayı {candidate_id} çizilemeyen düğüm içeriyor."
                    )
                candidate_image = args.output_dir / (
                    f"gemini_ma1_iteration_{iteration:02d}_"
                    f"candidate_{candidate_id:02d}.png"
                )
                plot_evaluation(
                    locations,
                    evaluation,
                    method=(
                        f"{args.model}_ma1_critic_{iteration}_"
                        f"candidate_{candidate_id}"
                    ),
                    output_path=candidate_image,
                )
                candidate_record = {
                    "candidate_id": candidate_id,
                    "raw_response": raw_response,
                    "image": str(candidate_image),
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
                    "candidates": candidate_records,
                },
            }
            save_checkpoint()
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
                pacer=pacer,
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
    write_json(result_path, result)

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


if __name__ == "__main__":
    main()
