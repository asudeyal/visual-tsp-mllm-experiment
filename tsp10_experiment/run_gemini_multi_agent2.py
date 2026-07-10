"""Gemini Initializer + Critic ile makaledeki Multi-Agent 2 uyarlaması."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from src.llm_routes import (
    GEMINI_MODEL,
    parse_single_salesman_route,
    request_gemini_critic_route,
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
        help="İlk doğrulamada 1; daha sonra makaledeki gibi 10 kullanılır.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=13.0,
        help=(
            "Gemini ücretsiz katmanındaki dakika başına 5 istek sınırı için "
            "çağrılar arasında beklenecek süre."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Checkpoint'teki başarılı iterasyonlardan devam eder; daha önce "
            "tamamlanan API çağrılarını tekrarlamaz."
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


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations en az 1 olmalıdır.")
    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds negatif olamaz.")
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY ortam değişkeni tanımlı değil.")
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
    checkpoint_path = args.output_dir / "gemini_multi_agent2_checkpoint.json"
    result_path = args.output_dir / "gemini_multi_agent2_results.json"
    current_image = args.output_dir / "gemini_zero_shot_route.png"
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
        resume_path = checkpoint_path if checkpoint_path.exists() else result_path
        if not resume_path.exists():
            raise SystemExit(
                "--resume istendi ancak checkpoint veya sonuç dosyası bulunamadı."
            )

        previous = json.loads(resume_path.read_text(encoding="utf-8"))
        if previous.get("model") != args.model:
            raise SystemExit(
                "Checkpoint modeli ile seçilen model farklı; güvenli biçimde "
                "devam edilemedi."
            )

        iterations = list(previous.get("critic_iterations", []))
        errors = []
        final_evaluation = previous.get("final_solution", initial_evaluation)
        best_valid = previous.get("best_valid_solution", best_valid)
        start_iteration = len(iterations) + 1

        if len(iterations) >= args.iterations:
            print(
                f"Checkpoint zaten {len(iterations)} iterasyon içeriyor; "
                "yeni API çağrısı yapılmadı."
            )
            return

        current_image = (
            args.output_dir / f"gemini_ma2_iteration_{len(iterations):02d}.png"
        )
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
        if iteration > start_iteration and args.delay_seconds > 0:
            print(
                f"\nKota sınırını aşmamak için {args.delay_seconds:g} saniye "
                "bekleniyor..."
            )
            time.sleep(args.delay_seconds)

        try:
            raw_response = request_gemini_critic_route(
                current_image,
                model=args.model,
                temperature=0.7,
            )
        except Exception as exc:
            error_record = {
                "iteration": iteration,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error_record)
            print(f"\nCritic iterasyon {iteration} tamamlanamadı.")
            print(f"Hata türü: {error_record['type']}")
            print(f"Hata: {error_record['message']}")
            print("Başarılı iterasyonlar sonuç dosyasına kaydedilecek.")
            break

        route = parse_single_salesman_route(raw_response)
        evaluation = evaluate_route(
            locations,
            route,
            or_tools_distance=or_tools_distance,
            exact_distance=exact_distance,
        )
        record = {
            "iteration": iteration,
            "temperature": 0.7,
            "input_image": str(current_image),
            "raw_response": raw_response,
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
            print("Rota görselleştirilemedi; iterasyonlar güvenli biçimde durduruldu.")
            break

        iteration_image = (
            args.output_dir / f"gemini_ma2_iteration_{iteration:02d}.png"
        )
        iteration_solution = TSPSolution(
            method=f"{args.model}_ma2_critic_{iteration}",
            route=route,
            distance=float(evaluation["distance"]),
            validation=validate_tsp_route(route, len(locations)),
        )
        plot_solution(locations, iteration_solution, iteration_image)
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
            "model": args.model,
            "requested_iterations": args.iterations,
            "completed_iterations": len(iterations),
            "initializer": {
                "source": str(args.zero_shot),
                **initial_evaluation,
            },
            "critic_iterations": iterations,
            "final_solution": final_evaluation,
            "best_valid_solution": best_valid,
            "errors": errors,
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    result = {
        "experiment": "gemini_visual_multi_agent_2_tsp",
        "model": args.model,
        "num_locations_including_depot": len(locations),
        "num_salesmen": 1,
        "requested_iterations": args.iterations,
        "completed_iterations": len(iterations),
        "delay_seconds_between_requests": args.delay_seconds,
        "initializer": {
            "source": str(args.zero_shot),
            **initial_evaluation,
        },
        "critic_iterations": iterations,
        "final_solution": final_evaluation,
        "best_valid_solution": best_valid,
        "errors": errors,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
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


if __name__ == "__main__":
    main()
