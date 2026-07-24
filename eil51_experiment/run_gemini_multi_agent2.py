"""eil51 için initializer + iteratif critic kullanan Gemini Multi-Agent 2."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core import (
    evaluate_route,
    method_dir,
    normalize_run_id,
    parse_tsplib,
    plot_route,
    read_json,
    write_json,
)
from src.gemini import GEMINI_MODEL, critic_prompt, parse_route, request_route
from src.metrics import elapsed_seconds, error_record, start_timer, summarize_api_calls
from src.summaries import multi_agent2_summary


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=ROOT / "data/eil51.tsp")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--run-id", default="eil51_run_01")
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def _solution(source: str, iteration: int, evaluation: dict) -> dict:
    return {"source": source, "iteration": iteration, **evaluation}


def _best(initializer: dict, iterations: list[dict]) -> dict | None:
    choices = [initializer] + [
        _solution("critic", item["iteration"], {
            key: item[key]
            for key in (
                "route",
                "validation",
                "legal_node_ids",
                "distance",
                "gap_to_known_optimum_percent",
            )
        })
        for item in iterations
    ]
    valid = [
        item
        for item in choices
        if item.get("validation", {}).get("is_valid")
        and item.get("distance") is not None
    ]
    return min(valid, key=lambda item: item["distance"]) if valid else None


def _result(
    *,
    run_id: str,
    model: str,
    requested: int,
    initializer: dict,
    iterations: list[dict],
    errors: list[dict],
    invocation_seconds: float,
) -> dict:
    final = (
        {
            key: iterations[-1][key]
            for key in (
                "route",
                "validation",
                "legal_node_ids",
                "distance",
                "gap_to_known_optimum_percent",
            )
        }
        if iterations
        else initializer
    )
    calls = [item["api_call"] for item in iterations if item.get("api_call")]
    calls.extend(
        error["api_call"] for error in errors if isinstance(error.get("api_call"), dict)
    )
    return {
        "experiment": "gemini_visual_multi_agent_2_eil51",
        "run_id": run_id,
        "model": model,
        "num_locations_including_depot": 51,
        "num_salesmen": 1,
        "requested_iterations": requested,
        "completed_iterations": len(iterations),
        "artificial_delay_enabled": False,
        "initializer": initializer,
        "critic_iterations": iterations,
        "final_solution": final,
        "best_valid_solution": _best(initializer, iterations),
        "errors": errors,
        "run_summary": {
            **summarize_api_calls(calls),
            "current_invocation_wall_seconds_before_result_write": invocation_seconds,
        },
    }


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations en az 1 olmalıdır.")
    run_id = normalize_run_id(args.run_id)
    output = method_dir(args.output_dir, run_id, "multi_agent2")
    result_path = output / "gemini_multi_agent2_results.json"
    summary_path = output / "gemini_multi_agent2_summary.json"
    checkpoint_path = output / "gemini_multi_agent2_checkpoint.json"
    if args.summary_only:
        write_json(summary_path, multi_agent2_summary(read_json(result_path)))
        print(f"Özet dosyası: {summary_path}")
        return

    instance = parse_tsplib(args.instance)
    invocation_timer = start_timer()
    zero_path = method_dir(args.output_dir, run_id, "zero_shot") / "gemini_zero_shot_results.json"
    if not zero_path.exists():
        raise SystemExit("Önce aynı --run-id ile zero-shot deneyi çalıştırılmalıdır.")
    zero = read_json(zero_path)
    zero_route = [int(value) for value in zero.get("route", [])]
    zero_evaluation = evaluate_route(instance, zero_route)
    initializer = _solution("zero_shot", 0, zero_evaluation)
    initializer["source_file"] = str(zero_path)

    iterations: list[dict] = []
    errors: list[dict] = []
    current_route = zero_route
    current_image = Path(zero.get("route_image", ""))
    if args.resume and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        if checkpoint.get("run_id") != run_id or checkpoint.get("model") != args.model:
            raise SystemExit("Checkpoint run-id/model ile mevcut komut uyuşmuyor.")
        iterations = checkpoint.get("critic_iterations", [])
        errors = checkpoint.get("errors", [])
        current_route = [int(value) for value in checkpoint["current_route"]]
        current_image = Path(checkpoint["current_image"])
        print(
            f"Checkpoint yüklendi: {len(iterations)} iterasyon tamamlanmış. "
            f"İterasyon {len(iterations) + 1}'den devam ediliyor."
        )
    elif args.resume:
        print("Checkpoint bulunamadı; deney baştan başlatılıyor.")
    if not current_image.exists():
        raise SystemExit(f"Girdi rota görseli bulunamadı: {current_image}")

    for iteration_number in range(len(iterations) + 1, args.iterations + 1):
        iteration_timer = start_timer()
        print(f"\n--- Critic iterasyon {iteration_number} ---")
        try:
            request = request_route(
                current_image,
                prompt=critic_prompt(),
                model=args.model,
                temperature=0.7,
                phase="critic_route_revision",
            )
            parsing_timer = start_timer()
            route = parse_route(request.text)
            parsing_seconds = elapsed_seconds(parsing_timer)
            evaluation_timer = start_timer()
            evaluation = evaluate_route(instance, route)
            evaluation_seconds = elapsed_seconds(evaluation_timer)
            rendering_timer = start_timer()
            image_path = output / "images" / f"iteration_{iteration_number:02d}.png"
            plot_route(
                instance,
                route,
                image_path,
                title=f"eil51 Multi-Agent 2 — iteration {iteration_number}",
            )
            rendering_seconds = elapsed_seconds(rendering_timer)
            record = {
                "iteration": iteration_number,
                "iteration_type": "critic_route_revision",
                "temperature": 0.7,
                "input_image": str(current_image),
                "raw_response": request.text,
                "api_call": request.api_call,
                "timing": {
                    "api_call_wall_seconds": request.api_call["api_call_wall_seconds"],
                    "response_parsing_seconds": parsing_seconds,
                    "validation_and_metrics_seconds": evaluation_seconds,
                    "route_rendering_seconds": rendering_seconds,
                    "iteration_total_wall_seconds": elapsed_seconds(iteration_timer),
                },
                **evaluation,
                "route_image": str(image_path),
            }
            iterations.append(record)
            current_route = route
            current_image = image_path
            write_json(
                checkpoint_path,
                {
                    "run_id": run_id,
                    "model": args.model,
                    "initializer": initializer,
                    "critic_iterations": iterations,
                    "current_route": current_route,
                    "current_image": str(current_image),
                    "errors": errors,
                },
            )
            print(f"Geçerli mi? {evaluation['validation']['is_valid']}")
            print(f"Mesafe: {evaluation['distance']}")
            gap = evaluation["gap_to_known_optimum_percent"]
            print(
                f"Bilinen optimum gap: "
                f"{'hesaplanamadı' if gap is None else f'%{gap:.4f}'}"
            )
        except Exception as exc:
            errors.append(
                error_record(exc, phase="critic_route_revision", iteration=iteration_number)
            )
            print(f"Critic iterasyon {iteration_number} tamamlanamadı: {exc}")
            break

    result = _result(
        run_id=run_id,
        model=args.model,
        requested=args.iterations,
        initializer=initializer,
        iterations=iterations,
        errors=errors,
        invocation_seconds=elapsed_seconds(invocation_timer),
    )
    write_json(result_path, result)
    write_json(summary_path, multi_agent2_summary(result))
    print("\nGemini eil51 Multi-Agent 2 durumu kaydedildi.")
    print(f"Tamamlanan iterasyon: {len(iterations)}")
    print(f"Sonuç dosyası: {result_path}")
    print(f"Özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
