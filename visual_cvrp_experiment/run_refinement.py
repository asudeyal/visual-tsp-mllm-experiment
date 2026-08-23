"""Görsel kodlamalar için geri bildirimli CVRP iyileştirme deneyi."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from run_experiment import (
    DEFAULT_OUTPUT_DIR,
    _gap_percent,
    _normalize_path_component,
    _relative,
    _write_json,
)
from src.exact_solver import solve_exact_cvrp

from src.providers import get_provider
from src.providers.gemini import DEFAULT_GEMINI_MODEL, GeminiClientError

from src.instances import (
    build_capacity_demo_10,
    build_cvrplib_problem,
)
from src.model_contract import (
    ModelResponseParseError,
    build_solver_prompt,
    parse_model_response,
)
from src.rendering import DemandEncoding, render_problem
from src.validation import evaluate_solution


DEFAULT_RUN_ID = "capacity_demo_10_refinement_01"
DEFAULT_HISTORICAL_RUN_ID = "capacity_demo_10_01"
DEFAULT_ENCODINGS = (
    DemandEncoding.BAR_LENGTH,
    DemandEncoding.COLOR_INTENSITY,
)
SUPPORTED_ENCODINGS = (
    DemandEncoding.BAR_LENGTH,
    DemandEncoding.COLOR_INTENSITY,
    DemandEncoding.SIZE,
)
OPTIMAL_GAP_TOLERANCE = 1e-9


def _progress(message: str) -> None:
    print(message, flush=True)


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
        "--historical-run-id",
        default=DEFAULT_HISTORICAL_RUN_ID,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        choices=[
            encoding.value
            for encoding in SUPPORTED_ENCODINGS
        ],
        default=[
            encoding.value
            for encoding in DEFAULT_ENCODINGS
        ],
    )
    parser.add_argument(
        "--max-refinement-iterations",
        type=int,
        default=3,
        help=(
            "İlk sıfırdan çağrıdan sonra yapılabilecek "
            "en fazla iyileştirme çağrısı. --resume ile "
            "var olan run'ın sınırı daha yüksek bir "
            "değere çıkarılabilir."
        ),
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
            "Görselleri, başlangıç promptlarını ve planı "
            "hazırlar; API çağrısı yapmaz."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Tamamlanmış iterasyonları atlayarak kesilen "
            "deneye devam eder."
        ),
    )
    parser.add_argument(
        "--extend-encodings",
        action="store_true",
        help=(
            "Var olan run'a yeni kodlamaları kontrollü "
            "olarak ekler. Yalnızca --resume ile "
            "kullanılabilir."
        ),
    )
    parser.add_argument(
        "--instance-file",
        type=Path,
        default=None,
        help=(
            "CVRPLIB .vrp dosyasının yolu. "
            "Verilmezse capacity_demo_10 kullanılır."
        ),
    )
    return parser.parse_args(argv)


def _normalize_encodings(
    encodings: Sequence[DemandEncoding | str],
) -> tuple[DemandEncoding, ...]:
    normalized = tuple(
        DemandEncoding(encoding)
        for encoding in encodings
    )
    if not normalized:
        raise ValueError(
            "En az bir talep kodlaması seçilmelidir."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Talep kodlamaları tekrar edemez."
        )
    unsupported = [
        encoding.value
        for encoding in normalized
        if encoding not in SUPPORTED_ENCODINGS
    ]
    if unsupported:
        raise ValueError(
            "Bu iyileştirme deneyinde desteklenmeyen "
            f"kodlamalar: {', '.join(unsupported)}"
        )
    return normalized


def _format_ids(value: Any) -> str:
    if not value:
        return "none"
    return ", ".join(str(item) for item in value)


def build_refinement_prompt(
    *,
    encoding: DemandEncoding | str,
    previous_result: dict[str, Any],
    iteration: int,
) -> str:
    """Önceki çözüm ve doğrulama ölçümlerinden yeni prompt üret."""

    if iteration < 2:
        raise ValueError(
            "İyileştirme promptu ikinci iterasyondan başlar."
        )

    parsed = previous_result.get("parsed_solution")
    validation = previous_result.get("validation")
    if (
        previous_result.get("status") != "completed"
        or not isinstance(parsed, dict)
        or not isinstance(validation, dict)
    ):
        raise ValueError(
            "İyileştirme için tamamlanmış ve doğrulanmış "
            "bir önceki sonuç gerekir."
        )

    route_lines = []
    capacity = int(
        previous_result["problem"]["vehicle_capacity"]
    )
    for route in validation.get("routes") or []:
        route_lines.append(
            "- Route "
            f"{route.get('route_index')}: load "
            f"{route.get('load')}/"
            f"{capacity}; "
            "capacity excess "
            f"{route.get('capacity_excess', 0)}; "
            "valid "
            f"{'yes' if route.get('valid') else 'no'}"
        )

    feedback_lines = [
        f"- Entire solution valid: {'yes' if validation.get('valid') else 'no'}",
        *route_lines,
        "- Missing customer IDs: "
        + _format_ids(
            validation.get("missing_customer_ids")
        ),
        "- Repeated customer IDs: "
        + _format_ids(
            validation.get("duplicated_customer_ids")
        ),
        "- Unknown node IDs: "
        + _format_ids(
            validation.get("unknown_node_ids")
        ),
        "- Total capacity excess: "
        + str(
            validation.get("total_capacity_excess", 0)
        ),
        "- Fleet limit exceeded: "
        + (
            "yes"
            if validation.get("fleet_limit_exceeded")
            else "no"
        ),
    ]
    if validation.get("valid") is True:
        feedback_lines.append(
            "- Total route distance: "
            f"{validation.get('total_distance')}"
        )

    previous_routes = json.dumps(
        parsed,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        build_solver_prompt(encoding=encoding)
        + "\n\n"
        + f"Refinement iteration {iteration}:\n"
        + "The following routes are your previous proposal. "
        + "Replace them with a better proposal; do not merely "
        + "explain or repair them in prose.\n"
        + previous_routes
        + "\n\nDeterministic validation feedback:\n"
        + "\n".join(feedback_lines)
        + "\n\n"
        + (
            "First restore feasibility, then reduce total "
            "distance."
            if validation.get("valid") is not True
            else (
                "Preserve every feasibility constraint and "
                "reduce total distance if possible."
            )
        )
        + " Do not output calculations. Return only the "
        + "replacement JSON object in the required routes format."
    )


def _is_exact_result(result: dict[str, Any]) -> bool:
    gap = result.get("optimality_gap_percent")
    validation = result.get("validation") or {}
    return (
        result.get("status") == "completed"
        and validation.get("valid") is True
        and isinstance(gap, (int, float))
        and abs(float(gap)) <= OPTIMAL_GAP_TOLERANCE
    )


def _iteration_number(path: Path) -> int:
    return int(path.parent.name.removeprefix("iteration_"))


def _load_iteration_results(
    method_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    for path in sorted(
        method_dir.glob(
            "iteration_*/iteration_results.json"
        )
    ):
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError(
                f"İterasyon JSON kökü nesne olmalıdır: {path}"
            )
        results.append(payload)
    return results


def _api_attempt_count(run_dir: Path) -> int:
    paths = [
        *run_dir.glob(
            "providers/*/*/*/iteration_*/"
            "iteration_results.json"
        ),
        *run_dir.glob(
            "providers/*/*/*/iteration_*/"
            "request_failure_*.json"
        ),
    ]
    count = 0
    for path in paths:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
        if payload.get("api_call_performed") is True:
            count += 1
    return count


def _validate_existing_manifest(
    *,
    manifest_path: Path,
    historical_run_id: str,
    model: str,
    encodings: Sequence[DemandEncoding],
    maximum_iterations: int,
    allow_encoding_extension: bool = False,
    allow_iteration_extension: bool = False,
) -> None:
    if not manifest_path.is_file():
        if (
            allow_encoding_extension
            or allow_iteration_extension
        ):
            raise ValueError(
                "Run genişletmesi için var olan bir "
                "refinement manifesti bulunamadı."
            )
        return
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    expected = {
        "historical_run_id": historical_run_id,
        "model": model.strip(),
    }
    mismatches = [
        key
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    requested_encodings = [
        encoding.value
        for encoding in encodings
    ]
    existing_encodings = payload.get("encodings")
    if allow_encoding_extension:
        encoding_matches = (
            isinstance(existing_encodings, list)
            and requested_encodings[
                : len(existing_encodings)
            ]
            == existing_encodings
        )
    else:
        encoding_matches = (
            existing_encodings == requested_encodings
        )
    if not encoding_matches:
        mismatches.append("encodings")

    existing_maximum = payload.get(
        "maximum_iterations_per_encoding"
    )
    if allow_iteration_extension:
        iteration_matches = (
            isinstance(existing_maximum, int)
            and maximum_iterations >= existing_maximum
        )
    else:
        iteration_matches = (
            existing_maximum == maximum_iterations
        )
    if not iteration_matches:
        mismatches.append(
            "maximum_iterations_per_encoding"
        )

    if mismatches:
        raise ValueError(
            "Var olan run yapılandırması yeni komutla "
            "eşleşmiyor: "
            + ", ".join(mismatches)
            + ". Yeni bir run ID kullanın."
        )


def _archive_failed_attempt(
    result_path: Path,
) -> Path | None:
    if not result_path.is_file():
        return None
    payload = json.loads(
        result_path.read_text(encoding="utf-8")
    )
    if payload.get("status") != "request_failed":
        return None

    attempt_number = 1
    while True:
        archive_path = result_path.with_name(
            "request_failure_"
            f"{attempt_number:02d}.json"
        )
        if not archive_path.exists():
            shutil.copy2(result_path, archive_path)
            return archive_path
        attempt_number += 1


def _iteration_result(
    *,
    run_id: str,
    problem: Any,
    exact_solution: Any,
    model: str,
    encoding: DemandEncoding,
    iteration: int,
    prompt: str,
    image_path: Path,
    run_dir: Path,
    prompt_path: Path,
    client: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": "visual_cvrp_refinement_iteration",
        "run_id": run_id,
        "status": "pending",
        "provider": "gemini",
        "model": model.strip(),
        "encoding": encoding.value,
        "iteration": iteration,
        "phase": (
            "initial"
            if iteration == 1
            else "refinement"
        ),
        "previous_iteration": (
            iteration - 1
            if iteration > 1
            else None
        ),
        "api_call_performed": False,
        "problem": problem.to_dict(),
        "exact_baseline": exact_solution.to_dict(),
        "prompt": prompt,
        "artifacts": {
            "problem_image": _relative(
                image_path,
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

    try:
        model_response = client.generate(
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
        return result

    result["api_call_performed"] = True
    result["model_response"] = model_response.to_dict()

    try:
        parsed = parse_model_response(
            model_response.text
        )
    except ModelResponseParseError as error:
        result["status"] = "parse_failed"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        return result

    evaluation = evaluate_solution(
        problem,
        parsed.routes,
    )
    result["status"] = "completed"
    result["parsed_solution"] = parsed.to_dict()
    result["validation"] = evaluation.to_dict()
    result["optimality_gap_percent"] = _gap_percent(
        candidate_distance=evaluation.total_distance,
        reference_distance=exact_solution.total_distance,
        candidate_valid=evaluation.valid,
    )
    return result


def _best_valid_iteration(
    results: Sequence[dict[str, Any]],
) -> int | None:
    valid_results = [
        result
        for result in results
        if result.get("status") == "completed"
        and (result.get("validation") or {}).get("valid")
        and (result.get("validation") or {}).get(
            "total_distance"
        )
        is not None
    ]
    if not valid_results:
        return None
    best = min(
        valid_results,
        key=lambda item: item["validation"][
            "total_distance"
        ],
    )
    return int(best["iteration"])


def _method_summary(
    *,
    encoding: DemandEncoding,
    model: str,
    results: Sequence[dict[str, Any]],
    maximum_iterations: int,
) -> dict[str, Any]:
    final = results[-1] if results else None
    if any(_is_exact_result(result) for result in results):
        status = "early_stopped"
        stop_reason = "proven_baseline_matched"
    elif final and final.get("status") == "request_failed":
        status = "paused"
        stop_reason = "request_failed"
    elif final and final.get("status") == "parse_failed":
        status = "stopped"
        stop_reason = "parse_failed"
    elif len(results) >= maximum_iterations:
        status = "completed"
        stop_reason = "iteration_limit"
    else:
        status = "pending"
        stop_reason = None

    return {
        "schema_version": "1.0",
        "experiment": "visual_cvrp_refinement_method",
        "provider": "gemini",
        "model": model.strip(),
        "encoding": encoding.value,
        "status": status,
        "stop_reason": stop_reason,
        "maximum_iterations": maximum_iterations,
        "recorded_iterations": [
            int(result["iteration"])
            for result in results
        ],
        "final_iteration": (
            int(final["iteration"])
            if final
            else None
        ),
        "best_valid_iteration": (
            _best_valid_iteration(results)
        ),
    }


def _write_manifest(
    *,
    path: Path,
    run_id: str,
    historical_run_id: str,
    model: str,
    encodings: Sequence[DemandEncoding],
    maximum_iterations: int,
    status: str,
    actual_api_calls: int,
    method_summaries: Sequence[dict[str, Any]],
    problem: Any,
    exact_solution: Any,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "1.0",
        "experiment": "visual_cvrp_refinement",
        "run_id": run_id,
        "historical_run_id": historical_run_id,
        "status": status,
        "provider": "gemini",
        "model": model.strip(),
        "encodings": [
            encoding.value
            for encoding in encodings
        ],
        "initial_calls_per_encoding": 1,
        "maximum_refinement_iterations": (
            maximum_iterations - 1
        ),
        "maximum_iterations_per_encoding": (
            maximum_iterations
        ),
        "maximum_total_calls": (
            maximum_iterations * len(encodings)
        ),
        "actual_api_calls": actual_api_calls,
        "problem": problem.to_dict(),
        "exact_baseline": exact_solution.to_dict(),
        "methods": list(method_summaries),
    }
    _write_json(path, manifest)
    return manifest


def execute_refinement(
    *,
    instance_file: Path | str | None = None,
    run_id: str,
    historical_run_id: str,
    output_dir: Path | str,
    model: str = DEFAULT_GEMINI_MODEL,
    encodings: Sequence[DemandEncoding | str] = (
        DEFAULT_ENCODINGS
    ),
    max_refinement_iterations: int = 3,
    validate_only: bool = False,
    resume: bool = False,
    extend_encodings: bool = False,
    client: Any | None = None,
) -> tuple[dict[str, Any], Path]:
    """Sıfırdan başlangıç ve geri bildirimli iterasyonları çalıştır."""

    if max_refinement_iterations < 0:
        raise ValueError(
            "İyileştirme iterasyonu sayısı negatif olamaz."
        )
    if extend_encodings and not resume:
        raise ValueError(
            "--extend-encodings yalnızca --resume ile "
            "kullanılabilir."
        )
    normalized_run_id = _normalize_path_component(
        run_id,
        field_name="Run ID",
    )
    normalized_historical_run_id = (
        _normalize_path_component(
            historical_run_id,
            field_name="Tarihsel run ID",
        )
    )
    normalized_model = _normalize_path_component(
        model,
        field_name="Model adı",
    )
    normalized_encodings = _normalize_encodings(
        encodings
    )
    maximum_iterations = 1 + max_refinement_iterations

    run_dir = (
        Path(output_dir)
        / "runs"
        / normalized_run_id
    )
    inputs_dir = run_dir / "inputs"
    baseline_dir = run_dir / "baseline"
    manifest_path = run_dir / "refinement_manifest.json"
    problem_path = inputs_dir / "problem.json"
    baseline_path = baseline_dir / "exact_results.json"

    if instance_file is None:
        problem = build_capacity_demo_10()
    else:
        problem = build_cvrplib_problem(
            instance_file
        )
    exact_solution = solve_exact_cvrp(problem)
    _validate_existing_manifest(
        manifest_path=manifest_path,
        historical_run_id=normalized_historical_run_id,
        model=model,
        encodings=normalized_encodings,
        maximum_iterations=maximum_iterations,
        allow_encoding_extension=extend_encodings,
        allow_iteration_extension=(
            resume and manifest_path.is_file()
        ),
    )
    method_dirs = {
        encoding: (
            run_dir
            / "providers"
            / "gemini"
            / normalized_model
            / encoding.value
        )
        for encoding in normalized_encodings
    }
    image_paths = {
        encoding: (
            inputs_dir
            / f"problem_{encoding.value}.png"
        )
        for encoding in normalized_encodings
    }
    existing_iteration_files = [
        path
        for method_dir in method_dirs.values()
        for path in method_dir.glob(
            "iteration_*/iteration_results.json"
        )
    ]
    if existing_iteration_files and not resume:
        raise FileExistsError(
            "Bu run içinde iterasyon sonuçları zaten var. "
            "Devam etmek için --resume kullanın veya yeni "
            "bir run ID seçin."
        )

    _write_json(problem_path, problem.to_dict())
    _write_json(baseline_path, exact_solution.to_dict())
    for encoding in normalized_encodings:
        method_dir = method_dirs[encoding]
        image_path = image_paths[encoding]
        render_problem(
            problem,
            image_path,
            encoding=encoding,
        )
        initial_prompt = build_solver_prompt(
            encoding=encoding
        )
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "initial_prompt.txt").write_text(
            initial_prompt + "\n",
            encoding="utf-8",
        )

    empty_summaries = [
        _method_summary(
            encoding=encoding,
            model=model,
            results=_load_iteration_results(
                method_dirs[encoding]
            ),
            maximum_iterations=maximum_iterations,
        )
        for encoding in normalized_encodings
    ]
    if validate_only:
        manifest = _write_manifest(
            path=manifest_path,
            run_id=normalized_run_id,
            historical_run_id=(
                normalized_historical_run_id
            ),
            model=model,
            encodings=normalized_encodings,
            maximum_iterations=maximum_iterations,
            status="validated_only",
            actual_api_calls=_api_attempt_count(run_dir),
            method_summaries=empty_summaries,
            problem=problem,
            exact_solution=exact_solution,
        )
        return manifest, manifest_path

    gemini_client = (
        client
        if client is not None
        else get_provider("gemini", model=model)
    )
    blocked: set[DemandEncoding] = set()
    maximum_total_calls = (
        maximum_iterations * len(normalized_encodings)
    )
    actual_api_calls = _api_attempt_count(run_dir)
    budget_exhausted = False
    reported_early_stops: set[DemandEncoding] = set()

    _progress("Görsel CVRP iyileştirme deneyi başladı.")
    _progress(
        "Planlanan azami API çağrısı: "
        f"{maximum_total_calls} | "
        f"önceden kaydedilen çağrı: {actual_api_calls}"
    )
    if resume:
        _progress(
            "Resume etkin: tamamlanmış iterasyonlar "
            "yeniden çağrılmayacak."
        )

    for iteration in range(1, maximum_iterations + 1):
        rotation = (iteration - 1) % len(
            normalized_encodings
        )
        iteration_order = (
            normalized_encodings[rotation:]
            + normalized_encodings[:rotation]
        )

        for encoding in iteration_order:
            if encoding in blocked:
                continue
            method_dir = method_dirs[encoding]
            existing_results = _load_iteration_results(
                method_dir
            )
            if any(
                _is_exact_result(result)
                for result in existing_results
            ):
                if encoding not in reported_early_stops:
                    _progress(
                        f"[{encoding.value}] Gap 0 sonucu "
                        "zaten kayıtlı; yöntem tamamlandı."
                    )
                    reported_early_stops.add(encoding)
                continue
            if actual_api_calls >= maximum_total_calls:
                budget_exhausted = True
                _progress(
                    "Azami API çağrısı bütçesine ulaşıldı; "
                    "yeni çağrı yapılmayacak."
                )
                break

            iteration_dir = (
                method_dir
                / f"iteration_{iteration:02d}"
            )
            prompt_path = iteration_dir / "prompt.txt"
            result_path = (
                iteration_dir
                / "iteration_results.json"
            )

            existing_current = next(
                (
                    result
                    for result in existing_results
                    if int(result["iteration"])
                    == iteration
                ),
                None,
            )
            if existing_current is not None:
                status = existing_current.get("status")
                if status == "completed":
                    _progress(
                        f"[{encoding.value}] İterasyon "
                        f"{iteration} kayıtlı; atlandı."
                    )
                    continue
                if status == "parse_failed":
                    _progress(
                        f"[{encoding.value}] İterasyon "
                        f"{iteration} parse_failed; yöntem "
                        "durduruldu."
                    )
                    blocked.add(encoding)
                    continue
                if status == "request_failed":
                    if not resume:
                        blocked.add(encoding)
                        continue
                    _archive_failed_attempt(result_path)

            completed_previous = [
                result
                for result in existing_results
                if result.get("status") == "completed"
                and int(result["iteration"]) < iteration
            ]
            if iteration == 1:
                prompt = build_solver_prompt(
                    encoding=encoding
                )
            else:
                previous = next(
                    (
                        result
                        for result in completed_previous
                        if int(result["iteration"])
                        == iteration - 1
                    ),
                    None,
                )
                if previous is None:
                    blocked.add(encoding)
                    continue
                prompt = build_refinement_prompt(
                    encoding=encoding,
                    previous_result=previous,
                    iteration=iteration,
                )

            iteration_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            prompt_path.write_text(
                prompt + "\n",
                encoding="utf-8",
            )
            phase_label = (
                "sıfırdan başlangıç"
                if iteration == 1
                else "geri bildirimli iyileştirme"
            )
            _progress(
                f"[API {actual_api_calls + 1}/"
                f"{maximum_total_calls}] "
                f"{encoding.value} | iterasyon "
                f"{iteration}/{maximum_iterations} | "
                f"{phase_label} başlatılıyor..."
            )
            result = _iteration_result(
                run_id=normalized_run_id,
                problem=problem,
                exact_solution=exact_solution,
                model=model,
                encoding=encoding,
                iteration=iteration,
                prompt=prompt,
                image_path=image_paths[encoding],
                run_dir=run_dir,
                prompt_path=prompt_path,
                client=gemini_client,
            )
            actual_api_calls += 1
            result["artifacts"]["result"] = _relative(
                result_path,
                run_dir,
            )
            _write_json(result_path, result)

            if result["status"] == "completed":
                validation = result["validation"] or {}
                response = result["model_response"] or {}
                _progress(
                    f"[{encoding.value}] İterasyon "
                    f"{iteration} tamamlandı | geçerli="
                    f"{validation.get('valid')} | aşım="
                    f"{validation.get('total_capacity_excess')} "
                    "| mesafe="
                    f"{validation.get('total_distance')} | gap="
                    f"{result.get('optimality_gap_percent')} | "
                    "API sn="
                    f"{response.get('elapsed_seconds')}"
                )
                if _is_exact_result(result):
                    _progress(
                        f"[{encoding.value}] Gap 0 bulundu; "
                        "bu yöntem erken durduruldu."
                    )
                    reported_early_stops.add(encoding)
            else:
                error = result.get("error") or {}
                _progress(
                    f"[{encoding.value}] İterasyon "
                    f"{iteration} tamamlanamadı | durum="
                    f"{result['status']} | hata="
                    f"{error.get('message', '-')}"
                )

            if result["status"] in {
                "request_failed",
                "parse_failed",
            }:
                blocked.add(encoding)

            current_results = _load_iteration_results(
                method_dir
            )
            summary = _method_summary(
                encoding=encoding,
                model=model,
                results=current_results,
                maximum_iterations=maximum_iterations,
            )
            _write_json(
                method_dir / "refinement_results.json",
                summary,
            )

        method_summaries = []
        for encoding in normalized_encodings:
            results = _load_iteration_results(
                method_dirs[encoding]
            )
            summary = _method_summary(
                encoding=encoding,
                model=model,
                results=results,
                maximum_iterations=maximum_iterations,
            )
            _write_json(
                method_dirs[encoding]
                / "refinement_results.json",
                summary,
            )
            method_summaries.append(summary)
        _write_manifest(
            path=manifest_path,
            run_id=normalized_run_id,
            historical_run_id=(
                normalized_historical_run_id
            ),
            model=model,
            encodings=normalized_encodings,
            maximum_iterations=maximum_iterations,
            status="running",
            actual_api_calls=actual_api_calls,
            method_summaries=method_summaries,
            problem=problem,
            exact_solution=exact_solution,
        )
        if budget_exhausted:
            break

    method_summaries = [
        _method_summary(
            encoding=encoding,
            model=model,
            results=_load_iteration_results(
                method_dirs[encoding]
            ),
            maximum_iterations=maximum_iterations,
        )
        for encoding in normalized_encodings
    ]
    if budget_exhausted:
        for summary in method_summaries:
            if summary["status"] == "pending":
                summary["status"] = "stopped"
                summary["stop_reason"] = (
                    "api_call_budget"
                )
        overall_status = "budget_exhausted"
    elif any(
        summary["status"] == "paused"
        for summary in method_summaries
    ):
        overall_status = "paused"
    elif any(
        summary["status"] == "stopped"
        for summary in method_summaries
    ):
        overall_status = "partially_completed"
    else:
        overall_status = "completed"

    manifest = _write_manifest(
        path=manifest_path,
        run_id=normalized_run_id,
        historical_run_id=normalized_historical_run_id,
        model=model,
        encodings=normalized_encodings,
        maximum_iterations=maximum_iterations,
        status=overall_status,
        actual_api_calls=actual_api_calls,
        method_summaries=method_summaries,
        problem=problem,
        exact_solution=exact_solution,
    )
    return manifest, manifest_path


def main(
    argv: list[str] | None = None,
) -> None:
    args = parse_args(argv)
    instance_file=args.instance_file,
    try:
        manifest, manifest_path = execute_refinement(
            instance_file=args.instance_file,
            run_id=args.run_id,
            historical_run_id=args.historical_run_id,
            output_dir=args.output_dir,
            model=args.model,
            encodings=args.encodings,
            max_refinement_iterations=(
                args.max_refinement_iterations
            ),
            validate_only=args.validate_only,
            resume=args.resume,
            extend_encodings=args.extend_encodings,
        )
    except KeyboardInterrupt:
        print(
            "\nDeney kullanıcı tarafından durduruldu. "
            "Tamamlanmış iterasyon kayıtları korundu. "
            "Aynı komutu --resume ile çalıştırarak devam "
            "edebilirsiniz.",
            flush=True,
        )
        raise SystemExit(130) from None

    print("Görsel CVRP iyileştirme deneyi hazırlandı.")
    print(f"Run ID: {manifest['run_id']}")
    print(
        "Kodlamalar: "
        + ", ".join(manifest["encodings"])
    )
    print(
        "Yöntem başına azami çağrı: "
        f"{manifest['maximum_iterations_per_encoding']}"
    )
    print(
        "Toplam azami çağrı: "
        f"{manifest['maximum_total_calls']}"
    )
    for method in manifest["methods"]:
        print(
            f"- {method['encoding']}: "
            f"{method['status']} | "
            "son iterasyon="
            f"{method['final_iteration']} | "
            "en iyi geçerli iterasyon="
            f"{method['best_valid_iteration']}"
        )
    if args.validate_only:
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
    print(f"Manifest: {manifest_path}")

    if manifest["status"] in {
        "paused",
        "partially_completed",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
