"""Geri bildirimli görsel CVRP iterasyonlarını birlikte raporla."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from run_analysis import (
    DEFAULT_OUTPUT_DIR,
    _baseline_route_table,
    _format_number,
    _load_json,
    _normalize_run_id,
    _problem_from_payload,
    render_table,
)
from src.rendering import DemandEncoding, render_solution


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def _iteration_paths(
    run_dir: Path,
) -> list[Path]:
    return sorted(
        run_dir.glob(
            "providers/*/*/*/iteration_*/"
            "iteration_results.json"
        )
    )


def _iteration_results(
    run_dir: Path,
) -> list[dict[str, Any]]:
    return [
        _load_json(path)
        for path in _iteration_paths(run_dir)
    ]


def _valid_text(validation: dict[str, Any]) -> str:
    if validation.get("valid") is True:
        return "evet"
    if validation.get("valid") is False:
        return "hayır"
    return "-"


def _usage(result: dict[str, Any]) -> dict[str, Any]:
    response = result.get("model_response") or {}
    return response.get("usage") or {}


def _phase_text(value: Any) -> str:
    return {
        "initial": "başlangıç",
        "refinement": "iyileştirme",
    }.get(str(value), str(value or "-"))


def _stop_reason_text(value: Any) -> str:
    return {
        "iteration_limit": "iterasyon sınırı",
        "proven_baseline_matched": (
            "kesin optimum eşleşti"
        ),
        "api_call_budget": "API çağrı sınırı",
        "request_failed": "API çağrısı başarısız",
        "parse_failed": "yanıt ayrıştırılamadı",
    }.get(str(value), str(value or "-"))


def _iteration_rows(
    iterations: Sequence[dict[str, Any]],
    *,
    method_labels: dict[
        tuple[str, str, str], str
    ] | None = None,
) -> list[list[str]]:
    labels = method_labels or {}
    rows = []
    for result in sorted(
        iterations,
        key=lambda item: (
            str(item.get("provider", "")),
            str(item.get("model", "")),
            str(item.get("encoding", "")),
            int(item.get("iteration", 0)),
        ),
    ):
        provider = str(result.get("provider", "-"))
        model = str(result.get("model", "-"))
        encoding = str(result.get("encoding", "-"))
        validation = result.get("validation") or {}
        distance = validation.get("total_distance")
        response = result.get("model_response") or {}
        usage = _usage(result)
        identity = (provider, model, encoding)
        method_label = labels.get(
            identity,
            " / ".join(identity),
        )
        rows.append(
            [
                method_label,
                provider,
                model,
                _format_number(result.get("iteration")),
                _phase_text(result.get("phase")),
                _valid_text(validation),
                _format_number(
                    validation.get("total_capacity_excess")
                ),
                _format_number(distance),
                _format_number(
                    result.get("optimality_gap_percent")
                ),
                _format_number(
                    response.get("elapsed_seconds")
                ),
                _format_number(
                    usage.get("total_token_count")
                ),
            ]
        )
    return rows


def _best_valid(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    valid = [
        result
        for result in results
        if result.get("status") == "completed"
        and (result.get("validation") or {}).get("valid")
        and isinstance(
            (result.get("validation") or {}).get(
                "total_distance"
            ),
            (int, float),
        )
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda item: item["validation"][
            "total_distance"
        ],
    )


def _method_summary_rows(
    manifest: dict[str, Any],
    iterations: Sequence[dict[str, Any]],
    *,
    method_labels: dict[
        tuple[str, str, str], str
    ] | None = None,
) -> list[list[str]]:
    labels = method_labels or {}
    rows = []
    for method in manifest.get("methods") or []:
        provider = str(
            method.get("provider")
            or manifest.get("provider")
            or "-"
        )
        model = str(
            method.get("model")
            or manifest.get("model")
            or "-"
        )
        encoding = str(method.get("encoding"))
        method_results = [
            result
            for result in iterations
            if str(result.get("provider", "-"))
            == provider
            and str(result.get("model", "-"))
            == model
            and str(result.get("encoding"))
            == encoding
        ]
        best = _best_valid(method_results)
        best_validation = (
            best.get("validation") or {}
            if best
            else {}
        )
        rows.append(
            [
                labels.get(
                    (provider, model, encoding),
                    "-",
                ),
                provider,
                model,
                _format_number(
                    method.get("final_iteration")
                ),
                _format_number(
                    best.get("iteration")
                    if best
                    else None
                ),
                _format_number(
                    best_validation.get("total_distance")
                ),
                _format_number(
                    best.get("optimality_gap_percent")
                    if best
                    else None
                ),
                _stop_reason_text(
                    method.get("stop_reason")
                ),
            ]
        )
    return rows


def _method_identity(
    item: dict[str, Any],
    *,
    manifest: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        str(
            item.get("provider")
            or manifest.get("provider")
            or "-"
        ),
        str(
            item.get("model")
            or manifest.get("model")
            or "-"
        ),
        str(item.get("encoding") or "-"),
    )


def _method_identities(
    manifest: dict[str, Any],
    iterations: Sequence[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    identities: list[tuple[str, str, str]] = []
    for item in manifest.get("methods") or []:
        identity = _method_identity(
            item,
            manifest=manifest,
        )
        if identity not in identities:
            identities.append(identity)
    for item in sorted(
        iterations,
        key=lambda result: (
            str(result.get("provider", "")),
            str(result.get("model", "")),
            str(result.get("encoding", "")),
        ),
    ):
        identity = _method_identity(
            item,
            manifest=manifest,
        )
        if identity not in identities:
            identities.append(identity)
    return identities


def _method_labels(
    identities: Sequence[tuple[str, str, str]],
) -> dict[tuple[str, str, str], str]:
    display_names = {
        "bar_length": "bar",
        "color_intensity": "color",
        "size": "size",
    }
    return {
        identity: display_names.get(
            identity[2],
            identity[2],
        )
        for identity in identities
    }


def _compact_route(route: Sequence[Any]) -> str:
    return "→".join(str(node_id) for node_id in route)


def _compact_ids(values: Any) -> str:
    if not values:
        return "-"
    return ",".join(str(value) for value in values)


def _violation_text(
    validation: dict[str, Any],
) -> str:
    violations = []
    missing = validation.get("missing_customer_ids")
    duplicated = validation.get("duplicated_customer_ids")
    unknown = validation.get("unknown_node_ids")
    capacity_excess = validation.get(
        "total_capacity_excess"
    )

    if missing:
        violations.append(f"eksik:{_compact_ids(missing)}")
    if duplicated:
        violations.append(
            f"tekrar:{_compact_ids(duplicated)}"
        )
    if unknown:
        violations.append(
            f"bilinmeyen:{_compact_ids(unknown)}"
        )
    if isinstance(capacity_excess, (int, float)):
        if capacity_excess > 0:
            violations.append(
                "kapasite:+"
                f"{_format_number(capacity_excess)}"
            )
    if validation.get("fleet_limit_exceeded"):
        violations.append("filo:aşıldı")
    if (
        validation.get("valid") is False
        and not violations
    ):
        violations.append("geçersiz")
    return " | ".join(violations) if violations else "-"


def _route_history_rows(
    results: Sequence[dict[str, Any]],
    *,
    capacity: int,
) -> list[list[str]]:
    rows = []
    for result in sorted(
        results,
        key=lambda item: int(item.get("iteration", 0)),
    ):
        validation = result.get("validation")
        if (
            result.get("status") != "completed"
            or not isinstance(validation, dict)
        ):
            rows.append(
                [
                    _format_number(result.get("iteration")),
                    "-",
                    "-",
                    _stop_reason_text(result.get("status")),
                ]
            )
            continue

        routes = validation.get("routes") or []
        route_text = " ; ".join(
            _compact_route(route.get("route") or [])
            for route in routes
        )
        load_text = " ; ".join(
            f"{_format_number(route.get('load'))}/{capacity}"
            for route in routes
        )
        rows.append(
            [
                _format_number(result.get("iteration")),
                route_text or "-",
                load_text or "-",
                _violation_text(validation),
            ]
        )
    return rows


def _filename_component(value: Any) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        str(value or "unknown"),
    ).strip("-._")
    return normalized or "unknown"


def generate_refinement_images(
    run_dir: Path,
) -> list[Path]:
    problem_payload = _load_json(
        run_dir / "inputs" / "problem.json"
    )
    baseline = _load_json(
        run_dir / "baseline" / "exact_results.json"
    )
    problem = _problem_from_payload(problem_payload)
    images_dir = run_dir / "analysis" / "images"
    generated = []

    for stale_path in images_dir.glob(
        "*_iteration_*_routes.png"
    ):
        if stale_path.is_file():
            stale_path.unlink()

    baseline_path = images_dir / "baseline_exact_routes.png"
    render_solution(
        problem,
        baseline.get("routes") or [],
        baseline_path,
        title=(
            "Exact CVRP baseline — distance "
            f"{_format_number(baseline.get('total_distance'))}"
        ),
        route_loads=baseline.get("route_loads") or None,
        encoding=DemandEncoding.NUMERIC,
    )
    generated.append(baseline_path)

    for result in _iteration_results(run_dir):
        validation = result.get("validation") or {}
        if (
            result.get("status") != "completed"
            or not validation.get("routes")
        ):
            continue
        encoding = DemandEncoding(result["encoding"])
        provider = str(result.get("provider") or "unknown")
        model = str(result.get("model") or "unknown")
        iteration = int(result["iteration"])
        image_path = images_dir / (
            f"{_filename_component(provider)}_"
            f"{_filename_component(model)}_"
            f"{_filename_component(encoding.value)}_iteration_"
            f"{iteration:02d}_routes.png"
        )
        title = (
            f"{provider} / {model} / {encoding.value} — "
            f"iteration {iteration} — "
            "distance "
            f"{_format_number(validation.get('total_distance'))}"
        )
        gap = result.get("optimality_gap_percent")
        if gap is not None:
            title += f" — gap {_format_number(gap)}%"
        render_solution(
            problem,
            [
                route.get("route") or []
                for route in validation["routes"]
            ],
            image_path,
            title=title,
            route_loads=[
                int(route.get("load", 0))
                for route in validation["routes"]
            ],
            encoding=encoding,
        )
        generated.append(image_path)
    return generated


def build_refinement_report(
    run_dir: Path,
    *,
    output_dir: Path,
    image_paths: Sequence[Path] = (),
) -> str:
    manifest = _load_json(
        run_dir / "refinement_manifest.json"
    )
    problem = _load_json(
        run_dir / "inputs" / "problem.json"
    )
    baseline = _load_json(
        run_dir / "baseline" / "exact_results.json"
    )
    iterations = _iteration_results(run_dir)
    capacity = int(problem["vehicle_capacity"])
    identities = _method_identities(
        manifest,
        iterations,
    )
    method_labels = _method_labels(identities)

    lines = [
        "Görsel CVRP iyileştirme raporu",
        "",
        f"Run ID: {run_dir.name}",
        (
            f"Problem: {problem['name']} | "
            f"Düğüm: {problem['dimension']} | "
            f"Müşteri: {problem['customer_count']}"
        ),
        (
            f"Araç kapasitesi Q: {capacity} | "
            f"Araç sınırı K: {problem['vehicle_count']} | "
            f"Toplam talep: {problem['total_demand']}"
        ),
        (
            "Kesin optimum: "
            f"{_format_number(baseline['total_distance'])} | "
            "Kanıtlı: "
            f"{'evet' if baseline.get('proven_optimal') else 'hayır'}"
        ),
        (
            "Plan: yöntem başına 1 sıfırdan + en fazla "
            f"{manifest['maximum_refinement_iterations']} "
            "iyileştirme çağrısı"
        ),
    ]

    actual_api_calls = manifest.get("actual_api_calls")
    if isinstance(actual_api_calls, int):
        additional_calls = max(
            actual_api_calls - len(iterations),
            0,
        )
        lines.append(
            f"API çağrıları: {actual_api_calls} | "
            f"Kayıtlı iterasyon: {len(iterations)} | "
            "Yeniden denenen/ek çağrı: "
            f"{additional_calls}"
        )

    lines.extend(
        [
            "",
            "Yöntem sonuç özeti",
            render_table(
                [
                    "Yöntem",
                    "Provider",
                    "Model",
                    "Son iter",
                    "En iyi iter",
                    "En iyi mesafe",
                    "En iyi gap %",
                    "Durma nedeni",
                ],
                _method_summary_rows(
                    manifest,
                    iterations,
                    method_labels=method_labels,
                ),
                right_aligned={3, 4, 5, 6},
            ),
            "",
            "İterasyon gelişimi",
            render_table(
                [
                    "Yöntem",
                    "Provider",
                    "Model",
                    "İter",
                    "Aşama",
                    "Geçerli",
                    "Kapasite aşımı",
                    "Mesafe",
                    "Gap %",
                    "API sn",
                    "Token",
                ],
                _iteration_rows(
                    iterations,
                    method_labels=method_labels,
                ),
                right_aligned={3, 6, 7, 8, 9, 10},
            )
            if iterations
            else "Kayıtlı iterasyon bulunamadı.",
            "",
            "Kesin optimum rotaları",
            _baseline_route_table(
                baseline,
                problem=problem,
            ),
        ]
    )

    if identities:
        lines.extend(
            [
                "",
                (
                    "Rota gösterimi: Araç rotaları ve "
                    "karşılık gelen yükler aynı sırada "
                    '";" ile ayrılmıştır.'
                ),
            ]
        )
    for provider, model, encoding in identities:
        method_label = method_labels[
            (provider, model, encoding)
        ]
        method_results = sorted(
            (
                result
                for result in iterations
                if str(result.get("provider", "-"))
                == provider
                and str(result.get("model", "-"))
                == model
                and str(result.get("encoding"))
                == encoding
            ),
            key=lambda item: int(
                item.get("iteration", 0)
            ),
        )
        if not method_results:
            continue
        lines.extend(
            [
                "",
                (
                    f"{method_label} — {provider} / {model} "
                    f"/ {encoding} "
                    "rota geçmişi"
                ),
                render_table(
                    [
                        "İter",
                        "Rotalar",
                        "Yükler",
                        "İhlal",
                    ],
                    _route_history_rows(
                        method_results,
                        capacity=capacity,
                    ),
                    right_aligned={0},
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main(
    argv: list[str] | None = None,
) -> None:
    args = parse_args(argv)
    run_id = _normalize_run_id(args.run_id)
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "runs" / run_id
    if not run_dir.is_dir():
        raise SystemExit(
            f"Run klasörü bulunamadı: {run_dir}"
        )

    image_paths = generate_refinement_images(run_dir)
    report = build_refinement_report(
        run_dir,
        output_dir=output_dir,
        image_paths=image_paths,
    )
    report_path = (
        run_dir
        / "analysis"
        / "terminal_analysis.txt"
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        report,
        encoding="utf-8",
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report, end="")
    print(f"Rapor dosyası: {report_path}")


if __name__ == "__main__":
    main()
