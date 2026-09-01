"""Geri bildirimli görsel CVRP iterasyonlarını birlikte raporla."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.rendering import DemandEncodingConfig, RenderConfig, RouteRenderingConfig, render_diagnostic_routes
from src.schemas import ProblemInstance

DEFAULT_OUTPUT_DIR = Path("output")


def _format_number(value: Any) -> str:
    """Sayıları formatlar."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "evet" if value else "hayır"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    """JSON dosyasını okur."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _normalize_run_id(value: str) -> str:
    """Run ID'yi temizler."""
    return value.strip()


def _problem_from_payload(payload: dict[str, Any]) -> ProblemInstance:
    """Problem nesnesini yükler."""
    return ProblemInstance(
        name=payload["name"],
        dimension=payload["dimension"],
        node_ids=tuple(payload["node_ids"]),
        coordinates={int(k): tuple(v) for k, v in payload["coordinates"].items()},
        depot=payload["depot"],
        capacity=payload["capacity"],
        demands={int(k): v for k, v in payload["demands"].items()},
        max_vehicles=payload.get("max_vehicles"),
        edge_weight_type=payload.get("edge_weight_type", "EUC_2D"),
        source_path=payload.get("source_path"),
        source_sha256=payload.get("source_sha256"),
        reference_optimum=payload.get("reference_optimum"),
    )


def render_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    right_aligned: set[int] | None = None,
) -> str:
    """Tabloyu render eder."""
    values = [[_format_number(value) for value in row] for row in rows]
    header_values = [str(value) for value in headers]
    if any(len(row) != len(header_values) for row in values):
        raise ValueError("Tablo satır ve başlık sütun sayıları eşit değil")

    widths = [
        max(len(header_values[index]), *([len(row[index]) for row in values] or [0]))
        for index in range(len(header_values))
    ]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("-" * (width + 2) for width in widths) + right

    aligns = right_aligned or set()

    def row_text(row: Sequence[str]) -> str:
        cells: list[str] = []
        for index, value in enumerate(row):
            rendered = value.rjust(widths[index]) if index in aligns else value.ljust(widths[index])
            cells.append(f" {rendered} ")
        return "|" + "|".join(cells) + "|"

    lines = [border("+", "+", "+"), row_text(header_values), border("+", "+", "+")]
    if values:
        lines.extend(row_text(row) for row in values)
    else:
        empty = ["Kayıt yok.", *([""] * (len(headers) - 1))]
        lines.append(row_text(empty))
    lines.append(border("+", "+", "+"))
    return "\n".join(lines)


def _baseline_route_table(baseline: dict[str, Any], problem: dict[str, Any]) -> str:
    """Kesin optimum rotalarını tablo halinde döndürür."""
    routes = baseline.get("routes") or []
    capacity = int(problem.get("vehicle_capacity", 0))
    
    rows = []
    for i, route in enumerate(routes):
        route_path = " -> ".join(str(n) for n in route.get("route", []))
        load = route.get("load", 0)
        distance = route.get("distance", 0)
        rows.append([
            i + 1,
            route_path,
            f"{load}/{capacity}",
            0,
            distance,
            "evet"
        ])
    return render_table(
        ["Rota", "Yol", "Yük/Q", "Aşım", "Mesafe", "Geçerli"],
        rows,
        right_aligned={0, 3, 4}
    )


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
    gbest_by_method: dict[tuple[str, str, str], float] = {}
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
        if (
            result.get("status") == "completed"
            and validation.get("valid") is True
            and isinstance(distance, (int, float))
        ):
            previous_gbest = gbest_by_method.get(identity)
            if previous_gbest is None or distance < previous_gbest:
                gbest_by_method[identity] = float(distance)
        gbest = gbest_by_method.get(identity)
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
                _format_number(gbest),
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
        "scale_position": "scale",
        "radial_fill": "radial",
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

    # Problem nesnesini yeni şemaya göre oluştur.
    # Bu kısım _problem_from_payload fonksiyonunun nasıl çalıştığına 
    # veya ProblemInstance'ı nasıl başlattığına bağlıdır.
    # Örnek olarak:
    problem = _problem_from_payload(problem_payload) 

    # Eğer _problem_from_payload doğrudan ProblemInstance dönmüyorsa, 
    # aşağıdaki gibi oluşturman gerekebilir:
    # problem = ProblemInstance(**problem_payload)

    images_dir = run_dir / "analysis" / "images"
    generated = []

    for stale_path in images_dir.glob(
        "*_iteration_*_routes.png"
    ):
        if stale_path.is_file():
            stale_path.unlink()

    baseline_path = images_dir / "baseline_exact_routes.png"

    # DemandEncodingConfig'i varsayılan (sayısal) olarak ayarla
    default_encoding = DemandEncodingConfig(mode="none")
    render_cfg = RenderConfig()

    render_diagnostic_routes(
        problem,
        [route.get("route", []) for route in baseline.get("routes", [])],
        baseline_path,
        render_cfg,
        demand_encoding=default_encoding
    )
    generated.append(baseline_path)

    for result in _iteration_results(run_dir):
        validation = result.get("validation") or {}
        if (
            result.get("status") != "completed"
            or not validation.get("routes")
        ):
            continue

        # String kodlamayı DemandEncodingConfig mode parametresine çevir
        encoding_str = str(result["encoding"])

        # Eğer encoding_str doğrudan uyumlu değilse eşlemen gerekebilir.
        # Örneğin: "NUMERIC" -> "none"
        mode = encoding_str.lower()
        if mode == "numeric":
             mode = "none"

        encoding_cfg = DemandEncodingConfig(mode=mode)

        provider = str(result.get("provider") or "unknown")
        model = str(result.get("model") or "unknown")
        iteration = int(result["iteration"])

        image_path = images_dir / (
            f"{_filename_component(provider)}_"
            f"{_filename_component(model)}_"
            f"{_filename_component(encoding_str)}_iteration_"
            f"{iteration:02d}_routes.png"
        )

        render_diagnostic_routes(
            problem,
            [
                route.get("route") or []
                for route in validation["routes"]
            ],
            image_path,
            render_cfg,
            demand_encoding=encoding_cfg,
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
                    "GBest",
                    "API sn",
                    "Token",
                ],
                _iteration_rows(
                    iterations,
                    method_labels=method_labels,
                ),
                right_aligned={3, 6, 7, 8, 9, 10, 11},
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