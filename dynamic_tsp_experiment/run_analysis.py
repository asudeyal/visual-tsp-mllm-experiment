"""Bir dinamik TSP koşusu için kompakt karşılaştırma JSON'u üretir."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from src.analysis import build_analysis
from src.core import normalize_run_id, write_json
from src.run_manifest import load_run_problem
from src.terminal_report import (
    compact_text,
    render_note,
    render_summary,
    render_table,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Tüm provider/model yöntemleri tamamlanmamışsa "
            "dosya yazmadan hata verir."
        ),
    )
    return parser.parse_args()


def _method_name(value: str) -> str:
    return {
        "zero_shot": "Zero-shot",
        "multi_agent_1": "Multi-Agent 1",
        "multi_agent_2": "Multi-Agent 2",
        "baseline": "Baseline",
    }.get(value, value)


def _selection_name(value: Any) -> str:
    return {
        "visual_scorer_after_feasibility_filter": "scorer",
        "single_valid_candidate_without_api": "tek aday",
        "retain_previous_route_no_valid_candidate": "fallback",
    }.get(str(value), compact_text(value, maximum=14))


def _selection_label(item: dict[str, Any]) -> str:
    valid = (
        f"{item.get('valid_candidate_count', 0)}/"
        f"{item.get('returned_candidate_count', 0)}"
    )
    name = _selection_name(item.get("selection_mode"))
    candidate_id = item.get("selected_candidate_id")
    choice = name if candidate_id is None else f"{name}(#{candidate_id})"
    return f"{valid} {choice}"


def _percent_count(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "-"
    return (
        f"{numerator}/{denominator} "
        f"(%{100.0 * numerator / denominator:.1f})"
    )


def _improvement_percent(
    initial_distance: Any,
    final_distance: Any,
) -> str | None:
    if (
        initial_distance is None
        or final_distance is None
        or float(initial_distance) == 0
    ):
        return None
    value = (
        100.0
        * (float(initial_distance) - float(final_distance))
        / float(initial_distance)
    )
    return f"%{value:.4f}"


def _print_method_table(analysis: dict[str, Any]) -> None:
    baseline = analysis["methods"]["baseline"]
    baseline_solution = baseline.get("or_tools") or {}
    baseline_timing = baseline.get("timing_seconds") or {}
    rows: list[list[Any]] = [
        [
            "baseline",
            "OR-Tools",
            "Referans",
            "-",
            baseline_solution.get("is_valid"),
            baseline_solution.get("distance"),
            baseline_solution.get("gap_to_reference_percent"),
            "-",
            baseline_timing.get("total"),
            0.0,
            0.0,
            0,
        ]
    ]
    for item in analysis["comparison"].get(
        "all_model_method_rows",
        [],
    ):
        rows.append(
            [
                item["provider"],
                item["model_alias"],
                _method_name(item["method"]),
                item["completed_iterations"],
                item["is_valid"],
                item["distance"],
                item["gap_to_reference_percent"],
                item["total_token_count"],
                item.get("active_wall_seconds"),
                item.get("deliberate_delay_seconds"),
                item.get("rate_limit_backoff_seconds"),
                item["error_count"],
            ]
        )
    print(
        render_table(
            "Tüm sağlayıcı, model ve yöntem sonuçları",
            [
                "Provider",
                "Model",
                "Yöntem",
                "İter.",
                "Geçerli",
                "Mesafe",
                "Gap %",
                "Token",
                "Aktif sn",
                "Planlı sn",
                "Backoff sn",
                "Hata",
            ],
            rows,
            right_align={3, 5, 6, 7, 8, 9, 10, 11},
            max_widths={1: 28},
        )
    )


def _termination_text(section: dict[str, Any]) -> str:
    termination = section.get("termination") or {}
    reason = termination.get("reason")
    if reason == "early_stop":
        early = termination.get("early_stop") or {}
        return (
            "erken durdurma; "
            f"GBest iter={early.get('system_gbest_iteration')}, "
            f"gap=%{early.get('system_gbest_gap_percent')}, "
            f"eşik=%{early.get('threshold_percent')}"
        )
    if reason == "iteration_failed":
        return (
            "iterasyon hatası; "
            f"iter={termination.get('failed_iteration')}"
        )
    if reason:
        return str(reason)
    return "normal tamamlanma veya tarihsel çıktı"


def _print_ma2_iterations(
    label: str,
    section: dict[str, Any],
) -> None:
    iterations = section.get("iterations", [])
    if not iterations:
        return
    rows = [
        [
            item["iteration"],
            item["is_valid"],
            item["distance"],
            item.get("iteration_best_distance"),
            item.get("system_gbest_distance"),
            item.get("system_gbest_gap_percent"),
            item["total_token_count"],
            item["timing_seconds"].get("active"),
            item["timing_seconds"].get("deliberate_delay"),
            item["timing_seconds"].get("rate_limit_backoff"),
            item["timing_seconds"].get("total"),
        ]
        for item in iterations
    ]
    print(
        render_table(
            f"{label} — Multi-Agent 2 iterasyonları",
            [
                "İter.",
                "Geçerli",
                "Mesafe",
                "İter. en iyi",
                "Sistem GBest",
                "GBest gap %",
                "Token",
                "Aktif sn",
                "Planlı sn",
                "Backoff sn",
                "Toplam sn",
            ],
            rows,
            right_align={0, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        )
    )
    initializer = section.get("initializer") or {}
    best = section.get("best_valid_solution") or {}
    final = section.get("final_solution") or {}
    print(
        render_summary(
            [
                (
                    "geçerli iterasyon",
                    _percent_count(
                        int(section.get("valid_iteration_count") or 0),
                        len(iterations),
                    ),
                ),
                ("en iyi mesafe", best.get("distance")),
                ("son mesafe", final.get("distance")),
                (
                    "başlangıca göre iyileşme",
                    _improvement_percent(
                        initializer.get("distance"),
                        best.get("distance"),
                    ),
                ),
                ("bitiş", _termination_text(section)),
            ],
            fields_per_line=3,
        )
    )


def _print_ma1_iterations(
    label: str,
    section: dict[str, Any],
) -> None:
    iterations = section.get("iterations", [])
    if not iterations:
        return
    rows = [
        [
            item["iteration"],
            _selection_label(item),
            item["selected_distance"],
            item.get("iteration_best_distance"),
            item.get("system_gbest_distance"),
            item.get("system_gbest_gap_percent"),
            item.get("observed_candidate_gbest_distance"),
            item["selection_regret_percent"],
            (item.get("token_count") or {}).get("total"),
            item["timing_seconds"].get("active"),
            item["timing_seconds"].get("deliberate_delay"),
            item["timing_seconds"].get("rate_limit_backoff"),
            item["timing_seconds"].get("total"),
        ]
        for item in iterations
    ]
    print(
        render_table(
            f"{label} — Multi-Agent 1 iterasyonları",
            [
                "İter.",
                "Aday / seçim",
                "Seçilen",
                "İter. en iyi",
                "Sistem GBest",
                "GBest gap %",
                "Aday GBest",
                "Regret %",
                "Token",
                "Aktif sn",
                "Planlı sn",
                "Backoff sn",
                "Toplam sn",
            ],
            rows,
            right_align={0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
            max_widths={1: 20},
        )
    )
    fallback_count = sum(
        item.get("selection_mode")
        == "retain_previous_route_no_valid_candidate"
        for item in iterations
    )
    scorer_count = int(
        section.get("scorer_evaluated_iteration_count") or 0
    )
    scorer_best = int(
        section.get("scorer_best_candidate_selection_count") or 0
    )
    progress = section.get("solution_progress") or {}
    best_system = (
        progress.get("system_gbest")
        or section.get("best_valid_solution")
        or {}
    )
    observed = progress.get("observed_candidate_gbest") or {}
    if not observed and iterations:
        observed_distances = [
            (
                item.get("observed_candidate_gbest_distance"),
                item.get("iteration"),
            )
            for item in iterations
            if item.get("observed_candidate_gbest_distance")
            is not None
        ]
        if observed_distances:
            observed_distance = min(
                value[0] for value in observed_distances
            )
            observed = {
                "distance": observed_distance,
                "iteration": next(
                    iteration
                    for distance, iteration in observed_distances
                    if distance == observed_distance
                ),
            }
    print(
        render_summary(
            [
                (
                    "geçerli aday",
                    _percent_count(
                        int(section.get("valid_candidate_count") or 0),
                        int(section.get("total_candidate_count") or 0),
                    ),
                ),
                (
                    "scorer en kısa seçimi",
                    _percent_count(scorer_best, scorer_count),
                ),
                ("fallback", fallback_count),
                ("sistem GBest", best_system.get("distance")),
                (
                    "GBest gap %",
                    best_system.get("gap_to_reference_percent"),
                ),
                ("GBest iter.", best_system.get("iteration")),
                ("gözlenen aday GBest", observed.get("distance")),
                (
                    "aday GBest gap %",
                    observed.get("gap_to_reference_percent"),
                ),
                ("aday GBest iter.", observed.get("iteration")),
                ("bitiş", _termination_text(section)),
            ],
            fields_per_line=3,
        )
    )


def _print_iterations(analysis: dict[str, Any]) -> None:
    for model in analysis.get("provider_models", []):
        label = f"{model['provider']} / {model['model_alias']}"
        _print_ma2_iterations(
            label,
            model["methods"]["multi_agent_2"],
        )
        _print_ma1_iterations(
            label,
            model["methods"]["multi_agent_1"],
        )


def _metric_pair(
    resources: dict[str, Any],
    name: str,
) -> str:
    metric = (
        resources.get("overall_metrics") or {}
    ).get(name)
    if not isinstance(metric, dict):
        return "-"
    average = metric.get("average")
    maximum = metric.get("maximum")
    if average is None and maximum is None:
        return "-"
    return f"{float(average):.1f}/{float(maximum):.1f}"


def _gpu_status(resources: dict[str, Any]) -> str:
    if resources.get("enabled") is not True:
        return "profil kapalı"
    gpu = resources.get("local_gpu") or {}
    if gpu.get("available") is True:
        utilization = _metric_pair(
            resources,
            "local_gpu_utilization_percent",
        )
        return f"ölçüldü {utilization}%"
    reason = str(gpu.get("unavailable_reason") or "desteklenmiyor")
    if "NVML" in reason or "LibraryNotFound" in reason:
        return "ölçülemedi (NVIDIA/NVML yok)"
    return f"ölçülemedi ({compact_text(reason, maximum=22)})"


def _resource_sections(
    analysis: dict[str, Any],
) -> Iterable[tuple[str, str, str, dict[str, Any]]]:
    baseline = analysis["methods"]["baseline"]
    yield (
        "baseline",
        "OR-Tools",
        "Referans",
        baseline,
    )
    for model in analysis.get("provider_models", []):
        for method_name, section in model["methods"].items():
            yield (
                model["provider"],
                model["model_alias"],
                _method_name(method_name),
                section,
            )


def _print_resources(analysis: dict[str, Any]) -> None:
    records: list[tuple[list[Any], dict[str, Any]]] = []
    for provider, model, method, section in _resource_sections(analysis):
        observability = section.get("observability")
        if not isinstance(observability, dict):
            continue
        resources = observability.get("resources") or {}
        records.append(
            (
                [
                    provider,
                    model,
                    method,
                    resources.get("sample_count"),
                    _metric_pair(resources, "system_cpu_percent"),
                    _metric_pair(resources, "process_cpu_percent"),
                    _metric_pair(resources, "process_memory_rss_mb"),
                    _metric_pair(resources, "system_memory_percent"),
                ],
                resources,
            )
        )
    if not records:
        return

    show_gpu = any(
        (resources.get("local_gpu") or {}).get("available") is True
        for _, resources in records
    )
    headers = [
        "Provider",
        "Model",
        "Yöntem",
        "Ölçüm",
        "Sistem CPU %",
        "Süreç CPU %",
        "Süreç RSS MB",
        "Sistem RAM %",
    ]
    rows = [row.copy() for row, _ in records]
    max_widths = {1: 24}
    if show_gpu:
        headers.append("Yerel GPU")
        for row, (_, resources) in zip(rows, records):
            row.append(_gpu_status(resources))
        max_widths[8] = 30

    print(
        render_table(
            "Yerel kaynak kullanımı (ortalama/azami)",
            headers,
            rows,
            right_align={3},
            max_widths=max_widths,
        )
    )
    if not show_gpu:
        statuses = sorted(
            {
                _gpu_status(resources)
                for _, resources in records
            }
        )
        print(
            render_note(
                "Yerel GPU",
                ["; ".join(statuses)],
            )
        )


def _print_errors(analysis: dict[str, Any]) -> None:
    rows: list[list[Any]] = []
    for model in analysis.get("provider_models", []):
        for method_name, section in model["methods"].items():
            errors = section.get("errors", [])
            if not errors:
                continue
            last_error = errors[-1]
            recovered = section.get("status") == "completed"
            rows.append(
                [
                    model["provider"],
                    model["model_alias"],
                    _method_name(method_name),
                    "aşıldı" if recovered else "çözülmedi",
                    len(errors),
                    last_error.get("error_type"),
                    compact_text(
                        last_error.get("message"),
                        maximum=42,
                    ),
                ]
            )
    if rows:
        print(
            render_table(
                "Hata özeti (ayrıntılar analiz JSON'unda)",
                [
                    "Provider",
                    "Model",
                    "Yöntem",
                    "Durum",
                    "Adet",
                    "Hata",
                    "Son mesaj",
                ],
                rows,
                right_align={4},
                max_widths={1: 24, 6: 42},
            )
        )


def main() -> None:
    args = parse_args()
    run_id = normalize_run_id(args.run_id)
    run_dir = Path(args.output_dir) / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            "Run manifesti bulunamadı. Önce baseline çalıştırılmalıdır."
        )
    manifest, _ = load_run_problem(manifest_path)
    analysis = build_analysis(
        run_dir=run_dir,
        manifest=manifest,
    )
    if (
        args.require_complete
        and not analysis["completion"]["all_methods_completed"]
    ):
        raise SystemExit(
            "Analiz tamamlanmadı: yöntemlerden en az biri eksik "
            "veya kısmi. --require-complete olmadan kısmi rapor "
            "üretebilirsiniz."
        )

    output_path = (
        run_dir
        / "analysis"
        / "experiment_analysis_summary.json"
    )
    write_json(output_path, analysis)

    problem = analysis["problem"]
    reference = problem["reference"]
    print(
        render_table(
            "Dinamik TSP analiz raporu",
            [
                "Run ID",
                "Problem",
                "Düğüm",
                "Referans",
                "Mesafe",
                "Optimum kanıtlı",
            ],
            [
                [
                    run_id,
                    problem["name"],
                    problem["dimension"],
                    reference["type"],
                    reference["distance"],
                    reference["is_proven_optimal"],
                ]
            ],
            right_align={2, 4},
            max_widths={0: 24, 1: 28, 3: 28},
        )
    )
    print(
        render_note(
            "Sürelerin yorumu",
            [
                "Aktif sn, planlı bekleme ve rate-limit backoff süreleri çıkarılmış yerel + API süresidir.",
                "API aktif süresi ağ, sağlayıcı kuyruğu ve uzak model çıkarımını birlikte içerir; bunlar istemci tarafında kesin ayrıştırılamaz.",
                "CPU/RAM yerel bilgisayarı gösterir. Uzak API modelinin GPU/CPU kullanımı bu ölçüme dahil değildir.",
            ],
        )
    )
    _print_method_table(analysis)
    _print_iterations(analysis)
    _print_resources(analysis)
    _print_errors(analysis)
    print(f"\nAnaliz dosyası: {output_path}")


if __name__ == "__main__":
    main()
