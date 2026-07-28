"""Bir dinamik TSP koşusu için kompakt karşılaştırma JSON'u üretir."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.analysis import build_analysis
from src.core import normalize_run_id, write_json
from src.run_manifest import load_run_problem
from src.terminal_report import compact_text, render_table


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
        help="Dört yöntem tamamlanmamışsa dosya yazmadan hata verir.",
    )
    return parser.parse_args()


def _value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _status(value: Any) -> str:
    return {
        "completed": "tamamlandı",
        "partial": "kısmi",
        "failed": "başarısız",
        "not_run": "çalışmadı",
        "legacy_or_incompatible": "uyumsuz",
    }.get(str(value), str(value))


def _method_name(value: str) -> str:
    return {
        "zero_shot": "Zero-shot",
        "multi_agent_1": "Multi-Agent 1",
        "multi_agent_2": "Multi-Agent 2",
    }.get(value, value)


def _selection_name(value: Any) -> str:
    return {
        "visual_scorer_after_feasibility_filter": "görsel scorer",
        "single_valid_candidate_without_api": "tek geçerli aday",
        "retain_previous_route_no_valid_candidate": "önceki rota",
    }.get(str(value), compact_text(value, maximum=18))


def _print_method_table(analysis: dict[str, Any]) -> None:
    baseline = analysis["methods"]["baseline"]
    baseline_solution = baseline.get("or_tools") or {}
    rows: list[list[Any]] = [
        [
            "baseline",
            "OR-Tools",
            "Referans",
            _status(baseline.get("status")),
            "-",
            baseline_solution.get("is_valid"),
            baseline_solution.get("distance"),
            baseline_solution.get("gap_to_reference_percent"),
            (baseline.get("timing_seconds") or {}).get("total"),
            "-",
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
                _status(item["status"]),
                item["completed_iterations"],
                item["is_valid"],
                item["distance"],
                item["gap_to_reference_percent"],
                item["api_wall_seconds"],
                item["total_token_count"],
                item["error_count"],
            ]
        )
    print(
        render_table(
            "Tüm sağlayıcı ve model sonuçları",
            [
                "Provider",
                "Model",
                "Yöntem",
                "Durum",
                "İter.",
                "Geçerli",
                "Mesafe",
                "Gap %",
                "API/çözüm sn",
                "Token",
                "Hata",
            ],
            rows,
            right_align={4, 6, 7, 8, 9, 10},
            max_widths={1: 28, 3: 12},
        )
    )


def _print_iterations(analysis: dict[str, Any]) -> None:
    for model in analysis.get("provider_models", []):
        label = f"{model['provider']} / {model['model_alias']}"
        ma2 = model["methods"]["multi_agent_2"]
        ma2_rows = [
            [
                item["iteration"],
                _status(item["status"]),
                item["is_valid"],
                item["distance"],
                item["gap_to_reference_percent"],
                item["timing_seconds"]["api"],
                item["timing_seconds"]["total"],
                item["total_token_count"],
            ]
            for item in ma2.get("iterations", [])
        ]
        if ma2_rows:
            print(
                render_table(
                    f"{label} — Multi-Agent 2 iterasyonları",
                    [
                        "İter.",
                        "Durum",
                        "Geçerli",
                        "Mesafe",
                        "Gap %",
                        "API sn",
                        "Toplam sn",
                        "Token",
                    ],
                    ma2_rows,
                    right_align={0, 3, 4, 5, 6, 7},
                )
            )

        ma1 = model["methods"]["multi_agent_1"]
        ma1_rows = [
            [
                item["iteration"],
                (
                    f"{item['valid_candidate_count']}/"
                    f"{item['returned_candidate_count']}"
                ),
                _selection_name(item["selection_mode"]),
                item["selected_candidate_id"],
                item["selected_distance"],
                item["best_valid_candidate_distance"],
                item["selection_regret_percent"],
                item["selected_best_valid_candidate"],
                item["timing_seconds"]["total"],
            ]
            for item in ma1.get("iterations", [])
        ]
        if ma1_rows:
            print(
                render_table(
                    f"{label} — Multi-Agent 1 iterasyonları",
                    [
                        "İter.",
                        "Geçerli aday",
                        "Seçim",
                        "Aday",
                        "Seçilen",
                        "En iyi aday",
                        "Regret %",
                        "Doğru seçim",
                        "Toplam sn",
                    ],
                    ma1_rows,
                    right_align={0, 3, 4, 5, 6, 8},
                    max_widths={2: 18},
                )
            )
            print(
                render_table(
                    f"{label} — scorer performansı",
                    [
                        "Değerlendirilen",
                        "En kısa aday seçimi",
                        "Başarı %",
                    ],
                    [
                        [
                            ma1.get(
                                "scorer_evaluated_iteration_count"
                            ),
                            ma1.get(
                                "scorer_best_candidate_selection_count"
                            ),
                            ma1.get(
                                "scorer_best_candidate_selection_rate_percent"
                            ),
                        ]
                    ],
                    right_align={0, 1, 2},
                )
            )


def _print_errors(analysis: dict[str, Any]) -> None:
    rows: list[list[Any]] = []
    for model in analysis.get("provider_models", []):
        for method_name, section in model["methods"].items():
            for error in section.get("errors", []):
                rows.append(
                    [
                        model["provider"],
                        model["model_alias"],
                        _method_name(method_name),
                        error.get("iteration"),
                        error.get("error_type"),
                        compact_text(
                            error.get("message"),
                            maximum=56,
                        ),
                    ]
                )
    if rows:
        print(
            render_table(
                "Kaydedilmiş API ve deney hataları",
                [
                    "Provider",
                    "Model",
                    "Yöntem",
                    "İter.",
                    "Hata",
                    "Mesaj",
                ],
                rows,
                right_align={3},
                max_widths={1: 24, 5: 56},
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

    analysis_dir = run_dir / "analysis"
    output_path = (
        analysis_dir
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
    _print_method_table(analysis)
    _print_iterations(analysis)
    _print_errors(analysis)
    print(f"\nAnaliz dosyası: {output_path}")


if __name__ == "__main__":
    main()
