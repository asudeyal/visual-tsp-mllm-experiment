"""Bir dinamik TSP koşusu için kompakt karşılaştırma JSON'u üretir."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.analysis import build_analysis
from src.core import normalize_run_id, write_json
from src.run_manifest import load_run_problem


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


def _print_method_table(analysis: dict[str, Any]) -> None:
    methods = analysis["methods"]
    rows = [
        (
            "OR-Tools",
            methods["baseline"].get("status"),
            methods["baseline"].get("or_tools"),
        ),
        (
            "Zero-shot",
            methods["zero_shot"].get("status"),
            methods["zero_shot"].get("solution"),
        ),
        (
            "Multi-Agent 1",
            methods["multi_agent_1"].get("status"),
            methods["multi_agent_1"].get(
                "best_valid_solution"
            ),
        ),
        (
            "Multi-Agent 2",
            methods["multi_agent_2"].get("status"),
            methods["multi_agent_2"].get(
                "best_valid_solution"
            ),
        ),
    ]
    print("\nYöntem özeti")
    print("Yöntem            Durum          Geçerli  Mesafe      Gap (%)")
    print("-" * 66)
    for name, status, solution in rows:
        solution = solution or {}
        print(
            f"{name:<17} "
            f"{str(status):<14} "
            f"{_value(solution.get('is_valid')):<8} "
            f"{_value(solution.get('distance')):<11} "
            f"{_value(solution.get('gap_to_reference_percent'))}"
        )


def _print_iterations(analysis: dict[str, Any]) -> None:
    ma2 = analysis["methods"]["multi_agent_2"]
    print("\nMulti-Agent 2 iterasyonları")
    if not ma2.get("iterations"):
        print("Kayıt yok.")
    for item in ma2.get("iterations", []):
        print(
            f"İterasyon {item['iteration']:>2}: "
            f"geçerli={item['is_valid']}, "
            f"mesafe={_value(item['distance'])}, "
            f"gap={_value(item['gap_to_reference_percent'])}%, "
            "API="
            f"{_value(item['timing_seconds']['api'])} sn, "
            "toplam="
            f"{_value(item['timing_seconds']['total'])} sn"
        )

    ma1 = analysis["methods"]["multi_agent_1"]
    print("\nMulti-Agent 1 iterasyonları")
    if not ma1.get("iterations"):
        print("Kayıt yok.")
    for item in ma1.get("iterations", []):
        print(
            f"İterasyon {item['iteration']:>2}: "
            f"geçerli aday={item['valid_candidate_count']}/"
            f"{item['returned_candidate_count']}, "
            f"seçim={item['selection_mode']}, "
            f"aday={_value(item['selected_candidate_id'])}, "
            f"mesafe={_value(item['selected_distance'])}, "
            "gap="
            f"{_value(item['selected_gap_to_reference_percent'])}%, "
            "toplam="
            f"{_value(item['timing_seconds']['total'])} sn"
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
    print("Dinamik TSP analiz raporu oluşturuldu.")
    print(f"Run ID: {run_id}")
    print(
        f"Problem: {problem['name']} "
        f"({problem['dimension']} düğüm)"
    )
    print(
        "Referans: "
        f"{reference['type']}, "
        f"mesafe={_value(reference['distance'])}, "
        "kanıtlanmış optimum="
        f"{reference['is_proven_optimal']}"
    )
    _print_method_table(analysis)
    _print_iterations(analysis)
    print(f"\nAnaliz dosyası: {output_path}")


if __name__ == "__main__":
    main()
