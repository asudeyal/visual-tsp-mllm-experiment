"""CVRP iyileştirme analiz raporu testleri."""

from __future__ import annotations

import json
from pathlib import Path

from run_refinement import execute_refinement
from run_refinement_analysis import (
    _iteration_rows,
    _method_summary_rows,
    _route_history_rows,
    build_refinement_report,
    generate_refinement_images,
    main,
)
from src.gemini_client import GeminiModelResponse


EXACT_RESPONSE = (
    '{"routes": [[0,9,2,1,0], [0,8,5,3,0], '
    '[0,7,6,4,0]]}'
)


class ExactClient:
    def generate(
        self,
        *,
        prompt: str,
        image_path: Path | str,
    ) -> GeminiModelResponse:
        return GeminiModelResponse(
            model="gemini-test",
            text=EXACT_RESPONSE,
            elapsed_seconds=2.0,
            prompt_token_count=10,
            output_token_count=5,
            thoughts_token_count=20,
            total_token_count=35,
        )


def write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def create_refinement_run(tmp_path: Path) -> Path:
    execute_refinement(
        run_id="refinement_report",
        historical_run_id="historical_pilot",
        output_dir=tmp_path,
        model="gemini-test",
        encodings=["bar_length"],
        client=ExactClient(),
    )
    historical_path = (
        tmp_path
        / "runs"
        / "historical_pilot"
        / "providers"
        / "gemini"
        / "gemini-test"
        / "bar_length"
        / "single_call_results.json"
    )
    write_json(
        historical_path,
        {
            "status": "completed",
            "provider": "gemini",
            "model": "gemini-test",
            "encoding": "bar_length",
            "validation": {
                "valid": False,
                "total_capacity_excess": 2,
                "total_distance": 500.0,
            },
            "optimality_gap_percent": None,
        },
    )
    return tmp_path / "runs" / "refinement_report"


def test_report_groups_iterations_and_removes_redundant_sections(
    tmp_path: Path,
) -> None:
    run_dir = create_refinement_run(tmp_path)
    report = build_refinement_report(
        run_dir,
        output_dir=tmp_path,
    )

    assert "Görsel CVRP iyileştirme raporu" in report
    assert "Tarihsel pilot ile yeni sıfırdan" not in report
    assert "tarihsel pilot" not in report
    assert "Yöntem sonuç özeti" in report
    assert "İterasyon gelişimi" in report
    assert "Δ Mesafe" not in report
    assert "erken durdu" not in report
    assert "kesin optimum eşleşti" in report
    assert "Provider" in report
    assert "gemini-test" in report
    assert (
        "bar — gemini / gemini-test / bar_length "
        "rota geçmişi"
    ) in report
    assert "İterasyon 1 — başlangıç" not in report
    assert "Kısıt özeti:" not in report
    assert "0→9→2→1→0" in report
    assert "Rota gösterimi:" in report
    assert "API çağrıları:" in report
    assert "Görsel çıktılar" not in report


def test_iteration_rows_are_grouped_by_encoding() -> None:
    iterations = [
        {
            "encoding": "size",
            "iteration": 1,
            "phase": "initial",
            "status": "completed",
            "validation": {},
        },
        {
            "encoding": "bar_length",
            "iteration": 2,
            "phase": "refinement",
            "status": "completed",
            "validation": {},
        },
        {
            "encoding": "bar_length",
            "iteration": 1,
            "phase": "initial",
            "status": "completed",
            "validation": {},
        },
    ]

    labels = {
        ("-", "-", "bar_length"): "bar",
        ("-", "-", "size"): "size",
    }
    rows = _iteration_rows(
        iterations,
        method_labels=labels,
    )

    assert [row[:5] for row in rows] == [
        ["bar", "-", "-", "1", "başlangıç"],
        ["bar", "-", "-", "2", "iyileştirme"],
        ["size", "-", "-", "1", "başlangıç"],
    ]


def test_method_summary_keeps_providers_separate() -> None:
    manifest = {
        "methods": [
            {
                "provider": "gemini",
                "model": "model-a",
                "encoding": "size",
                "final_iteration": 1,
            },
            {
                "provider": "openrouter",
                "model": "model-b",
                "encoding": "size",
                "final_iteration": 1,
            },
        ]
    }
    iterations = [
        {
            "provider": "gemini",
            "model": "model-a",
            "encoding": "size",
            "iteration": 1,
            "status": "completed",
            "validation": {
                "valid": True,
                "total_distance": 420.0,
            },
            "optimality_gap_percent": 0.1,
        },
        {
            "provider": "openrouter",
            "model": "model-b",
            "encoding": "size",
            "iteration": 1,
            "status": "completed",
            "validation": {
                "valid": True,
                "total_distance": 500.0,
            },
            "optimality_gap_percent": 19.1,
        },
    ]

    labels = {
        ("gemini", "model-a", "size"): "size",
        ("openrouter", "model-b", "size"): "size",
    }
    rows = _method_summary_rows(
        manifest,
        iterations,
        method_labels=labels,
    )

    assert rows[0][:3] == [
        "size",
        "gemini",
        "model-a",
    ]
    assert rows[0][5] == "420.0000"
    assert rows[1][:3] == [
        "size",
        "openrouter",
        "model-b",
    ]
    assert rows[1][5] == "500.0000"


def test_route_history_uses_one_row_per_iteration() -> None:
    rows = _route_history_rows(
        [
            {
                "iteration": 1,
                "status": "completed",
                "validation": {
                    "valid": False,
                    "routes": [
                        {
                            "route": [0, 1, 2, 0],
                            "load": 7,
                        },
                        {
                            "route": [0, 3, 0],
                            "load": 3,
                        },
                    ],
                    "missing_customer_ids": [],
                    "duplicated_customer_ids": [],
                    "unknown_node_ids": [],
                    "total_capacity_excess": 1,
                    "fleet_limit_exceeded": False,
                },
            }
        ],
        capacity=6,
    )

    assert rows == [
        [
            "1",
            "0→1→2→0 ; 0→3→0",
            "7/6 ; 3/6",
            "kapasite:+1",
        ]
    ]


def test_generate_images_and_main_write_analysis(
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = create_refinement_run(tmp_path)
    paths = generate_refinement_images(run_dir)
    assert len(paths) == 2
    assert all(path.is_file() for path in paths)

    main(
        [
            "--run-id",
            "refinement_report",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report_path = (
        run_dir
        / "analysis"
        / "terminal_analysis.txt"
    )
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "Görsel çıktılar" not in report
    assert (
        "bar — gemini / gemini-test / bar_length "
        "rota geçmişi"
    ) in report
    assert any(
        path.name.startswith(
            "gemini_gemini-test_bar_length_"
        )
        for path in paths
    )
    assert "Rapor dosyası:" in capsys.readouterr().out
