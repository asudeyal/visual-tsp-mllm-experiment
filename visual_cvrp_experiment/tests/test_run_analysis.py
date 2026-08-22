"""Görsel CVRP terminal analiz raporu testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_analysis import (
    _normalize_run_id,
    build_report,
    main,
    render_table,
)


def write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def create_run(
    output_dir: Path,
    *,
    status: str = "completed",
) -> Path:
    run_dir = output_dir / "runs" / "report_run"

    write_json(
        run_dir / "inputs" / "problem.json",
        {
            "name": "capacity_demo_10",
            "dimension": 10,
            "customer_count": 9,
            "depot_id": 0,
            "vehicle_capacity": 6,
            "vehicle_count": 3,
            "total_demand": 18,
            "nodes": [
                {
                    "id": 0,
                    "x": 0.0,
                    "y": 0.0,
                    "demand": 0,
                },
                {
                    "id": 1,
                    "x": 3.0,
                    "y": 0.0,
                    "demand": 3,
                },
                {
                    "id": 2,
                    "x": 3.0,
                    "y": 4.0,
                    "demand": 3,
                },
            ],
        },
    )
    write_json(
        run_dir
        / "baseline"
        / "exact_results.json",
        {
            "proven_optimal": True,
            "routes": [
                [0, 1, 2, 0],
            ],
            "route_loads": [6],
            "vehicle_count": 1,
            "total_distance": 100.0,
        },
    )

    result: dict[str, object] = {
        "status": status,
        "provider": "gemini",
        "model": "gemini-test",
        "encoding": "numeric",
        "optimality_gap_percent": 3.0,
        "error": None,
    }

    if status == "completed":
        result["model_response"] = {
            "elapsed_seconds": 2.5,
            "usage": {
                "prompt_token_count": 10,
                "output_token_count": 5,
                "thoughts_token_count": 20,
                "total_token_count": 35,
            },
        }
        result["validation"] = {
            "valid": True,
            "route_count": 1,
            "routes": [
                {
                    "route_index": 1,
                    "route": [0, 1, 2, 0],
                    "load": 6,
                    "capacity_excess": 0,
                    "distance": 103.0,
                    "valid": True,
                }
            ],
            "missing_customer_ids": [],
            "duplicated_customer_ids": [],
            "unknown_node_ids": [],
            "fleet_limit_exceeded": False,
            "total_capacity_excess": 0,
            "total_distance": 103.0,
        }
    else:
        result["model_response"] = None
        result["validation"] = None
        result["error"] = {
            "message": "quota exceeded",
        }

    write_json(
        run_dir
        / "providers"
        / "gemini"
        / "gemini-test"
        / "numeric"
        / "single_call_results.json",
        result,
    )
    return run_dir


def test_render_table_rejects_mismatched_row() -> None:
    with pytest.raises(ValueError):
        render_table(
            ["A", "B"],
            [["only-one"]],
        )


def test_normalize_run_id_rejects_path_traversal() -> None:
    with pytest.raises(ValueError):
        _normalize_run_id("../other-run")


def test_build_report_contains_result_and_routes(
    tmp_path: Path,
) -> None:
    run_dir = create_run(tmp_path)

    report = build_report(run_dir)

    assert "Görsel CVRP deney raporu" in report
    assert "capacity_demo_10" in report
    assert "Exact DP" in report
    assert "gemini-test" in report
    assert "numeric" in report
    assert "103.0000" in report
    assert "3.0000" in report
    assert "0 -> 1 -> 2 -> 0" in report
    assert "12.0000" in report
    assert "eksik=-" in report


def test_build_report_lists_failed_record(
    tmp_path: Path,
) -> None:
    run_dir = create_run(
        tmp_path,
        status="request_failed",
    )

    report = build_report(run_dir)

    assert "Tamamlanamayan kayıtlar" in report
    assert "request_failed" in report
    assert "quota exceeded" in report


def test_main_writes_terminal_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = create_run(tmp_path)

    main(
        [
            "--run-id",
            "report_run",
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
    assert (
        run_dir
        / "analysis"
        / "images"
        / "baseline_exact_routes.png"
    ).is_file()
    assert (
        run_dir
        / "analysis"
        / "images"
        / "gemini_gemini-test_numeric_routes.png"
    ).is_file()
    assert "gemini-test" in report_path.read_text(
        encoding="utf-8"
    )
    assert "Görsel çıktılar" in (
        report_path.read_text(encoding="utf-8")
    )
    assert "Rapor dosyası:" in capsys.readouterr().out
