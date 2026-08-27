from __future__ import annotations

import json
from pathlib import Path

from run_analysis import build_analysis, resolve_analysis_run_dir, write_analysis


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _evaluation(distance: float, gap: float) -> dict:
    return {
        "validation": {"valid": True},
        "distance": distance,
        "gap_percent": gap,
        "crossings": 0,
    }


def test_analysis_writes_only_canonical_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "problem": {
                "name": "eil51",
                "dimension": 51,
                "edge_weight_type": "EUC_2D",
                "reference_optimum": 426.0,
            },
            "provider": {"name": "gemini", "model": "gemini-3.6-flash"},
            "config": {"experiment": {"name": "smoke", "iterations": 1}},
        },
    )
    _write_json(
        run_dir / "initializer" / "initializer_result.json",
        {"evaluation": _evaluation(490.0, 15.0235)},
    )
    _write_json(run_dir / "summary.json", {"completed_iterations": 1})
    _write_json(
        run_dir / "iterations" / "iteration_001" / "iteration_result.json",
        {
            "iteration": 1,
            "critic_candidates": [
                {"candidate_id": 1, "evaluation": _evaluation(476.0, 11.7371)},
                {"candidate_id": 2, "evaluation": _evaluation(473.0, 11.0329)},
                {"candidate_id": 3, "evaluation": _evaluation(480.0, 12.6761)},
            ],
            "selected_before_repair": {
                "candidate_id": 1,
                "evaluation": _evaluation(476.0, 11.7371),
            },
            "working_evaluation": _evaluation(476.0, 11.7371),
            "structural_stagnation": {
                "stagnated": False,
                "mean_consecutive_similarity": 0.5,
            },
            "escape": None,
            "observer_only": {
                "selected_best_distance": 476.0,
                "observed_oracle_best_distance": 473.0,
            },
        },
    )
    iteration_dir = run_dir / "iterations" / "iteration_001"
    _write_json(
        iteration_dir / "candidates" / "candidate_01" / "critic_call.json",
        {
            "latency_seconds": 2.5,
            "usage": {"total_token_count": 900},
            "raw_metadata": {"candidate_index": 1, "native_candidate_count": 3},
        },
    )
    _write_json(
        iteration_dir / "candidates" / "candidate_02" / "critic_call.json",
        {
            "latency_seconds": None,
            "usage": {},
            "raw_metadata": {"candidate_index": 2, "native_candidate_count": 3},
        },
    )
    _write_json(
        iteration_dir / "candidates" / "candidate_03" / "critic_call.json",
        {
            "latency_seconds": None,
            "usage": {},
            "raw_metadata": {"candidate_index": 3, "native_candidate_count": 3},
        },
    )
    _write_json(
        iteration_dir / "scorer_call.json",
        {
            "latency_seconds": 1.25,
            "usage": {"total_token_count": 100},
            "raw_metadata": {},
        },
    )

    legacy_dir = run_dir / "analysis"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "analysis_summary.json").write_text("{}", encoding="utf-8")
    (legacy_dir / "observed_oracle_best.png").write_bytes(b"legacy")

    summary, rows, report = build_analysis(run_dir)
    outputs = write_analysis(run_dir, summary, rows, report)

    assert summary["performance"]["selected_best_distance"] == 476.0
    assert summary["performance"]["observed_oracle_best_distance"] == 473.0
    assert summary["scorer"]["oracle_selection_count"] == 0
    assert summary["scorer"]["mean_selection_regret"] == 3.0
    assert summary["scorer"]["max_selection_regret"] == 3.0
    assert summary["critic"]["total_candidate_count"] == 3
    assert summary["critic"]["valid_candidate_count"] == 3
    assert summary["repair"]["total_attempt_count"] == 0
    assert summary["adaptive_search"]["valid_hybrid_two_opt_count"] == 0
    assert rows[0]["iteration_oracle_distance"] == 473.0
    assert rows[0]["selection_regret"] == 3.0
    assert rows[0]["api_calls"] == 2
    assert rows[0]["total_tokens"] == 1000
    assert rows[0]["active_seconds"] == 3.75
    assert rows[0]["provider_errors"] == 0
    assert "AVMA-TSP ANALİZ RAPORU" in report
    assert "GENEL SONUÇ" not in report
    assert "SONUÇ" not in report
    assert "Critic Valid" in report
    assert "│ Critic Best │" not in report
    assert "Critic Best seçimi" in report
    assert "AGENT ÖZETİ" not in report
    assert "Initializer fallback" not in report
    headings = [
        "RUN BİLGİLERİ",
        "INITIALIZER",
        "İTERASYONLAR",
        "REPAIR",
        "CRITIC",
        "SCORER",
        "HYBRID & RESTART",
        "RUN EXECUTION",
    ]
    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "İterasyon içi aktivasyon = 0" in report
    assert "RUN EXECUTION" in report
    assert "Sistem GBest" in report
    assert "Oracle GBest" in report
    assert "Olay" not in report
    assert "API" in report
    assert "Token" in report
    assert "Aktif sn" in report
    assert "Hata" in report
    assert "Analiz tamamlandı." not in report
    assert all(path.exists() for path in outputs.values())
    assert sorted(path.name for path in legacy_dir.iterdir()) == [
        "analysis_report.txt",
        "iterations.csv",
        "selected_vs_oracle.png",
        "summary.json",
    ]


def test_analysis_resolves_legacy_and_shared_multi_provider_runs(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    _write_json(legacy / "run_manifest.json", {"provider": {"name": "gemini", "model": "old"}})
    (legacy / "iterations").mkdir(parents=True)
    assert resolve_analysis_run_dir(legacy) == legacy

    shared = tmp_path / "260827-eil_51_p10"
    gemini = shared / "providers" / "gemini" / "gemini-3.6-flash"
    _write_json(gemini / "run_manifest.json", {"provider": {"name": "gemini", "model": "gemini-3.6-flash"}})
    (gemini / "summary.json").write_text("{}", encoding="utf-8")
    assert resolve_analysis_run_dir(shared) == gemini
    assert resolve_analysis_run_dir(
        shared, provider="gemini", model="gemini-3.6-flash"
    ) == gemini

    groq = shared / "providers" / "groq" / "vision-model"
    _write_json(groq / "run_manifest.json", {"provider": {"name": "groq", "model": "vision-model"}})
    (groq / "summary.json").write_text("{}", encoding="utf-8")
    try:
        resolve_analysis_run_dir(shared)
    except SystemExit as exc:
        assert "birden fazla model" in str(exc)
    else:
        raise AssertionError("multi-model shared run selector olmadan analiz edilmemeliydi")


def test_repair_section_excludes_initializer_repair_and_counts_iteration_repair(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "problem": {
                "name": "eil51",
                "dimension": 51,
                "edge_weight_type": "EUC_2D",
                "reference_optimum": 426.0,
            },
            "provider": {"name": "gemini", "model": "gemini-3.6-flash"},
            "config": {"experiment": {"name": "smoke", "iterations": 1}},
        },
    )
    _write_json(
        run_dir / "initializer" / "initializer_result.json",
        {
            "evaluation": {"validation": {"valid": False}, "distance": None},
            "repair": [
                {
                    "attempt": 1,
                    "evaluation": {"validation": {"valid": False}, "distance": None},
                }
            ],
            "restart": {
                "attempts": [
                    {
                        "route": [1, 2, 3, 1],
                        "evaluation": _evaluation(490.0, 15.0235),
                    }
                ]
            },
        },
    )
    _write_json(
        run_dir / "initializer" / "repair" / "attempt_01" / "repair_result.json",
        {"evaluation": {"validation": {"valid": False}, "distance": None}},
    )
    _write_json(run_dir / "summary.json", {"completed_iterations": 1})
    _write_json(
        run_dir / "iterations" / "iteration_001" / "iteration_result.json",
        {
            "iteration": 1,
            "critic_candidates": [
                {"candidate_id": 1, "evaluation": _evaluation(476.0, 11.7371)},
            ],
            "selected_before_repair": {
                "candidate_id": 1,
                "evaluation": {"validation": {"valid": False}, "distance": None},
            },
            "repair": [
                {
                    "attempt": 1,
                    "evaluation": _evaluation(476.0, 11.7371),
                }
            ],
            "working_evaluation": _evaluation(476.0, 11.7371),
            "structural_stagnation": {
                "stagnated": False,
                "mean_consecutive_similarity": 0.0,
            },
            "escape": None,
            "observer_only": {
                "selected_best_distance": 476.0,
                "observed_oracle_best_distance": 476.0,
            },
        },
    )
    _write_json(
        run_dir / "iterations" / "iteration_001" / "selected_repair" / "attempt_01" / "repair_result.json",
        {"evaluation": _evaluation(476.0, 11.7371)},
    )

    summary, _, report = build_analysis(run_dir)

    assert summary["initializer"]["repair_attempt_count"] == 1
    assert summary["repair"]["activation_count"] == 1
    assert summary["repair"]["total_attempt_count"] == 1
    assert summary["repair"]["successful_repair_count"] == 1
    assert summary["repair"]["failed_repair_count"] == 0
    assert summary["repair"]["iterations"] == [
        {
            "iteration": 1,
            "activation_count": 1,
            "attempt_count": 1,
            "successful_count": 1,
            "failed_count": 0,
        }
    ]
    assert "İterasyon 1" in report
    assert "Initializer fallback" not in report
