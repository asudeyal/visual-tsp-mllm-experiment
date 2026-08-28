import json
from pathlib import Path

from src.experiment.compact import TraceStore
from src.experiment.compact_analysis import build_compact_analysis


def _iteration_result(iteration: int, distance: float) -> dict:
    evaluation = {"validation": {"valid": True}, "distance": distance}
    return {
        "iteration": iteration,
        "critic_candidates": [{"candidate_id": 1, "evaluation": evaluation}],
        "selected_before_repair": {"candidate_id": 1, "evaluation": evaluation},
        "working_evaluation": evaluation,
        "observer_only": {
            "selected_best_distance": distance,
            "observed_oracle_best_distance": distance,
        },
        "structural_stagnation": {"stagnated": False},
        "escape": None,
    }


def _compact_run(tmp_path: Path) -> tuple[Path, TraceStore]:
    root = tmp_path / "run"
    model_dir = root / "providers" / "gemini" / "fake-model"
    model_dir.mkdir(parents=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "problem": {
                    "name": "square4",
                    "dimension": 4,
                    "edge_weight_type": "EUC_2D",
                    "reference_optimum": 4.0,
                },
                "config": {"experiment": {"name": "pilot10_v2", "iterations": 10}},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "state.json").write_text(
        json.dumps({"provider": "gemini", "model": "fake-model", "status": "completed"}),
        encoding="utf-8",
    )
    trace = TraceStore(model_dir / "trace.jsonl")
    trace.append(
        {
            "event": "initializer_candidate",
            "evaluation": {"validation": {"valid": True}, "distance": 10.0},
        }
    )
    trace.append(
        {
            "event": "initializer_result",
            "evaluation": {"validation": {"valid": True}, "distance": 10.0},
        }
    )
    return model_dir, trace


def test_unresolved_provider_error_cuts_ghost_iterations(tmp_path: Path) -> None:
    model_dir, trace = _compact_run(tmp_path)
    trace.append({"event": "iteration_result", "iteration": 1, "result": _iteration_result(1, 9.0)})
    trace.append({"event": "iteration_result", "iteration": 2, "result": _iteration_result(2, 8.0)})
    trace.append(
        {
            "event": "provider_error",
            "iteration": 3,
            "candidate": 2,
            "phase": "critic_candidate_02",
            "status_code": 429,
            "error_type": "ClientError",
            "message": "quota exhausted",
        }
    )
    # Historical buggy controllers could still persist bookkeeping-only
    # iteration_result events after provider failure. They must not count as
    # completed search iterations.
    for iteration in range(3, 11):
        ghost = _iteration_result(iteration, 8.0)
        ghost["critic_candidates"] = []
        ghost["selected_before_repair"] = {}
        trace.append(
            {
                "event": "iteration_result",
                "iteration": iteration,
                "result": ghost,
            }
        )

    summary, rows, report = build_compact_analysis(model_dir)

    assert summary["run"]["completed_iterations"] == 2
    assert summary["run"]["partial_iteration"] == 3
    assert summary["termination"]["status"] == "partial"
    assert [row["iteration"] for row in rows] == [1, 2]
    assert summary["iteration_cost"][-1]["label"] == "3*"
    assert "2/10" in report
    assert "10/10" not in report


def test_provider_error_resolved_by_resume_does_not_cut_run(tmp_path: Path) -> None:
    model_dir, trace = _compact_run(tmp_path)
    trace.append({"event": "iteration_result", "iteration": 1, "result": _iteration_result(1, 9.0)})
    trace.append(
        {
            "event": "provider_error",
            "iteration": 2,
            "candidate": 2,
            "phase": "critic_candidate_02",
            "status_code": 503,
            "error_type": "ServerError",
            "message": "temporarily unavailable",
        }
    )
    trace.append(
        {
            "event": "agent_call",
            "iteration": 2,
            "candidate": 2,
            "agent": "critic",
            "call": {"phase": "critic_candidate_02_output_01", "usage": {}, "raw_metadata": {}},
        }
    )
    trace.append({"event": "iteration_result", "iteration": 2, "result": _iteration_result(2, 8.0)})

    summary, rows, _ = build_compact_analysis(model_dir)

    assert summary["run"]["completed_iterations"] == 2
    assert summary["run"]["partial_iteration"] is None
    assert [row["iteration"] for row in rows] == [1, 2]


def test_provider_error_superseded_by_completed_iteration_does_not_cut_run(tmp_path: Path) -> None:
    model_dir, trace = _compact_run(tmp_path)
    trace.append({"event": "iteration_result", "iteration": 1, "result": _iteration_result(1, 9.0)})
    trace.append(
        {
            "event": "provider_error",
            "iteration": 2,
            "scope": "iteration_002.selected",
            "attempt": 1,
            "phase": "repair_attempt_01",
            "status_code": 503,
            "error_type": "ServerError",
            "message": "first repair attempt failed",
        }
    )
    # A later attempt in the same logical recovery chain succeeds. There is no
    # same-attempt agent_call, but the completed iteration is definitive proof
    # that this provider error was superseded.
    trace.append({"event": "iteration_result", "iteration": 2, "result": _iteration_result(2, 8.0)})

    summary, rows, _ = build_compact_analysis(model_dir)

    assert summary["run"]["completed_iterations"] == 2
    assert summary["run"]["partial_iteration"] is None
    assert [row["iteration"] for row in rows] == [1, 2]


def test_initializer_provider_error_superseded_by_initializer_result(tmp_path: Path) -> None:
    root = tmp_path / "run"
    model_dir = root / "providers" / "gemini" / "fake-model"
    model_dir.mkdir(parents=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "problem": {
                    "name": "square4",
                    "dimension": 4,
                    "edge_weight_type": "EUC_2D",
                    "reference_optimum": 4.0,
                },
                "config": {"experiment": {"name": "pilot10_v2", "iterations": 10}},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "state.json").write_text(
        json.dumps({"provider": "gemini", "model": "fake-model", "status": "running"}),
        encoding="utf-8",
    )
    trace = TraceStore(model_dir / "trace.jsonl")
    trace.append(
        {
            "event": "initializer_candidate",
            "evaluation": {"validation": {"valid": False}, "distance": None},
        }
    )
    trace.append(
        {
            "event": "provider_error",
            "scope": "initializer",
            "attempt": 1,
            "phase": "repair_attempt_01",
            "status_code": 503,
            "error_type": "ServerError",
            "message": "temporary initializer repair failure",
        }
    )
    trace.append(
        {
            "event": "repair_result",
            "scope": "initializer",
            "attempt": 2,
            "evaluation": {"validation": {"valid": True}, "distance": 10.0},
        }
    )
    trace.append(
        {
            "event": "initializer_result",
            "evaluation": {"validation": {"valid": True}, "distance": 10.0},
        }
    )
    trace.append({"event": "iteration_result", "iteration": 1, "result": _iteration_result(1, 9.0)})

    summary, rows, _ = build_compact_analysis(model_dir)

    assert summary["run"]["completed_iterations"] == 1
    assert summary["initializer"]["repair_attempt_count"] == 1
    assert summary["initializer"]["accepted_source"] == "repair"
    assert [row["iteration"] for row in rows] == [1]
