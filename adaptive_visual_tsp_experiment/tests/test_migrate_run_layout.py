import json
from pathlib import Path

from migrate_run_layout import migrate
from run_analysis import build_analysis
from src.experiment.layout import provider_model_dir


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _eval(distance: float) -> dict:
    return {"validation": {"valid": True}, "distance": distance, "gap_percent": 0.0, "crossings": 0}


def test_v2_shared_run_migrates_without_api_and_keeps_backup(tmp_path: Path) -> None:
    run = tmp_path / "260827-eil_51_p10"
    shared_manifest = {
        "layout_version": "multi_provider_v2",
        "prompt_set": "v1",
        "config_sha256": "cfg",
        "run_parameters": {"seed": 42},
        "problem": {"name": "eil51", "source_sha256": "inst", "reference_optimum": 426.0},
        "config": {"experiment": {"iterations": 1}},
    }
    _write_json(run / "run_manifest.json", shared_manifest)
    (run / "inputs").mkdir(parents=True)
    (run / "inputs" / "problem_model.png").write_bytes(b"problem")

    model = run / "providers" / "gemini" / "gemini-3.6-flash"
    model_manifest = dict(shared_manifest)
    model_manifest["provider"] = {"name": "gemini", "model": "gemini-3.6-flash"}
    _write_json(model / "run_manifest.json", model_manifest)
    _write_json(
        model / "initializer" / "initializer_candidate_result.json",
        {"route": [1, 2, 3, 1], "evaluation": _eval(4.0)},
    )
    (model / "initializer" / "initial_route_model.png").write_bytes(b"init")
    _write_json(
        model / "initializer" / "initializer_result.json",
        {"route": [1, 2, 3, 1], "evaluation": _eval(4.0), "resumed": False},
    )
    _write_json(
        model / "iterations" / "iteration_001" / "candidates" / "candidate_01" / "candidate_result.json",
        {"candidate_id": 1, "route": [1, 2, 3, 1], "evaluation": _eval(4.0)},
    )
    (model / "iterations" / "iteration_001" / "candidates" / "candidate_01" / "route_model.png").write_bytes(b"c1")
    _write_json(
        model / "iterations" / "iteration_001" / "iteration_result.json",
        {
            "iteration": 1,
            "critic_candidates": [{"candidate_id": 1, "evaluation": _eval(4.0)}],
            "selected_before_repair": {"candidate_id": 1, "evaluation": _eval(4.0)},
            "working_route_after_iteration": [1, 2, 3, 1],
            "working_evaluation": _eval(4.0),
            "structural_stagnation": {"stagnated": False},
            "escape": None,
            "observer_only": {"selected_best_distance": 4.0, "observed_oracle_best_distance": 4.0},
        },
    )
    _write_json(
        model / "checkpoint.json",
        {
            "completed_iteration": 1,
            "working_route": [1, 2, 3, 1],
            "structural_history": [[1, 2, 3, 1]],
            "hybrid_used_since_restart": False,
            "restart_count": 0,
            "observed_oracle_best_distance": 4.0,
            "observed_oracle_best_route": [1, 2, 3, 1],
            "selected_best_distance": 4.0,
            "selected_best_route": [1, 2, 3, 1],
            "config_sha256": "cfg",
            "instance_sha256": "inst",
        },
    )

    migrated, backup = migrate(run)

    assert migrated == run
    assert backup is not None and backup.exists()
    assert (run / "run.json").exists()
    assert (run / "problem.png").read_bytes() == b"problem"
    compact_model = provider_model_dir(run, "gemini", "gemini-3.6-flash")
    assert (compact_model / "state.json").exists()
    assert (compact_model / "trace.jsonl").exists()
    assert (compact_model / "routes" / "initializer" / "candidate.png").exists()
    assert (compact_model / "routes" / "iteration_001" / "C1.png").exists()
    assert not (compact_model / "iterations").exists()
    assert not (compact_model / "initializer").exists()

    summary, rows, report = build_analysis(compact_model)
    assert rows[0]["selected_distance"] == 4.0
    assert summary["run"]["provider"] == "gemini"
    assert "│ Critic Best │" not in report
    assert "RUN EXECUTION" in report


def test_legacy_single_model_run_moves_under_provider_branch(tmp_path: Path) -> None:
    run = tmp_path / "260827-eil_51_s3"
    manifest = {
        "prompt_set": "v1",
        "config_sha256": "cfg",
        "run_parameters": {"seed": 42},
        "problem": {"name": "eil51", "source_sha256": "inst", "reference_optimum": 426.0},
        "config": {"experiment": {"iterations": 1}},
        "provider": {"name": "gemini", "model": "gemini-3.6-flash"},
    }
    _write_json(run / "run_manifest.json", manifest)
    (run / "inputs").mkdir(parents=True)
    (run / "inputs" / "problem_model.png").write_bytes(b"problem")
    _write_json(
        run / "initializer" / "initializer_candidate_result.json",
        {"route": [1, 2, 3, 1], "evaluation": _eval(4.0)},
    )
    (run / "initializer" / "initial_route_model.png").write_bytes(b"init")
    _write_json(
        run / "initializer" / "initializer_result.json",
        {"route": [1, 2, 3, 1], "evaluation": _eval(4.0)},
    )
    _write_json(
        run / "iterations" / "iteration_001" / "iteration_result.json",
        {
            "iteration": 1,
            "critic_candidates": [],
            "selected_before_repair": {},
            "working_route_after_iteration": [1, 2, 3, 1],
            "working_evaluation": _eval(4.0),
            "structural_stagnation": {"stagnated": False},
            "escape": None,
            "observer_only": {"selected_best_distance": 4.0, "observed_oracle_best_distance": 4.0},
        },
    )
    _write_json(run / "summary.json", {"completed_iterations": 1})

    migrate(run)

    model = provider_model_dir(run, "gemini", "gemini-3.6-flash")
    assert (run / "run.json").exists()
    assert (model / "state.json").exists()
    assert (model / "trace.jsonl").exists()
    assert not (run / "iterations").exists()
    assert not (run / "initializer").exists()
