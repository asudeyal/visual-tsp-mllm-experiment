import json
from pathlib import Path

import pytest

from organize_output import (
    apply_plan,
    build_json_updates,
    build_move_plan,
    validate_move_plan,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_organizer_moves_flat_and_run_images_and_updates_json(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()

    (output / "points.png").write_bytes(b"points")
    (output / "gemini_ma1_iteration_01_candidate_01.png").write_bytes(b"ma1")
    write_json(
        output / "baseline_results.json",
        {"image": f"{output}\\points.png"},
    )
    write_json(
        output / "gemini_multi_agent1_results.json",
        {
            "image": (
                f"{output}\\gemini_ma1_iteration_01_candidate_01.png"
            )
        },
    )

    ma2_dir = output / "runs" / "run_02" / "multi_agent2"
    ma2_dir.mkdir(parents=True)
    ma2_image = ma2_dir / "gemini_ma2_iteration_01.png"
    ma2_image.write_bytes(b"ma2")
    write_json(
        ma2_dir / "gemini_multi_agent2_results.json",
        {"image": str(ma2_image)},
    )

    moves, unknown = build_move_plan(
        output,
        legacy_run_id="seed42_initial_run",
    )
    validate_move_plan(moves)
    updates = build_json_updates(output, moves)
    apply_plan(moves, updates)

    initial = output / "runs" / "seed42_initial_run"
    baseline_json = initial / "baseline" / "baseline_results.json"
    ma1_json = initial / "multi_agent1" / "gemini_multi_agent1_results.json"
    assert not unknown
    assert (initial / "baseline" / "images" / "points.png").exists()
    assert (
        initial
        / "multi_agent1"
        / "images"
        / "iteration_01"
        / "candidate_01.png"
    ).exists()
    assert (ma2_dir / "images" / "iteration_01.png").exists()

    baseline_data = json.loads(baseline_json.read_text(encoding="utf-8"))
    ma1_data = json.loads(ma1_json.read_text(encoding="utf-8"))
    ma2_data = json.loads(
        (ma2_dir / "gemini_multi_agent2_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline_data["image"].replace("\\", "/").endswith(
        "seed42_initial_run/baseline/images/points.png"
    )
    assert ma1_data["image"].replace("\\", "/").endswith(
        "seed42_initial_run/multi_agent1/images/iteration_01/candidate_01.png"
    )
    assert ma2_data["image"].replace("\\", "/").endswith(
        "multi_agent2/images/iteration_01.png"
    )


def test_organizer_rejects_existing_destination(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "points.png").write_bytes(b"source")
    destination = (
        output
        / "runs"
        / "seed42_initial_run"
        / "baseline"
        / "images"
        / "points.png"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    moves, _ = build_move_plan(
        output,
        legacy_run_id="seed42_initial_run",
    )

    with pytest.raises(SystemExit, match="Hedef dosya zaten var"):
        validate_move_plan(moves)


def test_organizer_leaves_unknown_root_file_unmodified(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    unknown_file = output / "notes.txt"
    unknown_file.write_text("not", encoding="utf-8")

    moves, unknown = build_move_plan(
        output,
        legacy_run_id="seed42_initial_run",
    )

    assert moves == {}
    assert unknown == [unknown_file]
