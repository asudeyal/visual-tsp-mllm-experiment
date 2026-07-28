from pathlib import Path

import pytest

from src.core import read_json, write_json
from src.output_migration import (
    OutputMigrationError,
    apply_plan,
    build_plan,
    undo_migration,
)


def _fixture(run_dir: Path) -> None:
    write_json(run_dir / "run_manifest.json", {"run_id": "run1"})
    write_json(
        run_dir / "zero_shot" / "zero_shot_results.json",
        {
            "model": {
                "provider": "google_gemini",
                "name": "gemini-2.5-flash",
            },
            "artifacts": {
                "route_image": "zero_shot/images/route.png",
            },
        },
    )
    image = run_dir / "zero_shot" / "images" / "route.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    write_json(
        run_dir
        / "model_comparisons"
        / "openrouter"
        / "model-a"
        / "zero_shot_results.json",
        {
            "model": {
                "provider": "openrouter",
                "alias": "model-a",
            },
            "artifacts": {
                "route_image": (
                    "model_comparisons/openrouter/model-a/"
                    "images/route.png"
                ),
            },
        },
    )
    openrouter_image = (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / "model-a"
        / "images"
        / "route.png"
    )
    openrouter_image.parent.mkdir(parents=True, exist_ok=True)
    openrouter_image.write_bytes(b"image")


def test_migration_moves_outputs_rewrites_paths_and_undoes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "output" / "runs" / "run1"
    _fixture(run_dir)

    plan = build_plan(run_dir)
    assert len(plan) == 3
    record = apply_plan(run_dir, plan)

    gemini_result = (
        run_dir
        / "providers"
        / "gemini"
        / "gemini-2.5-flash"
        / "zero_shot"
        / "zero_shot_results.json"
    )
    openrouter_result = (
        run_dir
        / "providers"
        / "openrouter"
        / "model-a"
        / "zero_shot"
        / "zero_shot_results.json"
    )
    assert gemini_result.exists()
    assert openrouter_result.exists()
    assert read_json(gemini_result)["artifacts"]["route_image"] == (
        "providers/gemini/gemini-2.5-flash/"
        "zero_shot/images/route.png"
    )
    assert read_json(openrouter_result)["artifacts"]["route_image"] == (
        "providers/openrouter/model-a/"
        "zero_shot/images/route.png"
    )
    assert record["status"] == "applied"

    undone = undo_migration(run_dir)

    assert undone["status"] == "undone"
    assert (
        run_dir / "zero_shot" / "zero_shot_results.json"
    ).exists()
    assert (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / "model-a"
        / "zero_shot_results.json"
    ).exists()


def test_migration_refuses_existing_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "output" / "runs" / "run1"
    _fixture(run_dir)
    target = (
        run_dir
        / "providers"
        / "gemini"
        / "gemini-2.5-flash"
        / "zero_shot"
    )
    target.mkdir(parents=True)

    with pytest.raises(OutputMigrationError, match="Hedef zaten"):
        build_plan(run_dir)


def test_migration_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(OutputMigrationError, match="manifest"):
        build_plan(tmp_path / "missing")
