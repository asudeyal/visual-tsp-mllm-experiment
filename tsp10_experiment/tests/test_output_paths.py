from pathlib import Path

import pytest

from src.output_paths import build_experiment_paths, validate_run_id


def test_missing_run_id_uses_organized_default_run() -> None:
    paths = build_experiment_paths(Path("output"), None)

    expected_root = Path("output/runs/default")
    assert paths.run_root == expected_root
    assert paths.baseline == expected_root / "baseline"
    assert paths.zero_shot == expected_root / "zero_shot"
    assert paths.multi_agent1 == expected_root / "multi_agent1"
    assert paths.multi_agent2 == expected_root / "multi_agent2"


def test_run_id_creates_method_subdirectories() -> None:
    paths = build_experiment_paths(Path("output"), "seed42_timing_run_01")

    expected_root = Path("output/runs/seed42_timing_run_01")
    assert paths.run_root == expected_root
    assert paths.baseline == expected_root / "baseline"
    assert paths.zero_shot == expected_root / "zero_shot"
    assert paths.multi_agent1 == expected_root / "multi_agent1"
    assert paths.multi_agent2 == expected_root / "multi_agent2"


@pytest.mark.parametrize(
    "run_id",
    ["", "../outside", "folder/name", "has space", "_starts_with_symbol"],
)
def test_invalid_run_ids_are_rejected(run_id: str) -> None:
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_valid_run_ids_are_preserved() -> None:
    assert validate_run_id("seed42.timing-run_01") == "seed42.timing-run_01"
