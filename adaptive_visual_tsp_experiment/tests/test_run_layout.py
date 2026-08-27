from datetime import datetime
from pathlib import Path

from src.experiment.layout import (
    automatic_run_id,
    discover_model_runs,
    problem_alias,
    provider_model_dir,
)


def test_compact_run_id_matches_protocol_format():
    assert problem_alias("eil51") == "eil_51"
    assert automatic_run_id(
        "eil51",
        "p10",
        now=datetime(2026, 8, 27, 3, 15, 22),
    ) == "260827-eil_51_p10"


def test_provider_model_layout_and_discovery(tmp_path: Path):
    root = tmp_path / "260827-eil_51_p10"
    gemini = provider_model_dir(root, "gemini", "gemini-3.6-flash")
    groq = provider_model_dir(root, "groq", "qwen/qwen3.6-27b")
    for path in (gemini, groq):
        path.mkdir(parents=True)
        (path / "run_manifest.json").write_text("{}", encoding="utf-8")
        (path / "summary.json").write_text("{}", encoding="utf-8")

    assert gemini == root / "providers" / "gemini" / "gemini-3.6-flash"
    assert groq == root / "providers" / "groq" / "qwen-qwen3.6-27b"
    assert discover_model_runs(root) == sorted([gemini, groq])
