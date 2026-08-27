from pathlib import Path

import pytest

from src.config import load_config
from src.experiment.manifest import (
    assert_shared_manifest_compatible,
    build_shared_manifest,
)
from src.prompts import PromptSet
from src.schemas import ProblemInstance


def _problem() -> ProblemInstance:
    return ProblemInstance(
        name="eil51",
        dimension=3,
        node_ids=(1, 2, 3),
        coordinates={1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0)},
        depot=1,
        edge_weight_type="EUC_2D",
        source_path="data/tsplib/eil51.tsp",
        source_sha256="abc",
        reference_optimum=426.0,
    )


def _manifest(seed: int = 42):
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "pilot10_v1.yaml",
        provider_name="gemini",
        model="gemini-3.6-flash",
        seed=seed,
    )
    return build_shared_manifest(
        config=config,
        problem=_problem(),
        prompts=PromptSet(root / "prompts", "v1"),
        project_root=root.parent,
    )


def test_shared_manifest_ignores_provider_identity():
    root = Path(__file__).resolve().parents[1]
    gemini = load_config(
        root / "configs" / "pilot10_v1.yaml",
        provider_name="gemini",
        model="gemini-3.6-flash",
        seed=42,
    )
    groq = load_config(
        root / "configs" / "pilot10_v1.yaml",
        provider_name="groq",
        model="vision-model",
        seed=42,
    )
    left = build_shared_manifest(
        config=gemini,
        problem=_problem(),
        prompts=PromptSet(root / "prompts", "v1"),
        project_root=root.parent,
    )
    right = build_shared_manifest(
        config=groq,
        problem=_problem(),
        prompts=PromptSet(root / "prompts", "v1"),
        project_root=root.parent,
    )
    assert_shared_manifest_compatible(left, right)


def test_shared_manifest_rejects_seed_change():
    with pytest.raises(ValueError, match="run_parameters"):
        assert_shared_manifest_compatible(_manifest(42), _manifest(7))
