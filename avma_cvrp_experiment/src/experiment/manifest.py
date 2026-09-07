"""Run manifest and provenance."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..prompts import PromptSet
from ..schemas import ProblemInstance

ARCHITECTURE_VERSION = "avma-cvrp-v1"
INFORMATION_POLICY = "visual_only_llm_passive_numeric_observer"
MULTI_PROVIDER_LAYOUT = "compact_v3"

def _git_sha(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def _base_manifest(*, config: ExperimentConfig, problem: ProblemInstance, prompts: PromptSet, project_root: Path, scale_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "information_policy": INFORMATION_POLICY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_sha(project_root),
        "config": config.raw,
        "config_sha256": config.sha256,
        "run_parameters": {"seed": config.seed, "protocol_name": config.name, "run_label": config.run_label},
        "prompt_set": prompts.version,
        "prompt_sha256": prompts.hashes(),
        "problem": problem.to_public_metadata(),
        "render_policy": {
            "demand_encoding_mode": config.demand_encoding.mode,
            "bar_layout": config.demand_encoding.bar_layout if config.demand_encoding.mode == "bar" else None,
            "fixed_canvas": config.render.fixed_canvas,
            "dpi": config.render.dpi,
            **(scale_policy or {}),
        },
        "llm_input_policy": {
            "allowed": ["problem images", "route images", "candidate images", "visible node labels", "visible depot marker", "agent task instructions"],
            "forbidden": ["coordinates", "distance matrix", "numeric edge lengths", "numeric route length", "capacity constraints", "gap", "known optimum", "optimal route", "GBest", "textual current routes as model input", "missing-node list", "validation reason"],
        },
    }

def build_shared_manifest(
    *,
    config: ExperimentConfig,
    problem: ProblemInstance,
    prompts: PromptSet,
    project_root: Path,
    scale_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = _base_manifest(
        config=config,
        problem=problem,
        prompts=prompts,
        project_root=project_root,
        scale_policy=scale_policy,
    )
    manifest["layout_version"] = MULTI_PROVIDER_LAYOUT
    manifest["prompts"] = {
        "common_policy": prompts.common,
        **{
            role: prompts.role(role)
            for role in (
                "initializer",
                "critic",
                "scorer",
                "repair",
                "hybrid",
                "diversity",
            )
        },
    }
    manifest["provider_policy"] = {
        "candidate_strategy": config.provider.candidate_strategy,
        "timeout_seconds": config.provider.timeout_seconds,
        "request_retries": config.provider.request_retries,
        "media_resolution": config.provider.media_resolution,
    }
    return manifest


def build_manifest(*, config: ExperimentConfig, problem: ProblemInstance, prompts: PromptSet, project_root: Path) -> dict[str, Any]:
    manifest = _base_manifest(config=config, problem=problem, prompts=prompts, project_root=project_root)
    manifest["layout_version"] = MULTI_PROVIDER_LAYOUT
    manifest["provider"] = {
        "name": config.provider.name,
        "model": config.provider.model,
        "candidate_strategy": config.provider.candidate_strategy,
        "media_resolution": config.provider.media_resolution,
    }
    return manifest

def assert_shared_manifest_compatible(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    checks = {
        "architecture_version": (existing.get("architecture_version"), expected.get("architecture_version")),
        "information_policy": (existing.get("information_policy"), expected.get("information_policy")),
        "git_commit_sha": (existing.get("git_commit_sha"), expected.get("git_commit_sha")),
        "config_sha256": (existing.get("config_sha256"), expected.get("config_sha256")),
        "prompt_set": (existing.get("prompt_set"), expected.get("prompt_set")),
        "prompt_sha256": (existing.get("prompt_sha256"), expected.get("prompt_sha256")),
        "problem": (existing.get("problem"), expected.get("problem")),
        "run_parameters": (existing.get("run_parameters"), expected.get("run_parameters")),
        "render_policy": (existing.get("render_policy"), expected.get("render_policy")),
        "provider_policy": (existing.get("provider_policy"), expected.get("provider_policy")),
    }
    mismatches = [name for name, (left, right) in checks.items() if left != right]
    if mismatches:
        raise ValueError("Aynı run-id altında deney koşulları değiştirilemez. Uyumsuz alanlar: " + ", ".join(mismatches))

def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
