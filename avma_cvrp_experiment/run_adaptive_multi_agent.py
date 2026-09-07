from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config
from src.controller.orchestrator import AdaptiveVisualCVRPOrchestrator
from src.experiment.compact import COMPACT_LAYOUT, read_state, update_state
from src.experiment.layout import automatic_run_id, is_legacy_run, provider_model_dir
from src.experiment.manifest import (
    assert_shared_manifest_compatible,
    build_shared_manifest,
    read_manifest,
    write_manifest,
)
from src.problem import load_cvrplib
from src.prompts import PromptSet
from src.scale_policy import (
    DEFAULT_SCALE_POLICY_NAME,
    DEFAULT_SCALE_POLICY_PATH,
    load_scale_policy,
    scale_manifest_context,
    scale_render_config,
)
from src.providers import create_provider, supported_providers
from src.rendering import render_problem


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AVMA-CVRP visual-only multi-agent search"
    )
    parser.add_argument("--instance", required=True, help="CVRP .vrp file")
    parser.add_argument("--config", default="configs/main_8method/size_collision.yaml")
    parser.add_argument(
        "--prompt-set",
        default=None,
        help=(
            "Optional prompt-set override, e.g. cvrp_capacity_v1, "
            "cvrp_capacity_v2, or cvrp_capacity_v3. "
            "If omitted, experiment.prompt_set from the YAML is used."
        ),
    )
    parser.add_argument(
        "--scale-policy",
        default=DEFAULT_SCALE_POLICY_PATH,
        help=(
            "Versioned instance workspace-scale policy JSON. "
            "Default: data/cvrplib/scale_policies/benchmark_scale_v1.json"
        ),
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=supported_providers(),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-optimum", type=float, default=None)
    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=None,
        help=(
            "Optional hard vehicle limit for validation and visual vehicle icons. "
            "This value is never shown numerically to the model."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Shared multi-provider run id. Omit on first run for an automatic "
            "YYMMDD-instance-protocol id."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Explicit run root; retained for historical single-model runs. "
            "Prefer --run-id."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="No API calls; validate config, instance, and model-facing rendering",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Minimum delay between provider request starts; runtime-only and "
            "not part of the visual protocol."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help=(
            "Per-request provider timeout override; runtime-only and not part "
            "of the visual protocol."
        ),
    )
    return parser.parse_args()


def _resolve_run_root(
    args: argparse.Namespace,
    config,
    problem,
    scale_policy_name: str,
) -> Path:
    if args.run_dir and args.run_id:
        raise SystemExit("--run-dir ve --run-id birlikte kullanılamaz")

    if args.run_dir:
        return _resolve_from_root(args.run_dir)

    if args.run_id:
        return _resolve_from_root(config.output_dir) / args.run_id

    if args.resume:
        raise SystemExit("--resume için --run-id veya --run-dir verilmelidir")

    effective_prompt_set = args.prompt_set or config.prompt_set
    prompt_tag = effective_prompt_set.removeprefix("cvrp_capacity_")
    run_label = f"{config.run_label}-{prompt_tag}"

    if scale_policy_name != DEFAULT_SCALE_POLICY_NAME:
        scale_tag = scale_policy_name.removeprefix("benchmark_scale_")
        run_label = f"{run_label}-{scale_tag}"

    return _resolve_from_root(config.output_dir) / automatic_run_id(
        problem.name,
        run_label,
    )



def _prepare_shared_run(
    run_root: Path,
    *,
    config,
    problem,
    prompts,
    scale_context: dict[str, object],
) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)

    compact_manifest_path = run_root / "run.json"
    old_manifest_path = run_root / "run_manifest.json"

    if old_manifest_path.exists() and not compact_manifest_path.exists():
        raise ValueError(
            "Bu run eski klasör düzeninde. Önce migrate_run_layout.py ile "
            "compact_v3 düzenine taşıyın."
        )

    expected = build_shared_manifest(
        config=config,
        problem=problem,
        prompts=prompts,
        project_root=PROJECT_ROOT,
        scale_policy=scale_context,
    )
    expected["supported_providers"] = list(supported_providers())

    if compact_manifest_path.exists():
        existing = read_manifest(compact_manifest_path)

        if existing.get("layout_version") != COMPACT_LAYOUT:
            raise ValueError(
                "Bu run compact_v3 düzeninde değil; migration gereklidir"
            )

        assert_shared_manifest_compatible(existing, expected)
    else:
        write_manifest(compact_manifest_path, expected)

    problem_image = run_root / "problem.png"

    if not problem_image.exists():
        render_problem(
            problem,
            problem_image,
            config.render,
            demand_encoding=config.demand_encoding,
        )

    return problem_image



def _prepare_model_state(
    model_dir: Path,
    *,
    config,
    problem,
    run_root: Path,
    resume: bool,
) -> None:
    state_path = model_dir / "state.json"
    trace_path = model_dir / "trace.jsonl"

    has_state = (
        state_path.exists()
        or trace_path.exists()
        or (model_dir / "routes").exists()
    )

    if has_state and not resume:
        raise ValueError(
            "Bu provider/model aynı run içinde zaten başlatılmış. "
            "Devam etmek için --resume kullanın veya farklı --run-id seçin."
        )

    if resume and not has_state:
        raise ValueError(
            "Resume reddedildi: bu run içinde provider/model state bulunamadı"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    state = read_state(state_path)

    if state:
        if (
            state.get("provider") != config.provider.name
            or state.get("model") != config.provider.model
        ):
            raise ValueError("state.json provider/model ile uyuşmuyor")

        if state.get("config_sha256") != config.sha256:
            raise ValueError("Resume reddedildi: protocol config değişmiş")

        if state.get("instance_sha256") not in {
            None,
            problem.source_sha256,
        }:
            raise ValueError("Resume reddedildi: CVRP instance değişmiş")

        if state.get("max_vehicles") != problem.max_vehicles:
            raise ValueError("Resume reddedildi: max_vehicles değişmiş")

        if state.get("demand_encoding_mode") != config.demand_encoding.mode:
            raise ValueError(
                "Resume reddedildi: görsel talep kodlama yöntemi değişmiş"
            )
    else:
        update_state(
            state_path,
            provider=config.provider.name,
            model=config.provider.model,
            shared_run_id=run_root.name,
            status="created",
            config_sha256=config.sha256,
            seed=config.seed,
            instance_sha256=problem.source_sha256,
            max_vehicles=problem.max_vehicles,
            demand_encoding_mode=config.demand_encoding.mode,
            checkpoint=None,
            working_image=None,
            summary=None,
            last_error=None,
        )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    if args.request_delay_seconds < 0:
        raise SystemExit("--request-delay-seconds negatif olamaz")

    if args.timeout_seconds is not None and args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds en az 1 olmalıdır")

    if args.max_vehicles is not None and args.max_vehicles < 1:
        raise SystemExit("--max-vehicles en az 1 olmalıdır")

    config = load_config(
        _resolve_from_root(args.config),
        provider_name=args.provider,
        model=args.model,
        seed=args.seed,
    )

    problem = load_cvrplib(
        _resolve_from_root(args.instance),
        reference_optimum=args.reference_optimum,
        strict_euc_2d=config.strict_euc_2d,
        max_vehicles=args.max_vehicles,
    )

    effective_prompt_set = args.prompt_set or config.prompt_set
    prompts = PromptSet(PROJECT_ROOT / "prompts", effective_prompt_set)
    scale_policy_path = _resolve_from_root(args.scale_policy)
    scale_policy = load_scale_policy(scale_policy_path)
    instance_key = _resolve_from_root(args.instance).name
    workspace_scale = scale_policy.scale_for_instance(instance_key)
    effective_render = scale_render_config(
        config.render,
        workspace_scale,
    )
    config = replace(
        config,
        render=effective_render,
    )
    scale_context = scale_manifest_context(
        policy=scale_policy,
        instance_name=instance_key,
        workspace_scale=workspace_scale,
        render=config.render,
        project_root=PROJECT_ROOT,
    )

    run_root = _resolve_run_root(
        args,
        config,
        problem,
        scale_policy.name,
    )

    if run_root.exists() and is_legacy_run(run_root):
        if not args.run_dir:
            raise ValueError(
                "Otomatik run-id legacy bir run ile çakıştı; "
                "--run-id ile yeni ad verin"
            )

        raise ValueError(
            "Legacy run doğrudan çalıştırılmaz. Önce migrate_run_layout.py ile "
            "compact_v3'e taşıyın."
        )

    problem_image = _prepare_shared_run(
        run_root,
        config=config,
        problem=problem,
        prompts=prompts,
        scale_context=scale_context,
    )

    if args.validate_only:
        vehicle_limit_text = (
            "unlimited"
            if problem.max_vehicles is None
            else str(problem.max_vehicles)
        )

        print(
            "VALIDATION OK: "
            f"{problem.name} "
            f"({problem.dimension} nodes, capacity={problem.capacity}, "
            f"max_vehicles={vehicle_limit_text}, "
            f"encoding={config.demand_encoding.mode}, "
            f"workspace_scale={workspace_scale:.3f}x, "
            f"prompt_set={prompts.version}, "
            f"{problem.edge_weight_type})"
        )
        print(
            f"Scale policy: {scale_policy.name} "
            f"(sha256={scale_policy.sha256[:12]}..., "
            f"scale={workspace_scale:.3f}x)"
        )
        print(f"Shared run: {run_root}")
        print(f"Shared model-facing problem image: {problem_image}")
        return

    model_dir = provider_model_dir(
        run_root,
        config.provider.name,
        config.provider.model,
    )

    _prepare_model_state(
        model_dir,
        config=config,
        problem=problem,
        run_root=run_root,
        resume=args.resume,
    )

    update_state(
        model_dir / "state.json",
        instance_sha256=problem.source_sha256,
        max_vehicles=problem.max_vehicles,
        demand_encoding_mode=config.demand_encoding.mode,
        status="running",
        last_error=None,
    )

    provider = create_provider(config.provider)

    if args.timeout_seconds is not None:
        provider.timeout_seconds = args.timeout_seconds

    provider.configure_request_delay(args.request_delay_seconds)

    update_state(
        model_dir / "state.json",
        runtime_request_policy={
            "request_delay_seconds": args.request_delay_seconds,
            "timeout_seconds": provider.timeout_seconds,
            "retry_backoff": (
                "Retry-After or 30/60/120 seconds for transient HTTP errors"
            ),
        },
    )

    orchestrator = AdaptiveVisualCVRPOrchestrator(
        config=config,
        problem=problem,
        provider=provider,
        prompts=prompts,
        run_dir=model_dir,
        problem_image=problem_image,
    )

    summary = orchestrator.run(resume=args.resume)

    print("AVMA-CVRP tamamlandı")
    print(f"Run: {run_root}")
    print(f"Model artifacts: {model_dir}")
    print(f"Selected best distance: {summary['selected_best_distance']}")
    print(
        "Observed oracle best distance: "
        f"{summary['observed_oracle_best_distance']}"
    )


if __name__ == "__main__":
    main()