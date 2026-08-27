"""API-free migration from historical AVMA run layouts to compact_v3.

The migration never invokes an LLM provider. It copies unique route images,
combines textual artifacts into trace.jsonl, and stores checkpoint/final state in
state.json. The original run is kept as ``*_legacy_backup`` by default.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from src.experiment.compact import COMPACT_LAYOUT, TraceStore, write_json_atomic
from src.experiment.layout import provider_model_dir


PROJECT_ROOT = Path(__file__).resolve().parent


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_run_json(path: Path, manifest: dict[str, Any]) -> None:
    data = dict(manifest)
    data["layout_version"] = COMPACT_LAYOUT
    data.pop("provider", None)
    if "prompts" not in data:
        version = data.get("prompt_set") or "v1"
        prompt_dir = PROJECT_ROOT / "prompts" / str(version)
        prompts: dict[str, str] = {}
        if prompt_dir.exists():
            for prompt in sorted(prompt_dir.glob("*.txt")):
                prompts[prompt.stem] = prompt.read_text(encoding="utf-8").strip()
        if prompts:
            data["prompts"] = prompts
    write_json_atomic(path, data)


def _number(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _context(rel: Path) -> dict[str, Any]:
    text = rel.as_posix()
    iteration = _number(r"iteration_(\d+)", text)
    candidate = _number(r"candidate_(\d+)", text)
    attempt = _number(r"attempt_(\d+)", text)
    restart_attempt = _number(r"restart_(\d+)", text)
    scope: str | None = None

    if text.startswith("initializer/repair/"):
        scope = "initializer"
    elif text.startswith("initializer/fallback_restart/"):
        scope = "initializer.fallback"
        if "/repair/" in text and restart_attempt is not None:
            scope = f"initializer.fallback.restart_{restart_attempt:02d}"
    elif iteration is not None:
        prefix = f"iteration_{iteration:03d}"
        if "/selected_repair/" in text:
            scope = f"{prefix}.selected"
        elif "/selected_restart/" in text:
            scope = f"{prefix}.selected_restart"
            if "/repair/" in text and restart_attempt is not None:
                scope = f"{scope}.restart_{restart_attempt:02d}"
        elif "/escape_hybrid/" in text:
            scope = f"{prefix}.hybrid"
        elif "/escape_restart/" in text:
            scope = f"{prefix}.escape_restart"
            if "/repair/" in text and restart_attempt is not None:
                scope = f"{scope}.restart_{restart_attempt:02d}"

    result = {
        "iteration": iteration,
        "candidate": candidate,
        "attempt": attempt,
        "restart_attempt": restart_attempt,
        "scope": scope,
    }
    return {key: value for key, value in result.items() if value is not None}


def _image_target(rel: Path) -> Path | None:
    text = rel.as_posix()
    iteration = _number(r"iteration_(\d+)", text)
    candidate = _number(r"candidate_(\d+)", text)
    attempt = _number(r"attempt_(\d+)", text)
    restart = _number(r"restart_(\d+)", text)

    if text == "initializer/initial_route_model.png":
        return Path("routes/initializer/candidate.png")
    if text.startswith("initializer/"):
        if rel.name == "repaired_route_model.png":
            if "fallback_restart" in text and restart is not None and attempt is not None:
                return Path(f"routes/initializer/restart_{restart:02d}_repair_{attempt:02d}.png")
            if attempt is not None:
                return Path(f"routes/initializer/repair_{attempt:02d}.png")
        if rel.name == "diversity_route_model.png" and restart is not None:
            return Path(f"routes/initializer/restart_{restart:02d}.png")

    if iteration is not None:
        base = Path(f"routes/iteration_{iteration:03d}")
        if rel.name == "route_model.png" and candidate is not None:
            return base / f"C{candidate}.png"
        if rel.name == "repaired_route_model.png" and attempt is not None:
            if "escape_hybrid" in text:
                return base / f"hybrid_repair_{attempt:02d}.png"
            if "selected_repair" in text:
                return base / f"repair_{attempt:02d}.png"
            if "selected_restart" in text and restart is not None:
                return base / f"restart_{restart:02d}_repair_{attempt:02d}.png"
            if "escape_restart" in text and restart is not None:
                return base / f"escape_restart_{restart:02d}_repair_{attempt:02d}.png"
        if rel.name == "hybrid_route_model.png":
            return base / "hybrid.png"
        if rel.name == "diversity_route_model.png" and restart is not None:
            if "escape_restart" in text:
                return base / f"escape_restart_{restart:02d}.png"
            if "selected_restart" in text:
                return base / f"restart_{restart:02d}.png"
            if "escape_hybrid" in text:
                return base / f"hybrid_restart_{restart:02d}.png"

    return None


def _result_image_source(result_path: Path) -> Path | None:
    name = result_path.name
    parent = result_path.parent
    if name == "initializer_candidate_result.json":
        return parent / "initial_route_model.png"
    if name == "candidate_result.json":
        return parent / "route_model.png"
    if name == "repair_result.json":
        return parent / "repaired_route_model.png"
    if name == "diversity_result.json":
        return parent / "diversity_route_model.png"
    if name == "hybrid_result.json":
        return parent / "hybrid_route_model.png"
    return None


def _route_from_result(path: Path, data: dict[str, Any]) -> list[int] | None:
    if path.name in {"initializer_candidate_result.json", "candidate_result.json", "diversity_result.json"}:
        route = data.get("route")
    elif path.name in {"repair_result.json", "hybrid_result.json"}:
        route = data.get("output_route")
    else:
        route = None
    return [int(value) for value in route] if isinstance(route, list) else None


def _accepted_initializer(data: dict[str, Any]) -> tuple[list[int] | None, dict[str, Any]]:
    evaluation = data.get("evaluation") or {}
    if (evaluation.get("validation") or {}).get("valid") is True:
        return data.get("route"), evaluation
    repair = data.get("repair")
    if isinstance(repair, list) and repair:
        last = repair[-1]
        last_eval = last.get("evaluation") or {}
        if (last_eval.get("validation") or {}).get("valid") is True:
            return last.get("output_route"), last_eval
    restart = data.get("restart") or {}
    attempts = restart.get("attempts") or []
    for item in reversed(attempts):
        nested = item.get("repair")
        if isinstance(nested, list) and nested:
            last = nested[-1]
            last_eval = last.get("evaluation") or {}
            if (last_eval.get("validation") or {}).get("valid") is True:
                return last.get("output_route"), last_eval
        item_eval = item.get("evaluation") or {}
        if (item_eval.get("validation") or {}).get("valid") is True:
            return item.get("route"), item_eval
    return None, {}


def _compact_call(data: dict[str, Any]) -> dict[str, Any]:
    request_parts = [
        part for part in (data.get("request_parts") or [])
        if not (part.get("kind") == "text" and part.get("label") == "instructions")
    ]
    return {
        "agent": data.get("agent"),
        "prompt_ref": data.get("agent"),
        "request_parts": request_parts,
        "raw_response": data.get("raw_response", ""),
        "provider": data.get("provider"),
        "model": data.get("model"),
        "phase": data.get("phase"),
        "latency_seconds": data.get("latency_seconds"),
        "usage": data.get("usage") or {},
        "raw_metadata": data.get("raw_metadata") or {},
    }


def _copy_images(source: Path, target: Path) -> tuple[dict[str, str], dict[tuple[int, ...], str]]:
    old_to_new: dict[str, str] = {}
    route_to_image: dict[tuple[int, ...], str] = {}

    for image in sorted(source.rglob("*.png")):
        rel = image.relative_to(source)
        new_rel = _image_target(rel)
        if new_rel is None:
            continue
        destination = target / new_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)
        old_to_new[rel.as_posix()] = new_rel.as_posix()

    for result_path in sorted(source.rglob("*.json")):
        try:
            data = _read_json(result_path)
        except Exception:
            continue
        route = _route_from_result(result_path, data)
        image_source = _result_image_source(result_path)
        if route is None or image_source is None or not image_source.exists():
            continue
        old_rel = image_source.relative_to(source).as_posix()
        new_rel = old_to_new.get(old_rel)
        if new_rel:
            route_to_image[tuple(route)] = new_rel

    return old_to_new, route_to_image


def _migrate_trace(source: Path, target: Path, image_map: dict[str, str], route_map: dict[tuple[int, ...], str]) -> TraceStore:
    trace = TraceStore(target / "trace.jsonl")
    files = [path for path in source.rglob("*") if path.is_file()]
    files.sort(key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))

    provider_attempt_dirs: set[Path] = set()
    for response in source.rglob("output_attempts/attempt_*/provider_response.json"):
        provider_attempt_dirs.add(response.parent)

    for path in files:
        rel = path.relative_to(source)
        context = _context(rel)
        name = path.name

        if name.endswith("_call.json"):
            data = _read_json(path)
            event = {
                "event": "agent_call",
                **context,
                "agent": data.get("agent") or name.removesuffix("_call.json"),
                "call": _compact_call(data),
                "legacy_source": rel.as_posix(),
            }
            trace.append(event)
            continue

        if name == "provider_error.json":
            data = _read_json(path)
            trace.append({"event": "provider_error", **context, **data, "legacy_source": rel.as_posix()})
            continue

        if name == "provider_response.json" and path.parent in provider_attempt_dirs:
            response = _read_json(path)
            raw_path = path.parent / "raw_response.txt"
            error_path = path.parent / "parse_error.json"
            error = _read_json(error_path) if error_path.exists() else {}
            raw = raw_path.read_text(encoding="utf-8-sig") if raw_path.exists() else ""
            trace.append(
                {
                    "event": "model_output_attempt",
                    **_context(path.parent.relative_to(source)),
                    "output_attempt": _number(r"attempt_(\d+)", path.parent.as_posix()),
                    "error_type": error.get("error_type"),
                    "error_message": error.get("message"),
                    "raw_response": raw,
                    "provider_response": response,
                    "legacy_source": path.parent.relative_to(source).as_posix(),
                }
            )
            continue

        if name == "initializer_candidate_result.json":
            data = _read_json(path)
            image_old = (path.parent / "initial_route_model.png").relative_to(source).as_posix()
            trace.append(
                {
                    "event": "initializer_candidate",
                    "route": data.get("route"),
                    "evaluation": data.get("evaluation") or {},
                    "image": image_map.get(image_old),
                    "legacy_source": rel.as_posix(),
                }
            )
            continue

        if name == "initializer_result.json":
            data = _read_json(path)
            route, evaluation = _accepted_initializer(data)
            image = route_map.get(tuple(int(x) for x in route)) if route else None
            trace.append(
                {
                    "event": "initializer_result",
                    "accepted_route": route,
                    "evaluation": evaluation,
                    "image": image,
                    "result": data,
                    "legacy_source": rel.as_posix(),
                }
            )
            continue

        if name == "candidate_result.json":
            data = _read_json(path)
            image_old = (path.parent / "route_model.png").relative_to(source).as_posix()
            trace.append(
                {
                    "event": "critic_result",
                    **context,
                    "route": data.get("route"),
                    "evaluation": data.get("evaluation") or {},
                    "image": image_map.get(image_old),
                    "legacy_source": rel.as_posix(),
                }
            )
            continue

        if name == "scorer_result.json":
            data = _read_json(path)
            trace.append({"event": "scorer_result", **context, **data, "legacy_source": rel.as_posix()})
            continue

        if name == "repair_result.json":
            data = _read_json(path)
            image_old = (path.parent / "repaired_route_model.png").relative_to(source).as_posix()
            trace.append(
                {
                    "event": "repair_result",
                    **context,
                    "output_route": data.get("output_route"),
                    "evaluation": data.get("evaluation") or {},
                    "image": image_map.get(image_old),
                    "result": data,
                    "legacy_source": rel.as_posix(),
                }
            )
            continue

        if name == "diversity_result.json":
            data = _read_json(path)
            image_old = (path.parent / "diversity_route_model.png").relative_to(source).as_posix()
            trace.append(
                {
                    "event": "diversity_result",
                    **context,
                    "global_restart_count": data.get("global_restart_count"),
                    "route": data.get("route"),
                    "evaluation": data.get("evaluation") or {},
                    "image": image_map.get(image_old),
                    "result": data,
                    "legacy_source": rel.as_posix(),
                }
            )
            continue

        if name == "hybrid_result.json":
            data = _read_json(path)
            image_old = (path.parent / "hybrid_route_model.png").relative_to(source).as_posix()
            trace.append(
                {
                    "event": "hybrid_result",
                    **context,
                    "input_route": data.get("input_route"),
                    "output_route": data.get("output_route"),
                    "evaluation": data.get("evaluation") or {},
                    "two_opt_audit": data.get("two_opt_audit") or {},
                    "image": image_map.get(image_old),
                    "legacy_source": rel.as_posix(),
                }
            )
            continue

        if name == "iteration_result.json":
            data = _read_json(path)
            trace.append(
                {
                    "event": "iteration_result",
                    "iteration": int(data.get("iteration") or context.get("iteration") or 0),
                    "result": data,
                    "legacy_source": rel.as_posix(),
                }
            )

    return trace


def _latest_error(source: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidates = sorted(source.rglob("provider_error.json"), key=lambda path: path.stat().st_mtime_ns)
    if not candidates:
        return None, None
    path = candidates[-1]
    data = _read_json(path)
    context = _context(path.relative_to(source))
    return data, {"phase": data.get("phase"), **context}


def _migrate_model(source: Path, target: Path, manifest: dict[str, Any]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    image_map, route_map = _copy_images(source, target)
    _migrate_trace(source, target, image_map, route_map)

    checkpoint_path = source / "checkpoint.json"
    summary_path = source / "summary.json"
    checkpoint = _read_json(checkpoint_path) if checkpoint_path.exists() else None
    summary = _read_json(summary_path) if summary_path.exists() else None
    provider = manifest.get("provider") or {}
    last_error, current = _latest_error(source)

    working_image = None
    if isinstance(checkpoint, dict):
        route = checkpoint.get("working_route") or []
        if route:
            working_image = route_map.get(tuple(int(value) for value in route))
    if working_image is None:
        old_working = source / "resume" / "working_route_model.png"
        if old_working.exists():
            destination = target / "routes" / "working.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_working, destination)
            working_image = "routes/working.png"

    status = "completed" if summary else "partial"
    state = {
        "layout_version": COMPACT_LAYOUT,
        "provider": provider.get("name"),
        "model": provider.get("model"),
        "status": status,
        "config_sha256": manifest.get("config_sha256"),
        "instance_sha256": (manifest.get("problem") or {}).get("source_sha256"),
        "seed": (manifest.get("run_parameters") or {}).get("seed"),
        "checkpoint": checkpoint,
        "working_image": working_image,
        "completed_iterations": (checkpoint or {}).get("completed_iteration", 0),
        "summary": summary,
        "last_error": last_error,
        "current": current or ({"phase": "completed"} if status == "completed" else None),
        "migrated_from": "legacy_artifact_layout",
    }
    write_json_atomic(target / "state.json", state)

    old_analysis = source / "analysis"
    if old_analysis.exists():
        new_analysis = target / "analysis"
        new_analysis.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "iterations.csv", "selected_vs_oracle.png", "analysis_report.txt"):
            src = old_analysis / name
            if src.exists():
                shutil.copy2(src, new_analysis / name)


def _source_models(root: Path) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]], Path | None]:
    root_manifest = _read_json(root / "run_manifest.json")
    provider_branches = sorted((root / "providers").glob("*/*/run_manifest.json")) if (root / "providers").exists() else []
    if provider_branches:
        models = [(path.parent, _read_json(path)) for path in provider_branches]
        problem = root / "inputs" / "problem_model.png"
        return root_manifest, models, problem if problem.exists() else None

    provider = root_manifest.get("provider") or {}
    if not provider.get("name") or not provider.get("model"):
        raise SystemExit("Legacy run manifest provider/model içermiyor")
    problem = root / "inputs" / "problem_model.png"
    return root_manifest, [(root, root_manifest)], problem if problem.exists() else None


def migrate(run_dir: Path, *, keep_backup: bool = True) -> tuple[Path, Path | None]:
    run_dir = run_dir.resolve()
    if (run_dir / "run.json").exists():
        raise SystemExit("Run zaten compact_v3 düzeninde")
    if not (run_dir / "run_manifest.json").exists():
        raise SystemExit("run_manifest.json bulunamadı")

    shared_manifest, models, problem_image = _source_models(run_dir)
    temp = run_dir.with_name(run_dir.name + ".__compact_tmp__")
    backup = run_dir.with_name(run_dir.name + "_legacy_backup")
    if temp.exists() or backup.exists():
        raise SystemExit("Migration temp/backup klasörü zaten var; önce kontrol edin")

    temp.mkdir(parents=True)
    try:
        _write_run_json(temp / "run.json", shared_manifest)
        if problem_image is not None:
            shutil.copy2(problem_image, temp / "problem.png")

        for source_model, manifest in models:
            provider = (manifest.get("provider") or {}).get("name")
            model = (manifest.get("provider") or {}).get("model")
            if not provider or not model:
                raise RuntimeError(f"Provider/model bulunamadı: {source_model}")
            target_model = provider_model_dir(temp, provider, model)
            _migrate_model(source_model, target_model, manifest)

        if not (temp / "run.json").exists() or not list((temp / "providers").glob("*/*/state.json")):
            raise RuntimeError("Compact migration doğrulaması başarısız")

        run_dir.rename(backup)
        temp.rename(run_dir)
        if not keep_backup:
            shutil.rmtree(backup)
            return run_dir, None
        return run_dir, backup
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate AVMA run artifacts to compact_v3 without API calls")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--delete-backup", action="store_true", help="Delete legacy backup after successful migration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.run_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    migrated, backup = migrate(path, keep_backup=not args.delete_backup)
    print("Migration tamamlandı (API çağrısı yapılmadı)")
    print(f"Run: {migrated}")
    if backup is not None:
        print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
