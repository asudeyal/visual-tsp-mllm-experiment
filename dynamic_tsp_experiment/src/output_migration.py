"""Tarihsel deney çıktılarını birleşik provider düzenine taşır."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.core import read_json, write_json
from src.metrics import utc_now_iso
from src.providers.registry import model_slug


MIGRATION_SCHEMA_VERSION = "1.0"
MIGRATION_FILE = "output_layout_migration.json"


class OutputMigrationError(RuntimeError):
    """Güvenli taşıma ön koşulları karşılanmadığında yükseltilir."""


@dataclass(frozen=True)
class Move:
    source: Path
    target: Path

    def relative_dict(self, run_dir: Path) -> dict[str, str]:
        return {
            "source": self.source.relative_to(run_dir).as_posix(),
            "target": self.target.relative_to(run_dir).as_posix(),
        }


def _model_alias(result_path: Path, fallback: str) -> str:
    if not result_path.exists():
        return fallback
    model = read_json(result_path).get("model")
    if isinstance(model, str):
        return model
    model = model or {}
    return str(
        model.get("alias")
        or model.get("name")
        or model.get("requested_name")
        or fallback
    )


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _gemini_moves(run_dir: Path) -> list[Move]:
    legacy = {
        "zero_shot": run_dir / "zero_shot",
        "multi_agent1": run_dir / "multi_agent1",
        "multi_agent2": run_dir / "multi_agent2",
    }
    result_paths = (
        legacy["zero_shot"] / "zero_shot_results.json",
        legacy["multi_agent1"] / "multi_agent1_results.json",
        legacy["multi_agent2"] / "multi_agent2_results.json",
    )
    anchor = _first_existing(result_paths)
    if anchor is None:
        return []
    alias = model_slug(_model_alias(anchor, "gemini-2.5-flash"))
    root = run_dir / "providers" / "gemini" / alias
    return [
        Move(source, root / method)
        for method, source in legacy.items()
        if source.exists()
    ]


def _openrouter_moves(run_dir: Path) -> list[Move]:
    root = run_dir / "model_comparisons" / "openrouter"
    if not root.exists():
        return []
    moves: list[Move] = []
    analysis_legacy = (
        run_dir / "analysis" / "legacy_openrouter"
    )
    for child in sorted(root.iterdir()):
        if child.is_file():
            moves.append(Move(child, analysis_legacy / child.name))
            continue
        if not child.is_dir():
            continue
        target = (
            run_dir
            / "providers"
            / "openrouter"
            / model_slug(child.name)
        )
        known = {
            "zero_shot": target / "zero_shot",
            "multi_agent1": target / "multi_agent1",
            "multi_agent2": target / "multi_agent2",
        }
        zero_dir = child / "zero_shot"
        if zero_dir.exists():
            moves.append(Move(zero_dir, known["zero_shot"]))
        zero_result = child / "zero_shot_results.json"
        if zero_result.exists():
            moves.append(
                Move(
                    zero_result,
                    known["zero_shot"]
                    / "zero_shot_results.json",
                )
            )
        zero_images = child / "images"
        if zero_images.exists():
            moves.append(
                Move(
                    zero_images,
                    known["zero_shot"] / "images",
                )
            )
        for method in ("multi_agent1", "multi_agent2"):
            source = child / method
            if source.exists():
                moves.append(Move(source, known[method]))
        planned_sources = {move.source for move in moves}
        for extra in sorted(child.iterdir()):
            if extra in planned_sources:
                continue
            moves.append(
                Move(
                    extra,
                    target / "legacy_artifacts" / extra.name,
                )
            )
    return moves


def build_plan(run_dir: Path) -> list[Move]:
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "run_manifest.json").exists():
        raise OutputMigrationError(
            f"Run manifesti bulunamadı: {run_dir}"
        )
    moves = [
        *_gemini_moves(run_dir),
        *_openrouter_moves(run_dir),
    ]
    _validate_plan(moves)
    return moves


def _validate_plan(moves: list[Move]) -> None:
    targets: set[Path] = set()
    for move in moves:
        if not move.source.exists():
            raise OutputMigrationError(
                f"Kaynak bulunamadı: {move.source}"
            )
        if move.target.exists():
            raise OutputMigrationError(
                "Hedef zaten var; hiçbir dosya taşınmadı: "
                f"{move.target}"
            )
        if move.target in targets:
            raise OutputMigrationError(
                f"Aynı hedef iki kez planlandı: {move.target}"
            )
        targets.add(move.target)


def _mapping(moves: list[Move], run_dir: Path) -> list[tuple[str, str]]:
    values = [
        (
            move.source.relative_to(run_dir).as_posix(),
            move.target.relative_to(run_dir).as_posix(),
        )
        for move in moves
    ]
    return sorted(values, key=lambda item: len(item[0]), reverse=True)


def _replace_path(
    value: str,
    mappings: list[tuple[str, str]],
) -> str:
    for old, new in mappings:
        if value == old:
            return new
        old_prefix = old + "/"
        if old_prefix in value:
            return value.replace(old_prefix, new + "/", 1)
        old_windows = old.replace("/", "\\") + "\\"
        if old_windows in value:
            return value.replace(
                old_windows,
                new.replace("/", "\\") + "\\",
                1,
            )
    return value


def _rewrite_value(
    value: Any,
    mappings: list[tuple[str, str]],
) -> tuple[Any, int]:
    if isinstance(value, str):
        updated = _replace_path(value, mappings)
        return updated, int(updated != value)
    if isinstance(value, list):
        output: list[Any] = []
        count = 0
        for item in value:
            updated, changes = _rewrite_value(item, mappings)
            output.append(updated)
            count += changes
        return output, count
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            updated, changes = _rewrite_value(item, mappings)
            output[key] = updated
            count += changes
        return output, count
    return value, 0


def rewrite_json_paths(
    run_dir: Path,
    mappings: list[tuple[str, str]],
) -> tuple[int, int]:
    changed_files = 0
    changed_values = 0
    for path in sorted(Path(run_dir).rglob("*.json")):
        if path.name == MIGRATION_FILE:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OutputMigrationError(
                f"JSON okunamadı: {path}"
            ) from exc
        updated, changes = _rewrite_value(value, mappings)
        if changes:
            write_json(path, updated)
            changed_files += 1
            changed_values += changes
    return changed_files, changed_values


def _prune_empty(path: Path, *, stop: Path) -> None:
    current = path
    stop = stop.resolve()
    while current.exists() and current.resolve() != stop:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def migration_path(run_dir: Path) -> Path:
    return Path(run_dir) / "analysis" / MIGRATION_FILE


def apply_plan(run_dir: Path, moves: list[Move]) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    _validate_plan(moves)
    record_path = migration_path(run_dir)
    if record_path.exists():
        record = read_json(record_path)
        if record.get("status") == "applied":
            raise OutputMigrationError(
                "Bu koşuya daha önce uygulanmış bir migration kaydı var."
            )
    mappings = _mapping(moves, run_dir)
    completed: list[Move] = []
    try:
        for move in moves:
            move.target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(move.source), str(move.target))
            completed.append(move)
        changed_files, changed_values = rewrite_json_paths(
            run_dir,
            mappings,
        )
    except Exception:
        for move in reversed(completed):
            if move.target.exists() and not move.source.exists():
                move.source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(move.target), str(move.source))
        reverse = [(new, old) for old, new in mappings]
        rewrite_json_paths(run_dir, reverse)
        raise
    for move in moves:
        _prune_empty(move.source.parent, stop=run_dir)
    record = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "status": "applied",
        "applied_at_utc": utc_now_iso(),
        "run_id": run_dir.name,
        "moves": [
            move.relative_dict(run_dir)
            for move in moves
        ],
        "json_rewrite": {
            "changed_file_count": changed_files,
            "changed_path_value_count": changed_values,
        },
    }
    write_json(record_path, record)
    return record


def undo_migration(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    record_path = migration_path(run_dir)
    if not record_path.exists():
        raise OutputMigrationError(
            "Geri alınacak migration kaydı bulunamadı."
        )
    record = read_json(record_path)
    if record.get("status") != "applied":
        raise OutputMigrationError(
            "Migration uygulanmış durumda değil."
        )
    moves = [
        Move(
            run_dir / item["source"],
            run_dir / item["target"],
        )
        for item in record.get("moves", [])
    ]
    for move in moves:
        if not move.target.exists():
            raise OutputMigrationError(
                f"Geri alma hedefi bulunamadı: {move.target}"
            )
        if move.source.exists():
            raise OutputMigrationError(
                f"Eski yol yeniden oluşmuş: {move.source}"
            )
    mappings = _mapping(moves, run_dir)
    for move in reversed(moves):
        move.source.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.target), str(move.source))
        _prune_empty(move.target.parent, stop=run_dir)
    changed_files, changed_values = rewrite_json_paths(
        run_dir,
        [(new, old) for old, new in mappings],
    )
    record["status"] = "undone"
    record["undone_at_utc"] = utc_now_iso()
    record["undo_json_rewrite"] = {
        "changed_file_count": changed_files,
        "changed_path_value_count": changed_values,
    }
    write_json(record_path, record)
    return record
