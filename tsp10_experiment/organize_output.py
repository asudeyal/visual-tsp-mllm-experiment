"""Eski TSP çıktılarını yöntem ve images klasörlerine güvenle düzenler."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.output_paths import validate_run_id


MA1_CANDIDATE_RE = re.compile(
    r"^gemini_ma1_iteration_(\d+)_candidate_(\d+)\.png$"
)
MA1_SELECTED_RE = re.compile(r"^gemini_ma1_iteration_(\d+)_selected\.png$")
MA2_ITERATION_RE = re.compile(r"^gemini_ma2_iteration_(\d+)\.png$")

FLAT_JSON_METHODS = {
    "baseline_results.json": "baseline",
    "baseline_summary.json": "baseline",
    "gemini_zero_shot_results.json": "zero_shot",
    "gemini_zero_shot_summary.json": "zero_shot",
    "gemini_multi_agent1_checkpoint.json": "multi_agent1",
    "gemini_multi_agent1_results.json": "multi_agent1",
    "gemini_multi_agent1_summary.json": "multi_agent1",
    "gemini_multi_agent2_checkpoint.json": "multi_agent2",
    "gemini_multi_agent2_results.json": "multi_agent2",
    "gemini_multi_agent2_summary.json": "multi_agent2",
}

FLAT_BASELINE_IMAGES = {"points.png", "or_tools_route.png", "exact_route.png"}
FLAT_ZERO_SHOT_IMAGES = {"gemini_zero_shot_route.png"}
METHOD_NAMES = ("baseline", "zero_shot", "multi_agent1", "multi_agent2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--legacy-run-id",
        default="seed42_initial_run",
        help=(
            "output kökündeki eski düz dosyaların taşınacağı run kimliği. "
            "Varsayılan: seed42_initial_run."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Hiçbir dosyayı değiştirmeden yapılacak işlemleri gösterir.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Önizlenen taşıma ve JSON yol güncellemelerini uygular.",
    )
    return parser.parse_args()


def ma1_image_destination(image: Path, method_dir: Path) -> Path | None:
    candidate_match = MA1_CANDIDATE_RE.fullmatch(image.name)
    if candidate_match:
        iteration, candidate = candidate_match.groups()
        return (
            method_dir
            / "images"
            / f"iteration_{int(iteration):02d}"
            / f"candidate_{int(candidate):02d}.png"
        )
    selected_match = MA1_SELECTED_RE.fullmatch(image.name)
    if selected_match:
        iteration = int(selected_match.group(1))
        return method_dir / "images" / f"iteration_{iteration:02d}" / "selected.png"
    return None


def ma2_image_destination(image: Path, method_dir: Path) -> Path | None:
    match = MA2_ITERATION_RE.fullmatch(image.name)
    if match:
        return method_dir / "images" / f"iteration_{int(match.group(1)):02d}.png"
    return None


def flat_file_destination(file_path: Path, target_run: Path) -> Path | None:
    method = FLAT_JSON_METHODS.get(file_path.name)
    if method is not None:
        return target_run / method / file_path.name
    if file_path.name in FLAT_BASELINE_IMAGES:
        return target_run / "baseline" / "images" / file_path.name
    if file_path.name in FLAT_ZERO_SHOT_IMAGES:
        return target_run / "zero_shot" / "images" / file_path.name

    destination = ma1_image_destination(file_path, target_run / "multi_agent1")
    if destination is not None:
        return destination
    return ma2_image_destination(file_path, target_run / "multi_agent2")


def method_image_destination(image: Path, method: str) -> Path:
    method_dir = image.parent
    if method == "multi_agent1":
        destination = ma1_image_destination(image, method_dir)
        if destination is not None:
            return destination
    if method == "multi_agent2":
        destination = ma2_image_destination(image, method_dir)
        if destination is not None:
            return destination
    return method_dir / "images" / image.name


def build_move_plan(
    output_dir: Path,
    *,
    legacy_run_id: str,
) -> tuple[dict[Path, Path], list[Path]]:
    """Taşınacak dosyaları ve kökte bırakılacak bilinmeyen dosyaları döndürür."""

    moves: dict[Path, Path] = {}
    unknown_flat_files: list[Path] = []
    target_run = output_dir / "runs" / legacy_run_id

    for file_path in sorted(path for path in output_dir.iterdir() if path.is_file()):
        destination = flat_file_destination(file_path, target_run)
        if destination is None:
            unknown_flat_files.append(file_path)
        else:
            moves[file_path] = destination

    runs_dir = output_dir / "runs"
    if runs_dir.exists():
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            for method in METHOD_NAMES:
                method_dir = run_dir / method
                if not method_dir.exists():
                    continue
                for image in sorted(method_dir.glob("*.png")):
                    moves[image] = method_image_destination(image, method)

    return moves, unknown_flat_files


def normalized_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def rewrite_path_string(value: str, moves: dict[Path, Path]) -> tuple[str, bool]:
    normalized_value = normalized_path(value)
    separator = "\\" if "\\" in value and "/" not in value else "/"

    for source, destination in moves.items():
        source_text = normalized_path(source)
        destination_text = normalized_path(destination)
        if normalized_value == source_text:
            return destination_text.replace("/", separator), True
        suffix = f"/{source_text}"
        if normalized_value.endswith(suffix):
            prefix = normalized_value[: -len(source_text)]
            updated = f"{prefix}{destination_text}"
            return updated.replace("/", separator), True
    return value, False


def rewrite_json_value(value: Any, moves: dict[Path, Path]) -> tuple[Any, int]:
    if isinstance(value, dict):
        updated: dict[str, Any] = {}
        changes = 0
        for key, child in value.items():
            updated_child, child_changes = rewrite_json_value(child, moves)
            updated[key] = updated_child
            changes += child_changes
        return updated, changes
    if isinstance(value, list):
        updated_list = []
        changes = 0
        for child in value:
            updated_child, child_changes = rewrite_json_value(child, moves)
            updated_list.append(updated_child)
            changes += child_changes
        return updated_list, changes
    if isinstance(value, str):
        updated, changed = rewrite_path_string(value, moves)
        return updated, int(changed)
    return value, 0


def build_json_updates(
    output_dir: Path,
    moves: dict[Path, Path],
) -> dict[Path, tuple[Any, int]]:
    updates: dict[Path, tuple[Any, int]] = {}
    for source_json in sorted(output_dir.rglob("*.json")):
        data = json.loads(source_json.read_text(encoding="utf-8"))
        updated_data, change_count = rewrite_json_value(data, moves)
        destination_json = moves.get(source_json, source_json)
        if change_count:
            updates[destination_json] = (updated_data, change_count)
    return updates


def validate_move_plan(moves: dict[Path, Path]) -> None:
    destinations: set[Path] = set()
    for source, destination in moves.items():
        if destination in destinations:
            raise SystemExit(f"Birden fazla dosya aynı hedefe taşınacak: {destination}")
        destinations.add(destination)
        if destination.exists() and destination != source:
            raise SystemExit(
                "Hedef dosya zaten var; hiçbir değişiklik yapılmadı: "
                f"{destination}"
            )


def atomic_write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_plan(
    moves: dict[Path, Path],
    json_updates: dict[Path, tuple[Any, int]],
) -> None:
    for source, destination in moves.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    for json_path, (data, _) in json_updates.items():
        atomic_write_json(json_path, data)


def main() -> None:
    args = parse_args()
    try:
        legacy_run_id = validate_run_id(args.legacy_run_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = args.output_dir
    if not output_dir.exists():
        raise SystemExit(f"Output klasörü bulunamadı: {output_dir}")

    moves, unknown_flat_files = build_move_plan(
        output_dir,
        legacy_run_id=legacy_run_id,
    )
    validate_move_plan(moves)
    json_updates = build_json_updates(output_dir, moves)

    print(f"\nPlanlanan dosya taşıma sayısı: {len(moves)}")
    for source, destination in moves.items():
        print(f"TAŞI: {source} -> {destination}")

    reference_count = sum(count for _, count in json_updates.values())
    print(f"\nGüncellenecek JSON dosyası: {len(json_updates)}")
    print(f"Güncellenecek JSON yol referansı: {reference_count}")

    if unknown_flat_files:
        print("\nTanınmadığı için output kökünde bırakılacak dosyalar:")
        for file_path in unknown_flat_files:
            print(f"BIRAK: {file_path}")

    if args.dry_run:
        print("\nDRY-RUN tamamlandı; hiçbir dosya değiştirilmedi.")
        return

    apply_plan(moves, json_updates)
    print("\nOutput düzenleme tamamlandı.")
    print(f"Eski düz çıktılar şu çalıştırmaya taşındı: {legacy_run_id}")


if __name__ == "__main__":
    main()
