"""Eski Gemini/OpenRouter çıktılarını providers/ düzenine taşır."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core import normalize_run_id
from src.output_migration import (
    OutputMigrationError,
    apply_plan,
    build_plan,
    undo_migration,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Gösterilen taşıma planını uygular.",
    )
    mode.add_argument(
        "--undo",
        action="store_true",
        help="En son uygulanmış taşıma işlemini geri alır.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = normalize_run_id(args.run_id)
    run_dir = Path(args.output_dir) / "runs" / run_id
    try:
        if args.undo:
            record = undo_migration(run_dir)
            print("Output migration geri alındı.")
            print(f"Run ID: {run_id}")
            print(f"Geri alınan taşıma: {len(record['moves'])}")
            print(
                "Analiz JSON'unu eski düzene göre yenilemek için "
                "run_analysis.py çalıştırılmalıdır."
            )
            return

        plan = build_plan(run_dir)
        if not plan:
            print("Taşınacak tarihsel çıktı bulunamadı.")
            print("Klasör zaten birleşik providers/ düzeninde.")
            return
        print(f"Planlanan taşıma: {len(plan)}")
        for move in plan:
            source = move.source.relative_to(run_dir).as_posix()
            target = move.target.relative_to(run_dir).as_posix()
            print(f"  {source} -> {target}")
        if not args.apply:
            print(
                "\nDry-run tamamlandı; hiçbir dosya değiştirilmedi."
            )
            print(
                "Plan uygunsa aynı komutu --apply ile çalıştırın."
            )
            return
        record = apply_plan(run_dir, plan)
        print("\nOutput migration tamamlandı.")
        print(f"Run ID: {run_id}")
        print(f"Taşınan öğe: {len(record['moves'])}")
        print(
            "Güncellenen JSON: "
            f"{record['json_rewrite']['changed_file_count']}"
        )
        print(
            "Güncellenen yol referansı: "
            f"{record['json_rewrite']['changed_path_value_count']}"
        )
        print(
            "Kontrol için run_analysis.py çalıştırılmalıdır."
        )
    except OutputMigrationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
