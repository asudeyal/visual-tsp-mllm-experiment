import argparse
import sys
from pathlib import Path

from src.experiment.compact_analysis import build_compact_analysis, write_compact_analysis

def main() -> None:
    parser = argparse.ArgumentParser(description="AVMA-CVRP Standart Analiz")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    run_id_dir = args.output_dir / "runs" / args.run_id
    if not run_id_dir.is_dir():
        sys.exit(f"Hata: Run klasörü bulunamadı -> {run_id_dir}")

    # compact_analysis.py, dizinin 'providers/provider_name/model_name' seviyesinde olmasını bekler
    provider_dirs = list(run_id_dir.glob("providers/*/*"))
    if not provider_dirs:
        sys.exit("Hata: 'providers/*/*' dizini altında model klasörü bulunamadı.")

    for run_dir in provider_dirs:
        if not (run_dir / "state.json").is_file():
            continue
        
        # Analiz fonksiyonlarını çağır ve raporu yazdır
        summary, rows, report = build_compact_analysis(run_dir)
        paths = write_compact_analysis(run_dir, summary, rows, report)
        
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        
        print(report, end="")
        print(f"Rapor dosyaları şuraya kaydedildi: {paths['report'].parent}")

if __name__ == "__main__":
    main()