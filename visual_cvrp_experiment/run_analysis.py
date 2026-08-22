"""Kayıtlı görsel CVRP deney sonuçlarını terminalde raporla."""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import hypot
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.problem import CVRPProblem, Node
from src.rendering import (
    DemandEncoding,
    render_solution,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output"


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--run-id",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def _normalize_run_id(run_id: str) -> str:
    stripped = run_id.strip()
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        stripped,
    ):
        raise ValueError(
            "Run ID yalnızca harf, rakam, nokta, "
            "alt çizgi ve tire içerebilir."
        )
    return stripped


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Gerekli JSON dosyası bulunamadı: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError(
            f"JSON kökü nesne olmalıdır: {path}"
        )
    return payload


def _format_number(
    value: Any,
    *,
    decimals: int = 4,
) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "evet" if value else "hayır"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _format_ids(value: Any) -> str:
    if not value:
        return "-"
    return ", ".join(
        str(item)
        for item in value
    )


def render_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    right_aligned: Iterable[int] = (),
) -> str:
    """Bağımlılıksız, ASCII terminal tablosu oluştur."""

    string_rows = [
        [str(value) for value in row]
        for row in rows
    ]
    widths = [
        len(header)
        for header in headers
    ]

    for row in string_rows:
        if len(row) != len(headers):
            raise ValueError(
                "Tablo satırı başlık sayısıyla eşleşmiyor."
            )
        for index, value in enumerate(row):
            widths[index] = max(
                widths[index],
                len(value),
            )

    right_aligned_set = set(right_aligned)

    def border() -> str:
        return "+" + "+".join(
            "-" * (width + 2)
            for width in widths
        ) + "+"

    def format_row(row: Sequence[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            if index in right_aligned_set:
                cells.append(
                    f" {value:>{widths[index]}} "
                )
            else:
                cells.append(
                    f" {value:<{widths[index]}} "
                )
        return "|" + "|".join(cells) + "|"

    lines = [
        border(),
        format_row(list(headers)),
        border(),
    ]
    lines.extend(
        format_row(row)
        for row in string_rows
    )
    lines.append(border())
    return "\n".join(lines)


def _load_results(
    run_dir: Path,
) -> list[dict[str, Any]]:
    result_paths = sorted(
        run_dir.glob(
            "providers/*/*/*/single_call_results.json"
        )
    )
    return [
        _load_json(path)
        for path in result_paths
    ]


def _problem_from_payload(
    payload: dict[str, Any],
) -> CVRPProblem:
    depot_id = int(payload["depot_id"])
    nodes = tuple(
        Node(
            node_id=int(node["id"]),
            x=float(node["x"]),
            y=float(node["y"]),
            demand=int(node["demand"]),
        )
        for node in payload["nodes"]
    )
    depot = next(
        (
            node
            for node in nodes
            if node.node_id == depot_id
        ),
        None,
    )
    if depot is None:
        raise ValueError(
            "Problem JSON'unda depo düğümü bulunamadı."
        )

    return CVRPProblem(
        name=str(payload["name"]),
        depot=depot,
        customers=tuple(
            node
            for node in nodes
            if node.node_id != depot_id
        ),
        vehicle_capacity=int(
            payload["vehicle_capacity"]
        ),
        vehicle_count=(
            int(payload["vehicle_count"])
            if payload.get("vehicle_count") is not None
            else None
        ),
    )


def _safe_file_component(value: Any) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        str(value).strip(),
    ).strip("-.")
    return normalized or "unknown"


def generate_route_images(
    run_dir: Path,
) -> list[Path]:
    """Kayıtlı baseline ve model rotalarından PNG üret."""

    problem_payload = _load_json(
        run_dir / "inputs" / "problem.json"
    )
    baseline = _load_json(
        run_dir
        / "baseline"
        / "exact_results.json"
    )
    results = _load_results(run_dir)
    problem = _problem_from_payload(
        problem_payload
    )
    images_dir = run_dir / "analysis" / "images"
    generated_paths = []

    baseline_path = (
        images_dir / "baseline_exact_routes.png"
    )
    render_solution(
        problem,
        baseline.get("routes") or [],
        baseline_path,
        title=(
            "Exact CVRP baseline — distance "
            f"{_format_number(baseline.get('total_distance'))}"
        ),
        route_loads=(
            baseline.get("route_loads") or None
        ),
        encoding=DemandEncoding.NUMERIC,
    )
    generated_paths.append(baseline_path)

    for result in results:
        validation = result.get("validation")
        if (
            result.get("status") != "completed"
            or not isinstance(validation, dict)
            or not validation.get("routes")
        ):
            continue

        provider = _safe_file_component(
            result.get("provider", "unknown")
        )
        model = _safe_file_component(
            result.get("model", "unknown")
        )
        encoding = DemandEncoding(
            result.get("encoding", "numeric")
        )
        result_path = images_dir / (
            f"{provider}_{model}_"
            f"{encoding.value}_routes.png"
        )
        distance = validation.get(
            "total_distance"
        )
        gap = result.get(
            "optimality_gap_percent"
        )
        title = (
            f"{result.get('provider', '-')} / "
            f"{result.get('model', '-')} / "
            f"{encoding.value} — distance "
            f"{_format_number(distance)}"
        )
        if gap is not None:
            title += (
                f" — gap {_format_number(gap)}%"
            )

        render_solution(
            problem,
            [
                route.get("route") or []
                for route in validation["routes"]
            ],
            result_path,
            title=title,
            route_loads=[
                int(route.get("load", 0))
                for route in validation["routes"]
            ],
            encoding=encoding,
        )
        generated_paths.append(result_path)

    return generated_paths


def _summary_rows(
    baseline: dict[str, Any],
    results: Sequence[dict[str, Any]],
) -> list[list[str]]:
    rows = [
        [
            "baseline",
            "Exact DP",
            "-",
            "evet",
            _format_number(
                baseline.get("vehicle_count")
            ),
            _format_number(
                baseline.get("total_distance")
            ),
            "0.0000",
            "-",
            "-",
            "-",
            "-",
            "-",
        ]
    ]

    for result in results:
        validation = result.get("validation") or {}
        model_response = (
            result.get("model_response") or {}
        )
        usage = model_response.get("usage") or {}

        rows.append(
            [
                str(result.get("provider", "-")),
                str(result.get("model", "-")),
                str(result.get("encoding", "-")),
                (
                    "evet"
                    if validation.get("valid") is True
                    else (
                        "hayır"
                        if validation.get("valid") is False
                        else "-"
                    )
                ),
                _format_number(
                    validation.get("route_count")
                ),
                _format_number(
                    validation.get("total_distance")
                ),
                _format_number(
                    result.get(
                        "optimality_gap_percent"
                    )
                ),
                _format_number(
                    model_response.get(
                        "elapsed_seconds"
                    )
                ),
                _format_number(
                    usage.get("prompt_token_count")
                ),
                _format_number(
                    usage.get("output_token_count")
                ),
                _format_number(
                    usage.get("thoughts_token_count")
                ),
                _format_number(
                    usage.get("total_token_count")
                ),
            ]
        )

    return rows


def _route_table(
    routes: Sequence[dict[str, Any]],
    *,
    capacity: int,
) -> str:
    rows = []
    for route in routes:
        rows.append(
            [
                _format_number(
                    route.get("route_index")
                ),
                " -> ".join(
                    str(node_id)
                    for node_id in route.get(
                        "route",
                        [],
                    )
                ),
                (
                    f"{route.get('load', '-')}/"
                    f"{capacity}"
                ),
                _format_number(
                    route.get("capacity_excess")
                ),
                _format_number(
                    route.get("distance")
                ),
                (
                    "evet"
                    if route.get("valid") is True
                    else "hayır"
                ),
            ]
        )

    return render_table(
        [
            "Rota",
            "Yol",
            "Yük/Q",
            "Aşım",
            "Mesafe",
            "Geçerli",
        ],
        rows,
        right_aligned={0, 2, 3, 4},
    )


def _baseline_route_table(
    baseline: dict[str, Any],
    *,
    problem: dict[str, Any],
) -> str:
    routes = baseline.get("routes") or []
    loads = baseline.get("route_loads") or []
    capacity = int(problem["vehicle_capacity"])
    coordinates = {
        int(node["id"]): (
            float(node["x"]),
            float(node["y"]),
        )
        for node in problem.get("nodes") or []
    }

    def route_distance(route: Sequence[int]) -> float | None:
        if not route or any(
            int(node_id) not in coordinates
            for node_id in route
        ):
            return None

        return sum(
            hypot(
                coordinates[int(route[index])][0]
                - coordinates[int(route[index + 1])][0],
                coordinates[int(route[index])][1]
                - coordinates[int(route[index + 1])][1],
            )
            for index in range(len(route) - 1)
        )

    rows = [
        {
            "route_index": index,
            "route": route,
            "load": (
                loads[index - 1]
                if index - 1 < len(loads)
                else "-"
            ),
            "capacity_excess": 0,
            "distance": route_distance(route),
            "valid": True,
        }
        for index, route in enumerate(
            routes,
            start=1,
        )
    ]

    return _route_table(
        rows,
        capacity=capacity,
    )


def build_report(
    run_dir: Path,
    *,
    image_paths: Sequence[Path] = (),
) -> str:
    """Bir run klasöründeki bütün kayıtlı sonuçları raporla."""

    problem = _load_json(
        run_dir / "inputs" / "problem.json"
    )
    baseline = _load_json(
        run_dir
        / "baseline"
        / "exact_results.json"
    )
    results = _load_results(run_dir)

    capacity = int(problem["vehicle_capacity"])
    lines = [
        "Görsel CVRP deney raporu",
        "",
        f"Run ID: {run_dir.name}",
        (
            f"Problem: {problem['name']} | "
            f"Düğüm: {problem['dimension']} | "
            f"Müşteri: {problem['customer_count']}"
        ),
        (
            f"Araç kapasitesi Q: {capacity} | "
            f"Araç sınırı K: {problem['vehicle_count']} | "
            f"Toplam talep: {problem['total_demand']}"
        ),
        (
            "Kesin optimum: "
            f"{_format_number(baseline['total_distance'])} "
            "| Kanıtlı: "
            f"{'evet' if baseline.get('proven_optimal') else 'hayır'}"
        ),
        "",
        "Genel sonuçlar",
        render_table(
            [
                "Provider",
                "Model",
                "Kodlama",
                "Geçerli",
                "Rota",
                "Mesafe",
                "Gap %",
                "API sn",
                "Prompt",
                "Çıktı",
                "Düşünme",
                "Toplam",
            ],
            _summary_rows(
                baseline,
                results,
            ),
            right_aligned={
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            },
        ),
        "",
        "Kesin optimum rotaları",
        _baseline_route_table(
            baseline,
            problem=problem,
        ),
    ]

    failed_results = []

    for result in results:
        status = result.get("status")
        validation = result.get("validation")
        label = (
            f"{result.get('provider', '-')} / "
            f"{result.get('model', '-')} / "
            f"{result.get('encoding', '-')}"
        )

        if status != "completed" or validation is None:
            error = result.get("error") or {}
            failed_results.append(
                [
                    str(result.get("provider", "-")),
                    str(result.get("model", "-")),
                    str(result.get("encoding", "-")),
                    str(status or "-"),
                    str(error.get("message", "-")),
                ]
            )
            continue

        lines.extend(
            [
                "",
                f"{label} rotaları",
                _route_table(
                    validation.get("routes") or [],
                    capacity=capacity,
                ),
                (
                    "Kısıt özeti: "
                    f"eksik={_format_ids(validation.get('missing_customer_ids'))} | "
                    f"tekrar={_format_ids(validation.get('duplicated_customer_ids'))} | "
                    f"bilinmeyen={_format_ids(validation.get('unknown_node_ids'))} | "
                    "kapasite aşımı="
                    f"{validation.get('total_capacity_excess', '-')} | "
                    "filo aşımı="
                    f"{'evet' if validation.get('fleet_limit_exceeded') else 'hayır'}"
                ),
            ]
        )

    if failed_results:
        lines.extend(
            [
                "",
                "Tamamlanamayan kayıtlar",
                render_table(
                    [
                        "Provider",
                        "Model",
                        "Kodlama",
                        "Aşama",
                        "Hata",
                    ],
                    failed_results,
                ),
            ]
        )

    if not results:
        lines.extend(
            [
                "",
                "Kayıtlı model sonucu bulunamadı.",
            ]
        )

    if image_paths:
        lines.extend(
            [
                "",
                "Görsel çıktılar",
                *[
                    "- "
                    + path.resolve().relative_to(
                        run_dir.resolve()
                    ).as_posix()
                    for path in image_paths
                ],
            ]
        )

    return "\n".join(lines) + "\n"


def main(
    argv: list[str] | None = None,
) -> None:
    args = parse_args(argv)
    run_id = _normalize_run_id(args.run_id)
    run_dir = (
        Path(args.output_dir)
        / "runs"
        / run_id
    )

    if not run_dir.is_dir():
        raise SystemExit(
            f"Run klasörü bulunamadı: {run_dir}"
        )

    image_paths = generate_route_images(
        run_dir
    )
    report = build_report(
        run_dir,
        image_paths=image_paths,
    )
    analysis_path = (
        run_dir
        / "analysis"
        / "terminal_analysis.txt"
    )
    analysis_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    analysis_path.write_text(
        report,
        encoding="utf-8",
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )

    print(report, end="")
    print(f"Rapor dosyası: {analysis_path}")


if __name__ == "__main__":
    main()
