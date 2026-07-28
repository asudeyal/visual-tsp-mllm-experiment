"""Bağımlılık gerektirmeyen, PowerShell uyumlu kutulu terminal tabloları."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def compact_text(value: Any, *, maximum: int = 32) -> str:
    if value is None:
        return "-"
    text = str(value).replace("\r", " ").replace("\n", " ")
    if len(text) <= maximum:
        return text
    return text[: max(1, maximum - 1)] + "…"


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "evet" if value else "hayır"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_table(
    title: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    right_align: set[int] | None = None,
    max_widths: dict[int, int] | None = None,
) -> str:
    values = [[_cell(value) for value in row] for row in rows]
    header_values = [str(value) for value in headers]
    if any(len(row) != len(header_values) for row in values):
        raise ValueError("Tablo satır ve başlık sütun sayıları eşit değil.")
    maximums = max_widths or {}
    for row in values:
        for index, value in enumerate(row):
            limit = maximums.get(index)
            if limit is not None:
                row[index] = compact_text(value, maximum=limit)
    widths = [
        max(
            len(header_values[index]),
            *(
                [len(row[index]) for row in values]
                or [0]
            ),
        )
        for index in range(len(header_values))
    ]

    def border(left: str, middle: str, right: str) -> str:
        return (
            left
            + middle.join("─" * (width + 2) for width in widths)
            + right
        )

    aligns = right_align or set()

    def format_row(row: Sequence[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            formatted = (
                value.rjust(widths[index])
                if index in aligns
                else value.ljust(widths[index])
            )
            cells.append(f" {formatted} ")
        return "│" + "│".join(cells) + "│"

    lines = [f"\n{title}", border("┌", "┬", "┐")]
    lines.append(format_row(header_values))
    lines.append(border("├", "┼", "┤"))
    if values:
        lines.extend(format_row(row) for row in values)
    else:
        empty = ["Kayıt yok.", *([""] * (len(headers) - 1))]
        lines.append(format_row(empty))
    lines.append(border("└", "┴", "┘"))
    return "\n".join(lines)
