import pytest

from src.terminal_report import (
    compact_text,
    render_note,
    render_summary,
    render_table,
)


def test_render_table_uses_box_borders_and_formats_values() -> None:
    rendered = render_table(
        "Özet",
        ["Model", "Geçerli", "Mesafe"],
        [["model-a", True, 12.34567]],
        right_align={2},
    )
    assert "Özet" in rendered
    assert "┌" in rendered
    assert "┼" in rendered
    assert "┘" in rendered
    assert "evet" in rendered
    assert "12.3457" in rendered


def test_render_table_rejects_wrong_column_count() -> None:
    with pytest.raises(ValueError, match="sütun"):
        render_table("Hatalı", ["A", "B"], [[1]])


def test_compact_text_flattens_and_shortens_messages() -> None:
    assert compact_text("bir\niki", maximum=20) == "bir iki"
    assert compact_text("abcdefgh", maximum=5) == "abcd…"


def test_render_summary_wraps_fields_without_creating_table() -> None:
    rendered = render_summary(
        [
            ("geçerli", "8/10 (%80.0)"),
            ("en iyi", 12.34567),
            ("fallback", 2),
        ],
        fields_per_line=2,
    )
    assert "Özet: geçerli=8/10 (%80.0) | en iyi=12.3457" in rendered
    assert "fallback=2" in rendered
    assert "┌" not in rendered


def test_render_summary_rejects_zero_fields_per_line() -> None:
    with pytest.raises(ValueError, match="en az 1"):
        render_summary([], fields_per_line=0)


def test_render_note_uses_bullets_without_another_table() -> None:
    rendered = render_note(
        "Sürelerin yorumu",
        [
            "Aktif süre kontrollü beklemeyi içermez.",
            "API süresi uzak servisi de içerir.",
        ],
    )
    assert "Sürelerin yorumu" in rendered
    assert "• Aktif süre" in rendered
    assert "• API süresi" in rendered
    assert "┌" not in rendered
