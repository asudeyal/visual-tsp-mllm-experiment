import pytest

from src.terminal_report import compact_text, render_table


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
