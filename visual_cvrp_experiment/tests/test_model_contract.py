"""Model prompt ve yanıt sözleşmesinin testleri."""

from __future__ import annotations

import pytest

from src.model_contract import (
    ModelResponseParseError,
    build_solver_prompt,
    parse_model_response,
)


def test_numeric_prompt_describes_constraints() -> None:
    prompt = build_solver_prompt(
        encoding="numeric"
    )

    assert "customer node ID" in prompt
    assert "d=<value>" in prompt
    assert "exactly once" in prompt
    assert "must not exceed Q" in prompt
    assert "no more than K routes" in prompt
    assert '"routes"' in prompt

    # Prompt gerçek problem taleplerini veya çözümü
    # metin olarak vermemelidir.
    assert "total demand is 18" not in prompt.lower()
    assert "419.727" not in prompt


def test_size_prompt_explains_visual_scale() -> None:
    prompt = build_solver_prompt(
        encoding="size"
    )

    assert "encoded only by circle area" in prompt
    assert "smallest circles have demand 1" in prompt
    assert "medium circles have demand 2" in prompt
    assert "largest circles have demand 3" in prompt
    assert "d=<value>" not in prompt
    assert "total demand is 18" not in prompt.lower()
    assert "419.727" not in prompt


def test_parse_direct_json_response() -> None:
    parsed = parse_model_response(
        '{"routes": [[0, 1, 2, 0], [0, 3, 0]]}'
    )

    assert parsed.routes == (
        (0, 1, 2, 0),
        (0, 3, 0),
    )


def test_parse_json_code_fence() -> None:
    parsed = parse_model_response(
        """
        ```json
        {
          "routes": [
            [0, 1, 0],
            [0, 2, 3, 0]
          ]
        }
        ```
        """
    )

    assert parsed.routes == (
        (0, 1, 0),
        (0, 2, 3, 0),
    )


def test_parse_json_surrounded_by_text() -> None:
    parsed = parse_model_response(
        """
        Here is the solution:
        {"routes": [[0, 4, 5, 0]]}
        This is my final answer.
        """
    )

    assert parsed.routes == (
        (0, 4, 5, 0),
    )


def test_empty_routes_are_preserved_for_validator() -> None:
    parsed = parse_model_response(
        '{"routes": []}'
    )

    assert parsed.routes == ()

    # Parser çözümü düzeltmez veya geçerlilik
    # kararı vermez.
    assert parsed.to_dict() == {
        "routes": [],
    }


def test_missing_routes_field_is_rejected() -> None:
    with pytest.raises(
        ModelResponseParseError,
        match="routes",
    ):
        parse_model_response(
            '{"answer": [[0, 1, 0]]}'
        )


def test_route_must_be_an_array() -> None:
    with pytest.raises(
        ModelResponseParseError,
        match="Rota 1",
    ):
        parse_model_response(
            '{"routes": ["0-1-0"]}'
        )


@pytest.mark.parametrize(
    "invalid_node_id",
    [
        1.0,
        True,
    ],
)
def test_node_ids_must_be_integers(
    invalid_node_id: object,
) -> None:
    response = (
        '{"routes": [[0, '
        + json_value(invalid_node_id)
        + ', 0]]}'
    )

    with pytest.raises(
        ModelResponseParseError,
        match="tam sayı",
    ):
        parse_model_response(response)


def json_value(value: object) -> str:
    if value is True:
        return "true"
    return str(value)


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(
        ModelResponseParseError,
        match="JSON",
    ):
        parse_model_response(
            "I could not solve this problem."
        )
