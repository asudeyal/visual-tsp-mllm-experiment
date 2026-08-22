"""Görsel CVRP modeli için prompt ve yanıt sözleşmesi."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any

from .rendering import DemandEncoding


class ModelResponseParseError(ValueError):
    """Model yanıtı beklenen JSON yapısında değil."""


@dataclass(frozen=True, slots=True)
class ParsedModelSolution:
    """Model yanıtından yapısal olarak çıkarılan rotalar."""

    routes: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "routes": [
                list(route)
                for route in self.routes
            ],
        }


_ENCODING_GUIDANCE = {
    DemandEncoding.NUMERIC: (
        "Each blue circle is a customer. "
        "The integer inside the circle is the customer "
        "node ID. The label d=<value> above the circle "
        "is that customer's demand."
    ),
}

_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _normalize_encoding(
    encoding: DemandEncoding | str,
) -> DemandEncoding:
    try:
        return DemandEncoding(encoding)
    except ValueError as error:
        supported = ", ".join(
            item.value
            for item in DemandEncoding
        )
        raise ValueError(
            "Desteklenmeyen talep gösterimi: "
            f"{encoding!r}. Desteklenenler: {supported}"
        ) from error


def build_solver_prompt(
    *,
    encoding: DemandEncoding | str = (
        DemandEncoding.NUMERIC
    ),
) -> str:
    """Tek çağrılık görsel CVRP çözüm prompt'unu oluştur."""

    normalized_encoding = _normalize_encoding(
        encoding
    )
    encoding_guidance = _ENCODING_GUIDANCE[
        normalized_encoding
    ]

    return (
        "Solve the Capacitated Vehicle Routing Problem "
        "shown in the image.\n\n"
        "Image interpretation:\n"
        "- The black square with node ID 0 is the depot.\n"
        f"- {encoding_guidance}\n"
        "- Vehicle capacity Q and the maximum number of "
        "available vehicles K are shown above the "
        "problem area.\n\n"
        "Solution requirements:\n"
        "- Every route must start at depot 0 and end at "
        "depot 0.\n"
        "- Visit every customer exactly once across all "
        "routes.\n"
        "- Do not invent, omit, or repeat customer IDs.\n"
        "- The sum of customer demands on each route "
        "must not exceed Q.\n"
        "- Use no more than K routes.\n"
        "- Minimize the sum of Euclidean route "
        "distances.\n\n"
        "Return exactly one JSON object with a field "
        'named "routes". The value of "routes" must be '
        "an array of integer arrays. Do not include "
        "markdown, commentary, calculations, or any "
        "text outside the JSON object."
    )


def _decode_json_payload(
    raw_response: str,
) -> Any:
    stripped_response = raw_response.strip()

    if not stripped_response:
        raise ModelResponseParseError(
            "Model yanıtı boş."
        )

    try:
        return json.loads(stripped_response)
    except JSONDecodeError:
        pass

    first_decoded_value: Any | None = None

    for fenced_content in (
        _JSON_FENCE_PATTERN.findall(
            stripped_response
        )
    ):
        try:
            decoded_value = json.loads(
                fenced_content
            )
        except JSONDecodeError:
            continue

        if first_decoded_value is None:
            first_decoded_value = decoded_value

        if (
            isinstance(decoded_value, dict)
            and "routes" in decoded_value
        ):
            return decoded_value

    decoder = JSONDecoder()

    for index, character in enumerate(
        stripped_response
    ):
        if character != "{":
            continue

        try:
            decoded_value, _ = decoder.raw_decode(
                stripped_response[index:]
            )
        except JSONDecodeError:
            continue

        if first_decoded_value is None:
            first_decoded_value = decoded_value

        if (
            isinstance(decoded_value, dict)
            and "routes" in decoded_value
        ):
            return decoded_value

    if first_decoded_value is not None:
        return first_decoded_value

    raise ModelResponseParseError(
        "Model yanıtında geçerli bir JSON nesnesi "
        "bulunamadı."
    )


def parse_model_response(
    raw_response: str,
) -> ParsedModelSolution:
    """Model yanıtından rota dizilerini çıkar."""

    payload = _decode_json_payload(
        raw_response
    )

    if not isinstance(payload, dict):
        raise ModelResponseParseError(
            "Model yanıtının JSON kökü bir nesne "
            "olmalıdır."
        )

    if "routes" not in payload:
        raise ModelResponseParseError(
            'Model yanıtında "routes" alanı bulunamadı.'
        )

    raw_routes = payload["routes"]

    if not isinstance(raw_routes, list):
        raise ModelResponseParseError(
            '"routes" alanı bir JSON dizisi olmalıdır.'
        )

    parsed_routes: list[tuple[int, ...]] = []

    for route_index, raw_route in enumerate(
        raw_routes,
        start=1,
    ):
        if not isinstance(raw_route, list):
            raise ModelResponseParseError(
                f"Rota {route_index} bir JSON dizisi "
                "olmalıdır."
            )

        parsed_node_ids: list[int] = []

        for position, raw_node_id in enumerate(
            raw_route,
            start=1,
        ):
            if (
                isinstance(raw_node_id, bool)
                or not isinstance(raw_node_id, int)
            ):
                raise ModelResponseParseError(
                    f"Rota {route_index}, konum "
                    f"{position}: düğüm kimliği tam sayı "
                    "olmalıdır."
                )

            parsed_node_ids.append(raw_node_id)

        parsed_routes.append(
            tuple(parsed_node_ids)
        )

    return ParsedModelSolution(
        routes=tuple(parsed_routes)
    )