"""Makaledeki görsel rota istemi, API çağrısı ve güvenli çıktı ayrıştırma."""

from __future__ import annotations

import base64
import re
from pathlib import Path


PAPER_MODEL = "gpt-4o"


class RouteParseError(ValueError):
    """Model cevabı beklenen TSP rota biçimine uymadığında oluşur."""


def initializer_prompt() -> str:
    """Makalede tek satıcı için kullanılan başlatıcı istemini üretir."""

    return """Inspect the provided image and find a route for 1 salesman starting from the depot, which is marked with a black square. Ensure that:

- All nodes are visited exactly once.
- The salesman starts from the depot and returns to the depot.
- Minimize intersections within the route.
- The route should be as short as possible.

Output the route in exactly the following format:
<<start>>
Salesman1: Depot-1-2-3-Depot
<<end>>

Replace the example node order with your proposed route. Do not include any additional explanation or text."""


def parse_single_salesman_route(model_text: str) -> list[int]:
    """Modelin ``Salesman1: Depot-...-Depot`` cevabını sayı listesine çevirir."""

    block = re.search(r"<<start>>(.*?)<<end>>", model_text, flags=re.DOTALL)
    if block is None:
        raise RouteParseError("Cevapta <<start>> ve <<end>> bloğu bulunamadı.")

    route_match = re.search(
        r"Salesman\s*1\s*:\s*(.+)", block.group(1), flags=re.IGNORECASE
    )
    if route_match is None:
        raise RouteParseError("Cevapta Salesman1 rota satırı bulunamadı.")

    route_text = route_match.group(1).strip()
    tokens = [token.strip() for token in route_text.split("-")]
    route: list[int] = []
    for token in tokens:
        if token.lower() in {"depot", "node0", "0"}:
            route.append(0)
            continue

        number_match = re.fullmatch(r"(?:node\s*)?(\d+)", token, re.IGNORECASE)
        if number_match is None:
            raise RouteParseError(f"Anlaşılamayan rota öğesi: {token!r}")
        route.append(int(number_match.group(1)))

    if len(route) < 2:
        raise RouteParseError("Rota en az başlangıç ve bitiş içermelidir.")
    return route


def _image_data_url(image_path: Path) -> str:
    suffix_to_mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = suffix_to_mime.get(image_path.suffix.lower())
    if mime is None:
        raise ValueError("Yalnızca PNG veya JPEG görseller desteklenir.")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def request_zero_shot_route(
    image_path: Path,
    *,
    model: str = PAPER_MODEL,
    temperature: float = 0.0,
) -> str:
    """Görseli modele gönderir ve işlenmemiş metin cevabını döndürür.

    OpenAI Python SDK, ``OPENAI_API_KEY`` ortam değişkenini otomatik okur.
    Makaleyi yeniden üretmek için Chat Completions ve ``gpt-4o`` korunmuştur.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    # Import burada yapılır; rota ayrıştırma testleri API paketi olmadan da çalışır.
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": initializer_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image_path)},
                    },
                ],
            }
        ],
    )

    text = completion.choices[0].message.content
    if not text:
        raise RuntimeError("Model boş bir cevap döndürdü.")
    return text
