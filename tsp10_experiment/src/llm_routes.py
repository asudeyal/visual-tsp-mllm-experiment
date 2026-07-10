"""Makaledeki görsel rota isteminin Gemini uyarlaması ve çıktı ayrıştırma."""

from __future__ import annotations

import os
import re
from pathlib import Path


GEMINI_MODEL = "gemini-2.5-flash"


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


def critic_prompt() -> str:
    """Makalede tek satıcı için kullanılan eleştirmen istemini üretir."""

    return """Inspect the current route shown in the provided image and propose an improved route for 1 salesman. The black square is the depot. Ensure that:

- All nodes are visited exactly once.
- The salesman starts from the depot and returns to the depot.
- Minimize intersections within the route.
- Make the route as short as possible.
- Aim to improve upon the current route shown in the image.

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


def image_mime_type(image_path: Path) -> str:
    """Dosya uzantısından Gemini'ye gönderilecek MIME türünü bulur."""

    suffix_to_mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = suffix_to_mime.get(image_path.suffix.lower())
    if mime is None:
        raise ValueError("Yalnızca PNG veya JPEG görseller desteklenir.")
    return mime


def request_gemini_route(
    image_path: Path,
    *,
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.0,
) -> str:
    """Bir görsel ve istemi Gemini'ye gönderip ham rota metnini döndürür.

    API anahtarı yalnızca ``GEMINI_API_KEY`` ortam değişkeninden okunur. Bu
    fonksiyon koordinatları modele göndermez; model yalnızca istemi ve görseli
    görür. Böylece makaledeki görsel akıl yürütme koşulu korunur.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY ortam değişkeni tanımlı değil.")

    # Import burada yapılır; rota ayrıştırma testleri API çağrısı yapmadan çalışır.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type=image_mime_type(image_path),
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=1000,
        ),
    )

    if not response.text or not response.text.strip():
        raise RuntimeError("Gemini boş veya metin içermeyen bir cevap döndürdü.")
    return response.text.strip()


def request_gemini_zero_shot_route(
    image_path: Path,
    *,
    model: str = GEMINI_MODEL,
    temperature: float = 0.0,
) -> str:
    """Nokta görselinden başlatıcı ajanın zero-shot rotasını ister."""

    return request_gemini_route(
        image_path,
        prompt=initializer_prompt(),
        model=model,
        temperature=temperature,
    )


def request_gemini_critic_route(
    image_path: Path,
    *,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
) -> str:
    """Mevcut rota görselinden eleştirmen ajanın yeni rotasını ister."""

    return request_gemini_route(
        image_path,
        prompt=critic_prompt(),
        model=model,
        temperature=temperature,
    )
