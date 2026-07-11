"""Makaledeki görsel rota isteminin Gemini uyarlaması ve çıktı ayrıştırma."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence


GEMINI_MODEL = "gemini-2.5-flash"


class RouteParseError(ValueError):
    """Model cevabı beklenen TSP rota biçimine uymadığında oluşur."""


class ScorerParseError(ValueError):
    """Puanlayıcı cevabı beklenen görsel puanlama biçimine uymadığında oluşur."""


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


def scorer_prompt(image_ids: Sequence[int]) -> str:
    """Makalede kullanılan puanlayıcı istemini mevcut görsel kimliklerine uyarlar."""

    ids = [int(image_id) for image_id in image_ids]
    if not ids:
        raise ValueError("Puanlayıcı için en az bir görsel kimliği gerekir.")
    if len(set(ids)) != len(ids):
        raise ValueError("Görsel kimlikleri benzersiz olmalıdır.")

    score_example = ", ".join(f"image{image_id}: score" for image_id in ids)
    id_text = ", ".join(str(image_id) for image_id in ids)
    return f"""Examine the provided images, each representing a different solution for the same TSP. Evaluate each image against the following criteria to select the best solution:

1. Complete Node Coverage: Ensure all nodes are visited exactly once. Prefer routes that miss the fewest nodes.
2. Minimized Crossing Lines: Fewer crossing lines generally indicate a shorter total distance.
3. Route Clarity: The path should be easy to follow visually, with minimal overlapping lines.
4. Starting and Ending Point: The route should start and end at node 0.

The image IDs are {id_text}. Your entire response must contain exactly two lines. On the first line, output one numeric score for every image in exactly this format:
<<{score_example}>>

On the second line, select the best image and output its ID in exactly this format:
<<the best route: ID>>

Replace "score" and "ID" with numeric values. A higher score indicates a better solution. Use colons exactly as shown. Do not use Markdown, code fences, bullets, tables, JSON, or additional commentary. Do not calculate or request coordinates or distances."""


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


def parse_scorer_response(
    model_text: str,
    expected_image_ids: Sequence[int] | None = None,
) -> tuple[dict[int, float], int]:
    """Puanlayıcının görsel skorlarını ve seçtiği aday kimliğini ayrıştırır."""

    scores: dict[int, float] = {}
    score_patterns = [
        (
            r"[\"']?image\s*[_-]?\s*(\d+)[\"']?\s*"
            r"(?:score\s*)?(?::|=|->)\s*[\"']?(-?\d+(?:[.,]\d+)?)"
        ),
        (
            r"image\s*[_-]?\s*(\d+)\s*[-–]\s*score\s*"
            r"(?::|=)?\s*(-?\d+(?:[.,]\d+)?)"
        ),
        (
            r"\|\s*image\s*[_-]?\s*(\d+)\s*\|\s*"
            r"(-?\d+(?:[.,]\d+)?)"
        ),
    ]
    for pattern in score_patterns:
        for match in re.finditer(
            pattern,
            model_text,
            flags=re.IGNORECASE,
        ):
            image_id = int(match.group(1))
            score = float(match.group(2).replace(",", "."))
            if image_id in scores:
                if scores[image_id] != score:
                    raise ScorerParseError(
                        f"Puanlayıcı cevabında image{image_id} için çelişkili "
                        "skorlar var."
                    )
                continue
            scores[image_id] = score

    best_match = re.search(
        (
            r"(?:the\s+best\s+(?:route|image|candidate)|"
            r"best[_\s-]*(?:route|image|candidate)(?:[_\s-]*id)?|"
            r"en\s+iyi\s+(?:rota|görsel|aday))"
            r"\s*[\"']?\s*(?::|=|->)\s*[\"']?"
            r"(?:image\s*)?(\d+)"
        ),
        model_text,
        flags=re.IGNORECASE,
    )
    if not scores:
        raise ScorerParseError("Puanlayıcı cevabında görsel skorları bulunamadı.")
    if expected_image_ids is not None:
        expected = {int(image_id) for image_id in expected_image_ids}
        received = set(scores)
        if received != expected:
            missing = sorted(expected - received)
            unexpected = sorted(received - expected)
            raise ScorerParseError(
                "Puanlayıcı skor kimlikleri beklenen adaylarla uyuşmuyor. "
                f"Eksik={missing}, beklenmeyen={unexpected}."
            )
    if best_match is None:
        # Makaledeki kurala göre en yüksek skor en iyi çözümdür. Model bütün
        # skorları verdiği halde son seçim satırını atlarsa eşitlikte en küçük
        # görsel kimliğini seçerek aynı kuralı deterministik uygularız.
        highest_score = max(scores.values())
        best_image_id = min(
            image_id
            for image_id, score in scores.items()
            if score == highest_score
        )
    else:
        best_image_id = int(best_match.group(1))
    if best_image_id not in scores:
        raise ScorerParseError(
            f"Seçilen image{best_image_id} için bir skor bulunmuyor."
        )
    return scores, best_image_id


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


def request_gemini_route_candidates(
    image_path: Path,
    *,
    prompt: str,
    candidate_count: int,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
) -> list[str]:
    """Tek bir Gemini çağrısında birden fazla rota adayı üretir.

    Makaledeki Multi-Agent 1 self-ensemble adımı, sıcaklık 0.7 ve yedi adayla
    çalışır. ``candidate_count`` geliştirme testleri için daha küçük verilebilir.
    """

    if candidate_count < 1:
        raise ValueError("candidate_count en az 1 olmalıdır.")
    if not image_path.exists():
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY ortam değişkeni tanımlı değil.")

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
            candidate_count=candidate_count,
            max_output_tokens=1000,
        ),
    )

    texts: list[str] = []
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        text_parts = [
            str(part.text)
            for part in (parts or [])
            if getattr(part, "text", None)
        ]
        candidate_text = "\n".join(text_parts).strip()
        if candidate_text:
            texts.append(candidate_text)

    if not texts:
        raise RuntimeError(
            "Gemini critic çağrısı metin içeren hiçbir aday döndürmedi."
        )
    return texts


def request_gemini_scorer(
    image_paths: Sequence[Path],
    *,
    image_ids: Sequence[int] | None = None,
    model: str = GEMINI_MODEL,
    temperature: float = 0.0,
) -> str:
    """Aday rota görsellerini tek çağrıda Gemini puanlayıcıya gönderir."""

    paths = [Path(path) for path in image_paths]
    ids = (
        [int(image_id) for image_id in image_ids]
        if image_ids is not None
        else list(range(1, len(paths) + 1))
    )
    if not paths:
        raise ValueError("Puanlayıcı için en az bir rota görseli gerekir.")
    if len(paths) != len(ids):
        raise ValueError("Görsel yolları ile görsel kimliklerinin sayısı eşit olmalıdır.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Aday rota görseli bulunamadı: {path}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY ortam değişkeni tanımlı değil.")

    from google import genai
    from google.genai import types

    contents: list[object] = [scorer_prompt(ids)]
    for image_id, path in zip(ids, paths):
        contents.append(f"Image {image_id}:")
        contents.append(
            types.Part.from_bytes(
                data=path.read_bytes(),
                mime_type=image_mime_type(path),
            )
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            # Gemini 2.5 Flash düşünme tokenlarını da çıktı bütçesinden
            # kullanabilir. Yedi görseli puanlarken görünür cevabın yarıda
            # kesilmemesi için makaledeki 4000 token sınırı kullanılır.
            max_output_tokens=4000,
            thinking_config=types.ThinkingConfig(thinking_budget=512),
        ),
    )
    if not response.text or not response.text.strip():
        raise RuntimeError("Gemini puanlayıcı boş veya metin içermeyen cevap döndürdü.")
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


def request_gemini_critic_candidates(
    image_path: Path,
    *,
    candidate_count: int = 7,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
) -> list[str]:
    """Multi-Agent 1 için critic self-ensemble rota adaylarını ister."""

    return request_gemini_route_candidates(
        image_path,
        prompt=critic_prompt(),
        candidate_count=candidate_count,
        model=model,
        temperature=temperature,
    )
