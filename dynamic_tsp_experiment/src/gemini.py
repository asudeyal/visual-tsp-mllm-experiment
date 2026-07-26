"""Dinamik TSP problemleri için Gemini görsel istemleri ve API çağrıları."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.metrics import (
    api_call_record,
    elapsed_seconds,
    start_timer,
    usage_metadata,
    utc_now_iso,
)
from src.problem_instance import ProblemInstance


GEMINI_MODEL = "gemini-2.5-flash"


class RouteParseError(ValueError):
    pass


class ScorerParseError(ValueError):
    pass


@dataclass(frozen=True)
class GeminiTextResult:
    text: str
    api_call: dict[str, Any]


@dataclass(frozen=True)
class GeminiCandidatesResult:
    texts: list[str]
    api_call: dict[str, Any]


def _node_description(
    node_ids: Sequence[int],
) -> str:
    ids = sorted(int(node_id) for node_id in node_ids)
    if ids == list(range(ids[0], ids[-1] + 1)):
        return f"{ids[0]} through {ids[-1]}"
    return ", ".join(str(node_id) for node_id in ids)


def _non_depot_description(
    problem: ProblemInstance,
) -> str:
    return _node_description(
        [
            node_id
            for node_id in problem.node_ids
            if node_id != problem.depot_id
        ]
    )


def _route_format_example(
    problem: ProblemInstance,
) -> str:
    visits = [
        node_id
        for node_id in problem.node_ids
        if node_id != problem.depot_id
    ][:3]
    middle = "-".join(str(node_id) for node_id in visits)
    return (
        f"Salesman1: Depot-{middle}-Depot"
        if middle
        else "Salesman1: Depot-Depot"
    )


def initializer_prompt(
    problem: ProblemInstance,
) -> str:
    all_nodes = _node_description(problem.node_ids)
    visit_nodes = _non_depot_description(problem)
    example = _route_format_example(problem)
    return f"""Inspect the provided image and find a short route for one salesman.

Problem information:
- Problem name: {problem.name}
- The image contains exactly {problem.dimension} nodes labelled {all_nodes}.
- Node {problem.depot_id}, marked with a black square, is the depot.

Important rules:
- Start at the depot, visit every non-depot node exactly once, and return to the depot.
- The required non-depot node IDs are: {visit_nodes}.
- Do not omit, repeat, invent, or renumber nodes.
- Minimize crossings and total route length.
- Use only visual information from the image; coordinates are not provided.

Output exactly one route and no explanation:
<<start>>
{example}
<<end>>

The displayed route is only a format example. Replace its order with every required
non-depot node. The word Depot means node {problem.depot_id}."""


def critic_prompt(
    problem: ProblemInstance,
) -> str:
    all_nodes = _node_description(problem.node_ids)
    visit_nodes = _non_depot_description(problem)
    example = _route_format_example(problem)
    return f"""Inspect the current route shown in the image and propose an improved route.

Problem information:
- Problem name: {problem.name}
- The image contains exactly {problem.dimension} nodes labelled {all_nodes}.
- Node {problem.depot_id}, marked with a black square, is the depot.

Important rules:
- Start at the depot, visit every non-depot node exactly once, and return to the depot.
- The required non-depot node IDs are: {visit_nodes}.
- Do not omit, repeat, invent, or renumber nodes.
- Prefer fewer crossings and a visually shorter route than the displayed route.
- Use only visual information from the image; coordinates are not provided.

Output exactly one route and no explanation:
<<start>>
{example}
<<end>>

The displayed route is only a format example. Replace its order with every required
non-depot node. The word Depot means node {problem.depot_id}."""


def scorer_prompt(
    problem: ProblemInstance,
    image_ids: Sequence[int],
) -> str:
    ids = [int(image_id) for image_id in image_ids]
    score_example = ", ".join(
        f"image{image_id}: score"
        for image_id in ids
    )
    return f"""Compare the provided route images and select the best visual solution.

Problem information:
- Problem name: {problem.name}
- Each image represents the same {problem.dimension}-node TSP problem.
- Node {problem.depot_id}, marked with a black square, is the depot.

Use these criteria:
1. Every non-depot node must be covered exactly once.
2. The route must start and end at depot node {problem.depot_id}.
3. Prefer fewer crossings, less overlap, and a visually shorter route.
4. Do not request coordinates or numeric distances.

Your entire response must contain exactly two lines:
<<{score_example}>>
<<the best route: ID>>

Use one score for every image. Higher is better. Do not add commentary."""


def parse_route(
    model_text: str,
    *,
    depot_id: int = 1,
) -> list[int]:
    block = re.search(
        r"<<start>>(.*?)<<end>>",
        model_text,
        flags=re.DOTALL | re.I,
    )
    searchable = block.group(1) if block else model_text
    match = re.search(
        r"Salesman\s*1\s*:\s*(.+)",
        searchable,
        flags=re.I,
    )
    if match is None:
        raise RouteParseError(
            "Salesman1 rota satırı bulunamadı."
        )
    route_text = (
        match.group(1)
        .strip()
        .replace("->", "-")
        .replace("→", "-")
    )
    tokens = [
        token.strip()
        for token in re.split(r"\s*[-,]\s*", route_text)
    ]
    route: list[int] = []
    for token in tokens:
        normalized = token.lower().strip()
        if normalized in {
            "depot",
            f"node{depot_id}",
            f"node {depot_id}",
        }:
            route.append(int(depot_id))
            continue
        number = re.fullmatch(
            r"(?:node\s*)?(\d+)",
            token,
            flags=re.I,
        )
        if number is None:
            raise RouteParseError(
                f"Anlaşılamayan rota öğesi: {token!r}"
            )
        route.append(int(number.group(1)))
    if len(route) < 2:
        raise RouteParseError(
            "Rota başlangıç ve bitiş içermelidir."
        )
    return route


def parse_scorer_response(
    model_text: str,
    *,
    expected_image_ids: Sequence[int],
) -> tuple[dict[int, float], int]:
    scores: dict[int, float] = {}
    for match in re.finditer(
        (
            r"image\s*[_-]?\s*(\d+)\s*"
            r"(?:score\s*)?(?::|=|->)\s*"
            r"(-?\d+(?:[.,]\d+)?)"
        ),
        model_text,
        flags=re.I,
    ):
        scores[int(match.group(1))] = float(
            match.group(2).replace(",", ".")
        )
    expected = {
        int(value)
        for value in expected_image_ids
    }
    if set(scores) != expected:
        raise ScorerParseError(
            "Scorer kimlikleri uyuşmuyor; "
            f"beklenen={sorted(expected)}, "
            f"alınan={sorted(scores)}."
        )
    best_match = re.search(
        (
            r"(?:the\s+best\s+(?:route|image)|"
            r"best[_\s-]*(?:route|image)"
            r"(?:[_\s-]*id)?)\s*"
            r"(?::|=|->)\s*(?:image\s*)?(\d+)"
        ),
        model_text,
        flags=re.I,
    )
    if best_match:
        best_id = int(best_match.group(1))
    else:
        best_score = max(scores.values())
        best_id = min(
            image_id
            for image_id, score in scores.items()
            if score == best_score
        )
    if best_id not in expected:
        raise ScorerParseError(
            f"Geçersiz scorer seçimi: {best_id}"
        )
    return scores, best_id


def _mime_type(path: Path) -> str:
    value = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower())
    if value is None:
        raise ValueError(
            "Yalnız PNG/JPEG görseller desteklenir."
        )
    return value


def _client_and_types() -> tuple[Any, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY ortam değişkeni tanımlı değil."
        )
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key), types


def _observed_call(
    *,
    client: Any,
    model: str,
    contents: list[Any],
    config: Any,
    phase: str,
    temperature: float,
    image_count: int,
    image_bytes: int,
) -> tuple[Any, dict[str, Any]]:
    started_at = utc_now_iso()
    timer = start_timer()
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as exc:
        call = api_call_record(
            phase=phase,
            model=model,
            temperature=temperature,
            started_at_utc=started_at,
            wall_seconds=elapsed_seconds(timer),
            success=False,
            input_image_count=image_count,
            input_image_bytes=image_bytes,
        )
        try:
            setattr(exc, "gemini_call_record", call)
        except Exception:
            pass
        raise
    call = api_call_record(
        phase=phase,
        model=model,
        temperature=temperature,
        started_at_utc=started_at,
        wall_seconds=elapsed_seconds(timer),
        success=True,
        input_image_count=image_count,
        input_image_bytes=image_bytes,
        usage=usage_metadata(response),
    )
    return response, call


def request_route(
    image_path: Path,
    *,
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float,
    phase: str,
) -> GeminiTextResult:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Görsel bulunamadı: {image_path}"
        )
    client, types = _client_and_types()
    image_data = image_path.read_bytes()
    response, call = _observed_call(
        client=client,
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_data,
                mime_type=_mime_type(image_path),
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(
                thinking_budget=2048
            ),
        ),
        phase=phase,
        temperature=temperature,
        image_count=1,
        image_bytes=len(image_data),
    )
    text = (response.text or "").strip()
    if not text:
        exc = RuntimeError(
            "Gemini boş rota cevabı döndürdü."
        )
        exc.gemini_call_record = call
        raise exc
    return GeminiTextResult(text=text, api_call=call)


def request_candidates(
    image_path: Path,
    *,
    problem: ProblemInstance,
    candidate_count: int,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
) -> GeminiCandidatesResult:
    if not 1 <= candidate_count <= 7:
        raise ValueError(
            "candidate-count 1 ile 7 arasında olmalıdır."
        )
    if not image_path.exists():
        raise FileNotFoundError(
            f"Görsel bulunamadı: {image_path}"
        )
    client, types = _client_and_types()
    image_data = image_path.read_bytes()
    response, call = _observed_call(
        client=client,
        model=model,
        contents=[
            critic_prompt(problem),
            types.Part.from_bytes(
                data=image_data,
                mime_type=_mime_type(image_path),
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            candidate_count=candidate_count,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(
                thinking_budget=2048
            ),
        ),
        phase="critic_candidate_generation",
        temperature=temperature,
        image_count=1,
        image_bytes=len(image_data),
    )
    texts: list[str] = []
    for candidate in response.candidates or []:
        parts = (
            getattr(
                getattr(candidate, "content", None),
                "parts",
                None,
            )
            or []
        )
        value = "\n".join(
            str(part.text)
            for part in parts
            if getattr(part, "text", None)
        ).strip()
        if value:
            texts.append(value)
    if not texts:
        exc = RuntimeError(
            "Gemini critic hiçbir rota adayı döndürmedi."
        )
        exc.gemini_call_record = call
        raise exc
    return GeminiCandidatesResult(
        texts=texts,
        api_call=call,
    )


def request_scorer(
    image_paths: Sequence[Path],
    *,
    problem: ProblemInstance,
    image_ids: Sequence[int],
    model: str = GEMINI_MODEL,
) -> GeminiTextResult:
    paths = [Path(path) for path in image_paths]
    ids = [int(image_id) for image_id in image_ids]
    if not paths or len(paths) != len(ids):
        raise ValueError(
            "Scorer görselleri ile kimliklerinin "
            "sayısı eşit olmalıdır."
        )
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Scorer görseli bulunamadı: {missing[0]}"
        )
    client, types = _client_and_types()
    contents: list[Any] = [
        scorer_prompt(problem, ids)
    ]
    total_bytes = 0
    for image_id, path in zip(ids, paths):
        image_data = path.read_bytes()
        total_bytes += len(image_data)
        contents.extend(
            [
                f"Image {image_id}:",
                types.Part.from_bytes(
                    data=image_data,
                    mime_type=_mime_type(path),
                ),
            ]
        )
    response, call = _observed_call(
        client=client,
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(
                thinking_budget=512
            ),
        ),
        phase="visual_scorer",
        temperature=0.0,
        image_count=len(paths),
        image_bytes=total_bytes,
    )
    text = (response.text or "").strip()
    if not text:
        exc = RuntimeError(
            "Gemini scorer boş cevap döndürdü."
        )
        exc.gemini_call_record = call
        raise exc
    return GeminiTextResult(text=text, api_call=call)
