"""Compact run persistence for AVMA-TSP.

The compact layout keeps all textual runtime evidence in one JSONL trace and one
mutable state file. Route images remain separate because they are first-class
visual experiment artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ..schemas import AgentCallRecord, CheckpointState


COMPACT_LAYOUT = "compact_v3"


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_atomic(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def relative_artifact(path: str | Path | None, model_dir: str | Path) -> str | None:
    if path is None:
        return None
    path = Path(path)
    root = Path(model_dir)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def resolve_artifact(value: str | None, model_dir: str | Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(model_dir) / path


class TraceStore:
    """Append-only JSONL trace with lightweight in-memory lookup."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    self.events.append(json.loads(line))
        self._next_seq = max((int(event.get("seq", 0)) for event in self.events), default=0) + 1

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        item = dict(event)
        item.setdefault("seq", self._next_seq)
        self._next_seq = max(self._next_seq, int(item["seq"]) + 1)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_json_dump(item) + "\n")
        self.events.append(item)
        return item

    def find_last(self, event_name: str, **filters: Any) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event.get("event") != event_name:
                continue
            if all(event.get(key) == value for key, value in filters.items()):
                return event
        return None

    def matching(self, event_name: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for event in self.events:
            if event_name is not None and event.get("event") != event_name:
                continue
            if all(event.get(key) == value for key, value in filters.items()):
                result.append(event)
        return result


def compact_call_record(call: AgentCallRecord) -> dict[str, Any]:
    """Serialize one call without repeating the frozen prompt text.

    Prompt bodies live once in the shared run.json. The trace keeps only the
    request's image/text labels, raw response, usage and provider metadata.
    """

    request_parts = [
        part
        for part in call.request_parts
        if not (part.get("kind") == "text" and part.get("label") == "instructions")
    ]
    return {
        "agent": call.agent,
        "prompt_ref": call.agent,
        "request_parts": request_parts,
        "raw_response": call.raw_response,
        "provider": call.provider_response.provider,
        "model": call.provider_response.model,
        "phase": call.provider_response.phase,
        "latency_seconds": call.provider_response.latency_seconds,
        "usage": call.provider_response.usage,
        "raw_metadata": call.provider_response.raw_metadata,
    }


def checkpoint_payload(state: CheckpointState) -> dict[str, Any]:
    return asdict(state)


def checkpoint_from_payload(data: dict[str, Any]) -> CheckpointState:
    return CheckpointState(**data)


def read_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return read_json(path) if path.exists() else {}


def update_state(path: str | Path, **updates: Any) -> dict[str, Any]:
    path = Path(path)
    state = read_state(path)
    state.update(updates)
    state.setdefault("layout_version", COMPACT_LAYOUT)
    write_json_atomic(path, state)
    return state


def _usage_total_tokens(usage: dict[str, Any]) -> int | None:
    """Normalize Gemini and OpenAI-compatible token usage fields."""
    for key in ("total_token_count", "total_tokens"):
        value = usage.get(key)
        if value is not None:
            return int(value)
    return None


def trace_api_metrics(events: Iterable[dict[str, Any]]) -> tuple[int, int | None, float | None, int]:
    """Return requests, tokens, active latency and recorded errors for trace events."""

    api_calls = 0
    total_tokens = 0
    saw_tokens = False
    active_seconds = 0.0
    latency_complete = True
    errors = 0

    for event in events:
        kind = event.get("event")
        if kind == "agent_call":
            call = event.get("call") or {}
            metadata = call.get("raw_metadata") or {}
            native_count = metadata.get("native_candidate_count")
            candidate_index = metadata.get("candidate_index")
            native_secondary = (
                isinstance(native_count, int)
                and native_count > 1
                and isinstance(candidate_index, int)
                and candidate_index > 1
            )
            if native_secondary:
                continue
            api_calls += 1
            usage = call.get("usage") or {}
            token_count = _usage_total_tokens(usage)
            if token_count is not None:
                total_tokens += int(token_count)
                saw_tokens = True
            latency = call.get("latency_seconds")
            if latency is None:
                latency_complete = False
            else:
                active_seconds += float(latency)
        elif kind == "model_output_attempt":
            api_calls += 1
            response = event.get("provider_response") or {}
            usage = response.get("usage") or {}
            token_count = _usage_total_tokens(usage)
            if token_count is not None:
                total_tokens += int(token_count)
                saw_tokens = True
            latency = response.get("latency_seconds")
            if latency is None:
                latency_complete = False
            else:
                active_seconds += float(latency)
            errors += 1
        elif kind == "provider_error":
            errors += 1

    tokens = total_tokens if saw_tokens else None
    active = active_seconds if api_calls > 0 and latency_complete else None
    return api_calls, tokens, active, errors


def trace_provider_wait_seconds(events: Iterable[dict[str, Any]]) -> float:
    """Sum deliberate provider throttling/backoff waits recorded in the trace."""

    total = 0.0
    for event in events:
        kind = event.get("event")
        if kind == "agent_call":
            call = event.get("call") or {}
            metadata = call.get("raw_metadata") or {}
            total += float(metadata.get("provider_wait_seconds") or 0.0)
        elif kind == "model_output_attempt":
            response = event.get("provider_response") or {}
            metadata = response.get("raw_metadata") or {}
            total += float(metadata.get("provider_wait_seconds") or 0.0)
        elif kind == "provider_error":
            total += float(event.get("provider_wait_seconds") or 0.0)
    return total
