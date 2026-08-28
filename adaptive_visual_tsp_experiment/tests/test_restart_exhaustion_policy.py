from pathlib import Path
from types import SimpleNamespace

import pytest

from src.controller.orchestrator import AdaptiveVisualTSPOrchestrator


class _TraceStub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def find_last(self, event: str, **context):
        if event == "diversity_result":
            attempt = int(context["restart_attempt"])
            return {
                "event": event,
                "scope": context["scope"],
                "restart_attempt": attempt,
                "global_restart_count": attempt,
                "result": {"restart_attempt": attempt},
            }
        for item in reversed(self.events):
            if item.get("event") != event:
                continue
            if all(item.get(key) == value for key, value in context.items()):
                return item
        return None

    def append(self, event: dict) -> None:
        self.events.append(event)


def _failed_restart_orchestrator():
    orchestrator = AdaptiveVisualTSPOrchestrator.__new__(AdaptiveVisualTSPOrchestrator)
    orchestrator.config = SimpleNamespace(max_restart_attempts=3)
    orchestrator.trace = _TraceStub()
    orchestrator.restart_count = 0
    orchestrator.state_machine = SimpleNamespace(mark_restart=lambda: None)

    invalid_eval = SimpleNamespace(
        validation=SimpleNamespace(valid=False),
        to_dict=lambda: {"validation": {"valid": False}},
    )
    invalid_candidate = SimpleNamespace(route=(1, 2), image_path="invalid.png")

    orchestrator._last_call = lambda *args, **kwargs: None
    orchestrator._candidate_from_event = (
        lambda *args, **kwargs: (invalid_candidate, invalid_eval, Path("invalid.png"))
    )
    orchestrator._repair_until_valid = lambda *args, **kwargs: None
    return orchestrator


def test_restart_exhaustion_retains_valid_incumbent() -> None:
    orchestrator = _failed_restart_orchestrator()
    incumbent = SimpleNamespace(route=(1, 2, 1), image_path="working.png")
    incumbent_eval = SimpleNamespace(validation=SimpleNamespace(valid=True))

    candidate, evaluation, trace = orchestrator._restart_until_valid(
        scope="iteration_002.selected_restart",
        image_dir=Path("."),
        image_prefix="restart",
        resume=True,
        incumbent=(incumbent, incumbent_eval),
    )

    assert candidate is incumbent
    assert evaluation is incumbent_eval
    assert trace["exhausted"] is True
    assert trace["fallback_action"] == "retain_incumbent"
    assert trace["retained_route"] == [1, 2, 1]
    assert len([e for e in orchestrator.trace.events if e["event"] == "restart_exhausted"]) == 1


def test_restart_exhaustion_without_incumbent_still_hard_fails() -> None:
    orchestrator = _failed_restart_orchestrator()

    with pytest.raises(RuntimeError, match="Diversity Restart 3"):
        orchestrator._restart_until_valid(
            scope="initializer.fallback",
            image_dir=Path("."),
            image_prefix="restart",
            resume=True,
        )
