from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.base import ModelOutputError, parse_hybrid, parse_route, parse_scorer
from src.controller.orchestrator import AdaptiveVisualTSPOrchestrator


class _TraceStub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def find_last(self, event: str, **context):
        for item in reversed(self.events):
            if item.get("event") != event:
                continue
            if all(item.get(key) == value for key, value in context.items()):
                return item
        return None

    def append(self, event: dict) -> None:
        self.events.append(event)


def test_route_parser_accepts_valid_json_with_trailing_extra_data() -> None:
    raw = '{"route":[1,2,3,1]}\n{"note":"extra model text"}'
    assert parse_route(raw) == (1, 2, 3, 1)


def test_parsers_find_the_object_with_the_required_shape() -> None:
    route_raw = 'analysis {"note":"draft"} final {"route":["1","2","1"]} thanks'
    assert parse_route(route_raw) == (1, 2, 1)

    scorer_raw = '{"note":"draft"}\n{"ranking":["2","1"],"best_id":"2"} trailing'
    assert parse_scorer(scorer_raw, {1, 2}) == ([2, 1], 2)

    hybrid_raw = (
        '{"comment":"draft"} '
        '{"selected_edges":[["1","2"],["3","4"]],"route":["1","3","2","4","1"]}'
    )
    route, edges = parse_hybrid(hybrid_raw)
    assert route == (1, 3, 2, 4, 1)
    assert edges == ((1, 2), (3, 4))


def test_recoverable_model_output_failure_is_rejected_without_raising() -> None:
    orchestrator = AdaptiveVisualTSPOrchestrator.__new__(AdaptiveVisualTSPOrchestrator)
    orchestrator.trace = _TraceStub()
    orchestrator._invoke_with_error_record = lambda *args, **kwargs: (_ for _ in ()).throw(
        ModelOutputError("bad json")
    )

    result = orchestrator._invoke_recoverable(
        {"iteration": 2},
        "visual_scorer",
        lambda: None,
    )

    assert result is None
    event = orchestrator.trace.find_last(
        "recoverable_agent_failure",
        phase="visual_scorer",
        iteration=2,
    )
    assert event is not None
    assert event["error_type"] == "ModelOutputError"


def test_provider_timeout_pauses_instead_of_becoming_stage_failure() -> None:
    class ReadTimeout(Exception):
        pass

    orchestrator = AdaptiveVisualTSPOrchestrator.__new__(AdaptiveVisualTSPOrchestrator)
    orchestrator.trace = _TraceStub()
    orchestrator._invoke_with_error_record = lambda *args, **kwargs: (_ for _ in ()).throw(
        ReadTimeout("slow model")
    )

    with pytest.raises(ReadTimeout, match="slow model"):
        orchestrator._invoke_recoverable({}, "repair_attempt_01", lambda: None)


def test_provider_429_pauses_instead_of_becoming_stage_failure() -> None:
    class TooManyRequests(Exception):
        status_code = 429

    orchestrator = AdaptiveVisualTSPOrchestrator.__new__(AdaptiveVisualTSPOrchestrator)
    orchestrator.trace = _TraceStub()
    orchestrator._invoke_with_error_record = lambda *args, **kwargs: (_ for _ in ()).throw(
        TooManyRequests("quota exhausted")
    )

    with pytest.raises(TooManyRequests, match="quota exhausted"):
        orchestrator._invoke_recoverable(
            {"iteration": 5, "candidate": 2},
            "critic_candidate_02",
            lambda: None,
        )


def test_nonrecoverable_programming_error_still_raises() -> None:
    orchestrator = AdaptiveVisualTSPOrchestrator.__new__(AdaptiveVisualTSPOrchestrator)
    orchestrator.trace = _TraceStub()
    orchestrator._invoke_with_error_record = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("broken invariant")
    )

    with pytest.raises(ValueError, match="broken invariant"):
        orchestrator._invoke_recoverable({}, "critic_candidate_01", lambda: None)


def test_restart_agent_failures_retain_valid_incumbent() -> None:
    orchestrator = AdaptiveVisualTSPOrchestrator.__new__(AdaptiveVisualTSPOrchestrator)
    orchestrator.config = SimpleNamespace(max_restart_attempts=3)
    orchestrator.trace = _TraceStub()
    orchestrator.restart_count = 0
    orchestrator.problem = SimpleNamespace(node_ids=(1, 2))
    orchestrator.problem_image = Path("problem.png")
    orchestrator.diversity = SimpleNamespace(run=lambda *args, **kwargs: None)
    orchestrator.state_machine = SimpleNamespace(mark_restart=lambda: None)
    orchestrator._last_call = lambda *args, **kwargs: None
    orchestrator._invoke_recoverable = lambda *args, **kwargs: None

    incumbent = SimpleNamespace(route=(1, 2, 1), image_path="working.png")
    incumbent_eval = SimpleNamespace(validation=SimpleNamespace(valid=True))

    candidate, evaluation, trace = orchestrator._restart_until_valid(
        scope="iteration_002.selected_restart",
        image_dir=Path("."),
        image_prefix="restart",
        incumbent=(incumbent, incumbent_eval),
    )

    assert candidate is incumbent
    assert evaluation is incumbent_eval
    assert trace["fallback_action"] == "retain_incumbent"
    assert len(trace["attempts"]) == 3
    assert all(item["status"] == "agent_failure" for item in trace["attempts"])
