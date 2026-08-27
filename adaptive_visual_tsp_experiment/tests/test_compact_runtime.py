from dataclasses import replace
from pathlib import Path

from src.config import load_config
from src.controller.orchestrator import AdaptiveVisualTSPOrchestrator
from src.prompts import PromptSet
from src.providers.base import ProviderAdapter
from src.schemas import ProblemInstance, ProviderCapabilities, ProviderResponse


class DeterministicProvider(ProviderAdapter):
    provider_id = "fake-compact"
    capabilities = ProviderCapabilities()

    def __init__(self):
        super().__init__("fake-model", request_retries=1)

    def _generate_once(self, parts, *, phase, temperature, thinking_level):
        if phase == "visual_scorer":
            text = '{"ranking":[1,2,3],"best_id":1}'
        else:
            text = '{"route":[1,2,3,4,1]}'
        return ProviderResponse(
            text=text,
            provider=self.provider_id,
            model=self.model,
            phase=phase,
            latency_seconds=0.1,
            usage={"total_token_count": 10},
        )


def _problem() -> ProblemInstance:
    return ProblemInstance(
        name="square4",
        dimension=4,
        node_ids=(1, 2, 3, 4),
        coordinates={1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0), 4: (0.0, 1.0)},
        depot=1,
        source_sha256="square",
        reference_optimum=4.0,
    )


def test_compact_runtime_keeps_only_state_trace_and_unique_routes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "pilot10_v1.yaml",
        provider_name="fake-compact",
        model="fake-model",
    )
    raw = dict(config.raw)
    raw["experiment"] = dict(raw["experiment"])
    raw["experiment"]["iterations"] = 1
    config = replace(config, iterations=1, raw=raw)
    run = tmp_path / "model"
    orchestrator = AdaptiveVisualTSPOrchestrator(
        config=config,
        problem=_problem(),
        provider=DeterministicProvider(),
        prompts=PromptSet(root / "prompts", "v1"),
        run_dir=run,
    )

    summary = orchestrator.run()

    assert summary["completed_iterations"] == 1
    assert (run / "state.json").exists()
    assert (run / "trace.jsonl").exists()
    assert (run / "routes" / "initializer" / "candidate.png").exists()
    assert (run / "routes" / "iteration_001" / "C1.png").exists()
    assert (run / "routes" / "iteration_001" / "C2.png").exists()
    assert (run / "routes" / "iteration_001" / "C3.png").exists()
    assert not (run / "initializer").exists()
    assert not (run / "iterations").exists()
    assert not (run / "checkpoint.json").exists()
    assert not (run / "summary.json").exists()
    assert not (run / "resume").exists()
