from pathlib import Path

import pytest

from src.config import load_config
from src.controller.orchestrator import AdaptiveVisualTSPOrchestrator
from src.prompts import PromptSet
from src.providers.base import ProviderAdapter
from src.schemas import ProblemInstance, ProviderCapabilities


class NeverCalledProvider(ProviderAdapter):
    provider_id = "fake"
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=False,
        supports_temperature=True,
        supports_thinking_level=False,
        max_native_choices=1,
    )

    def __init__(self):
        super().__init__("fake-model", request_retries=3)
        self.calls = 0

    def _generate_once(self, parts, *, phase, temperature, thinking_level):
        self.calls += 1
        raise AssertionError("cached phase should not call provider")


class SDKRetryProvider(ProviderAdapter):
    provider_id = "sdk-retry"
    sdk_managed_retries = True
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=False,
        supports_temperature=True,
        supports_thinking_level=False,
        max_native_choices=1,
    )

    def __init__(self):
        super().__init__("fake-model", request_retries=5)
        self.calls = 0

    def _generate_once(self, parts, *, phase, temperature, thinking_level):
        self.calls += 1
        raise RuntimeError("transient")


def _square_problem() -> ProblemInstance:
    return ProblemInstance(
        name="square4",
        dimension=4,
        node_ids=(1, 2, 3, 4),
        coordinates={1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0), 4: (0.0, 1.0)},
        depot=1,
        source_sha256="square",
        reference_optimum=4.0,
    )


def test_initializer_raw_response_is_reused_on_resume(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "pilot10_v1.yaml",
        provider_name="fake",
        model="fake-model",
    )
    provider = NeverCalledProvider()
    orchestrator = AdaptiveVisualTSPOrchestrator(
        config=config,
        problem=_square_problem(),
        provider=provider,
        prompts=PromptSet(project_root / "prompts", "v1"),
        run_dir=tmp_path / "run",
    )

    orchestrator.trace.append(
        {
            "event": "agent_call",
            "agent": "initializer",
            "call": {"raw_response": '{"route":[1,2,3,4,1]}'},
        }
    )

    candidate, evaluation, trace = orchestrator._initial_route(resume=True)

    assert provider.calls == 0
    assert candidate.route == (1, 2, 3, 4, 1)
    assert evaluation.validation.valid
    assert trace["resumed"] is True
    assert orchestrator.trace.find_last("initializer_candidate") is not None
    assert orchestrator.trace.find_last("initializer_result") is not None


def test_sdk_managed_retry_provider_is_not_double_retried():
    provider = SDKRetryProvider()
    with pytest.raises(RuntimeError, match="transient"):
        provider.generate([], phase="test")
    assert provider.calls == 1


class InvalidStructuredOutputProvider(ProviderAdapter):
    provider_id = "fake-invalid-output"
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=False,
        supports_temperature=True,
        supports_thinking_level=False,
        max_native_choices=1,
    )

    def __init__(self):
        super().__init__("fake-model", request_retries=1)
        self.calls = 0

    def _generate_once(self, parts, *, phase, temperature, thinking_level):
        from src.schemas import ProviderResponse

        self.calls += 1
        return ProviderResponse(
            text='{"route":["1","node 2","3","4","1"]}',
            provider=self.provider_id,
            model=self.model,
            phase=phase,
            latency_seconds=0.25,
            usage={"total_token_count": 17},
        )


def test_failed_model_output_keeps_every_raw_attempt(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "pilot10_v1.yaml",
        provider_name="fake-invalid-output",
        model="fake-model",
    )
    provider = InvalidStructuredOutputProvider()
    orchestrator = AdaptiveVisualTSPOrchestrator(
        config=config,
        problem=_square_problem(),
        provider=provider,
        prompts=PromptSet(project_root / "prompts", "v1"),
        run_dir=tmp_path / "run",
    )

    with pytest.raises(Exception, match="integer"):
        orchestrator._initial_route(resume=False)

    assert provider.calls == config.initializer.max_output_retries + 1
    attempts = orchestrator.trace.matching("model_output_attempt")
    assert len(attempts) == provider.calls
    for attempt in attempts:
        assert attempt["raw_response"] == '{"route":["1","node 2","3","4","1"]}'
        assert "integer" in attempt["error_message"]
        assert attempt["provider_response"]["usage"]["total_token_count"] == 17
    assert orchestrator.trace.find_last("model_output_failure") is not None
