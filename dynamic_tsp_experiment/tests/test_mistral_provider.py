from src.providers.registry import (
    create_provider,
    supported_providers,
)


def test_mistral_provider_is_registered() -> None:
    assert "mistral" in supported_providers()

    provider = create_provider(
        "mistral",
        None,
    )

    assert provider.provider_id == "mistral"
    assert (
        provider.model_alias
        == "mistral-small-latest"
    )
    assert provider.capabilities.supports_vision
    assert (
        provider.default_candidate_strategy
        == "independent_calls"
    )