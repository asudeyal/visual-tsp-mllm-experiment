from src.experiment_metrics import extract_usage_metadata, summarize_api_calls
from src.llm_routes import _generate_content_observed


class FakeUsage:
    prompt_token_count = 100
    candidates_token_count = 20
    thoughts_token_count = 5
    cached_content_token_count = 0
    total_token_count = 125


class FakeResponse:
    usage_metadata = FakeUsage()


class SuccessfulModels:
    def generate_content(self, **_: object) -> FakeResponse:
        return FakeResponse()


class SuccessfulClient:
    models = SuccessfulModels()


class FailingModels:
    def generate_content(self, **_: object) -> FakeResponse:
        raise RuntimeError("quota")


class FailingClient:
    models = FailingModels()


def test_extract_usage_metadata() -> None:
    assert extract_usage_metadata(FakeResponse()) == {
        "prompt_token_count": 100,
        "candidates_token_count": 20,
        "thoughts_token_count": 5,
        "cached_content_token_count": 0,
        "total_token_count": 125,
    }


def test_observed_api_call_records_success() -> None:
    response, record = _generate_content_observed(
        client=SuccessfulClient(),
        model="test-model",
        contents=["prompt"],
        config=object(),
        phase="critic",
        temperature=0.7,
        input_image_count=1,
        input_image_bytes=500,
    )

    assert isinstance(response, FakeResponse)
    assert record["success"] is True
    assert record["phase"] == "critic"
    assert record["api_call_wall_seconds"] >= 0
    assert record["usage"]["total_token_count"] == 125


def test_observed_api_call_attaches_failed_measurement() -> None:
    try:
        _generate_content_observed(
            client=FailingClient(),
            model="test-model",
            contents=["prompt"],
            config=object(),
            phase="scorer",
            temperature=0.0,
            input_image_count=7,
            input_image_bytes=3500,
        )
    except RuntimeError as exc:
        record = exc.gemini_call_record
    else:
        raise AssertionError("Başarısız istem RuntimeError oluşturmalıydı.")

    assert record["success"] is False
    assert record["phase"] == "scorer"
    assert record["input_image_count"] == 7


def test_summarize_api_calls() -> None:
    summary = summarize_api_calls(
        [
            {
                "success": True,
                "api_call_wall_seconds": 1.5,
                "usage": {"total_token_count": 100},
            },
            {
                "success": False,
                "api_call_wall_seconds": 0.5,
                "usage": {"total_token_count": None},
            },
        ]
    )

    assert summary["api_call_count"] == 2
    assert summary["successful_api_call_count"] == 1
    assert summary["failed_api_call_count"] == 1
    assert summary["total_api_call_wall_seconds"] == 2.0
    assert summary["total_token_count"] == 100
