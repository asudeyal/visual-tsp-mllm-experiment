import pytest

from run_analysis import (
    _print_ma1_iterations,
    _print_resources,
)
from src.terminal_report import (
    compact_text,
    render_note,
    render_summary,
    render_table,
)


def test_render_table_uses_box_borders_and_formats_values() -> None:
    rendered = render_table(
        "Özet",
        ["Model", "Geçerli", "Mesafe"],
        [["model-a", True, 12.34567]],
        right_align={2},
    )
    assert "Özet" in rendered
    assert "┌" in rendered
    assert "┼" in rendered
    assert "┘" in rendered
    assert "evet" in rendered
    assert "12.3457" in rendered


def test_render_table_rejects_wrong_column_count() -> None:
    with pytest.raises(ValueError, match="sütun"):
        render_table("Hatalı", ["A", "B"], [[1]])


def test_compact_text_flattens_and_shortens_messages() -> None:
    assert compact_text("bir\niki", maximum=20) == "bir iki"
    assert compact_text("abcdefgh", maximum=5) == "abcd…"


def test_render_summary_wraps_fields_without_creating_table() -> None:
    rendered = render_summary(
        [
            ("geçerli", "8/10 (%80.0)"),
            ("en iyi", 12.34567),
            ("fallback", 2),
        ],
        fields_per_line=2,
    )
    assert "Özet: geçerli=8/10 (%80.0) | en iyi=12.3457" in rendered
    assert "fallback=2" in rendered
    assert "┌" not in rendered


def test_render_summary_rejects_zero_fields_per_line() -> None:
    with pytest.raises(ValueError, match="en az 1"):
        render_summary([], fields_per_line=0)


def test_render_note_uses_bullets_without_another_table() -> None:
    rendered = render_note(
        "Sürelerin yorumu",
        [
            "Aktif süre kontrollü beklemeyi içermez.",
            "API süresi uzak servisi de içerir.",
        ],
    )
    assert "Sürelerin yorumu" in rendered
    assert "• Aktif süre" in rendered
    assert "• API süresi" in rendered
    assert "┌" not in rendered


def test_ma1_iteration_table_shows_gbest_gap_without_duplicate_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    section = {
        "iterations": [
            {
                "iteration": 1,
                "returned_candidate_count": 2,
                "valid_candidate_count": 2,
                "selection_mode": (
                    "visual_scorer_after_feasibility_filter"
                ),
                "selected_candidate_id": 2,
                "selected_distance": 12.0,
                "iteration_best_distance": 10.0,
                "system_gbest_distance": 11.0,
                "system_gbest_gap_percent": 0.75,
                "observed_candidate_gbest_distance": 10.0,
                "selection_regret_percent": 20.0,
                "selected_best_valid_candidate": False,
                "token_count": {"total": 120},
                "timing_seconds": {
                    "active": 1.0,
                    "deliberate_delay": 2.0,
                    "rate_limit_backoff": 3.0,
                    "total": 6.0,
                },
            }
        ],
        "valid_candidate_count": 2,
        "total_candidate_count": 2,
        "scorer_evaluated_iteration_count": 1,
        "scorer_best_candidate_selection_count": 0,
        "solution_progress": {
            "system_gbest": {
                "distance": 11.0,
                "gap_to_reference_percent": 0.75,
                "iteration": 0,
            },
            "observed_candidate_gbest": {
                "distance": 10.0,
                "gap_to_reference_percent": 0.25,
                "iteration": 1,
            },
        },
        "termination": {
            "reason": "requested_iterations_completed",
        },
    }

    _print_ma1_iterations("gemini / test-model", section)
    rendered = capsys.readouterr().out

    assert "GBest gap %" in rendered
    assert "Doğru seçim" not in rendered
    assert "0.7500" in rendered
    assert "GBest iter.=0" in rendered
    assert "aday GBest gap %=0.2500" in rendered
    assert "aday GBest iter.=1" in rendered


def test_resource_table_removes_profile_and_retry_and_hides_missing_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis = {
        "methods": {
            "baseline": {
                "observability": {
                    "resources": {
                        "enabled": True,
                        "sample_count": 8,
                        "local_gpu": {
                            "available": False,
                            "unavailable_reason": (
                                "NVMLError_LibraryNotFound"
                            ),
                        },
                        "overall_metrics": {
                            "system_cpu_percent": {
                                "average": 5.0,
                                "maximum": 10.0,
                            },
                            "process_cpu_percent": {
                                "average": 50.0,
                                "maximum": 90.0,
                            },
                            "process_memory_rss_mb": {
                                "average": 80.0,
                                "maximum": 100.0,
                            },
                            "system_memory_percent": {
                                "average": 70.0,
                                "maximum": 75.0,
                            },
                        },
                    },
                    "request_control": {
                        "retry_count": 2,
                    },
                }
            }
        },
        "provider_models": [],
    }

    _print_resources(analysis)
    rendered = capsys.readouterr().out

    assert "│ Ölçüm" in rendered
    assert "│ Profil" not in rendered
    assert "│ Retry" not in rendered
    assert "│ Yerel GPU" not in rendered
    assert rendered.count("NVIDIA/NVML yok") == 1


def test_resource_table_shows_gpu_column_when_measurement_exists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis = {
        "methods": {
            "baseline": {
                "observability": {
                    "resources": {
                        "enabled": True,
                        "sample_count": 4,
                        "local_gpu": {
                            "available": True,
                        },
                        "overall_metrics": {
                            "local_gpu_utilization_percent": {
                                "average": 25.0,
                                "maximum": 50.0,
                            },
                        },
                    }
                }
            }
        },
        "provider_models": [],
    }

    _print_resources(analysis)
    rendered = capsys.readouterr().out

    assert "│ Yerel GPU" in rendered
    assert "ölçüldü 25.0/50.0%" in rendered
