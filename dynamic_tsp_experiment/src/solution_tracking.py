from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


GAP_KEYS = (
    "gap_to_reference_percent",
    "gap_percent",
    "gap_to_known_optimum_percent",
    "gap_to_exact_percent",
    "gap_to_or_tools_percent",
)

DISTANCE_KEYS = (
    "distance",
    "route_distance",
    "total_distance",
)

CANDIDATE_INDEX_KEYS = (
    "candidate_index",
    "candidate",
    "candidate_id",
)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
        return None

    converted = float(value)

    if not math.isfinite(converted):
        return None

    return converted


def _first_number(
    record: Mapping[str, Any],
    keys: Iterable[str],
) -> float | None:
    for key in keys:
        converted = _finite_float(record.get(key))

        if converted is not None:
            return converted

    return None


def _candidate_index(
    record: Mapping[str, Any],
) -> int | None:
    for key in CANDIDATE_INDEX_KEYS:
        value = record.get(key)

        if isinstance(value, bool):
            continue

        if isinstance(value, int):
            return value

        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())

    return None


def _route(
    record: Mapping[str, Any],
) -> tuple[int, ...] | None:
    value = record.get("route")

    if not isinstance(value, (list, tuple)):
        return None

    route: list[int] = []

    for node in value:
        if isinstance(node, bool):
            return None

        if isinstance(node, int):
            route.append(node)
            continue

        if isinstance(node, str):
            normalized = node.strip()

            if normalized.lstrip("-").isdigit():
                route.append(int(normalized))
                continue

        return None

    return tuple(route)


def _is_valid(
    record: Mapping[str, Any],
) -> bool:
    validation = record.get("validation")

    if isinstance(validation, Mapping):
        validation_value = validation.get("is_valid")

        if isinstance(validation_value, bool):
            return validation_value

    direct_value = record.get("is_valid")

    if isinstance(direct_value, bool):
        return direct_value

    return False


@dataclass(frozen=True)
class SolutionObservation:
    source: str
    iteration: int
    candidate_index: int | None
    route: tuple[int, ...] | None
    distance: float | None
    gap_to_reference_percent: float | None
    is_valid: bool

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        source: str,
        iteration: int,
        candidate_index: int | None = None,
    ) -> SolutionObservation:
        resolved_candidate_index = (
            candidate_index
            if candidate_index is not None
            else _candidate_index(record)
        )

        return cls(
            source=str(source),
            iteration=int(iteration),
            candidate_index=resolved_candidate_index,
            route=_route(record),
            distance=_first_number(
                record,
                DISTANCE_KEYS,
            ),
            gap_to_reference_percent=_first_number(
                record,
                GAP_KEYS,
            ),
            is_valid=_is_valid(record),
        )

    @property
    def is_comparable(self) -> bool:
        return (
            self.is_valid
            and self.distance is not None
            and self.distance >= 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "iteration": self.iteration,
            "candidate_index": self.candidate_index,
            "route": (
                list(self.route)
                if self.route is not None
                else None
            ),
            "distance": self.distance,
            "gap_to_reference_percent": (
                self.gap_to_reference_percent
            ),
            "is_valid": self.is_valid,
        }


def best_valid_observation(
    observations: Iterable[SolutionObservation],
) -> SolutionObservation | None:
    best: SolutionObservation | None = None

    for observation in observations:
        if not observation.is_comparable:
            continue

        if best is None:
            best = observation
            continue

        assert observation.distance is not None
        assert best.distance is not None

        if observation.distance < best.distance:
            best = observation

    return best


def selection_regret_percent(
    selected: SolutionObservation | None,
    iteration_best: SolutionObservation | None,
) -> float | None:
    if (
        selected is None
        or iteration_best is None
        or not selected.is_comparable
        or not iteration_best.is_comparable
    ):
        return None

    assert selected.distance is not None
    assert iteration_best.distance is not None

    if iteration_best.distance <= 0:
        return None

    regret = (
        (selected.distance - iteration_best.distance)
        / iteration_best.distance
    ) * 100.0

    # Küçük floating-point farkları negatif değer üretmesin.
    return max(0.0, regret)


def selected_is_iteration_best(
    selected: SolutionObservation | None,
    iteration_best: SolutionObservation | None,
    *,
    absolute_tolerance: float = 1e-9,
) -> bool | None:
    if iteration_best is None:
        return None

    if selected is None or not selected.is_comparable:
        return False

    assert selected.distance is not None
    assert iteration_best.distance is not None

    return math.isclose(
        selected.distance,
        iteration_best.distance,
        rel_tol=1e-9,
        abs_tol=absolute_tolerance,
    )


@dataclass(frozen=True)
class EarlyStopPolicy:
    enabled: bool = True
    threshold_percent: float = 1.0
    allowed_providers: tuple[str, ...] = (
        "gemini",
        "groq",
    )
    require_proven_optimum: bool = True
    allow_heuristic_reference: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.threshold_percent)
            or self.threshold_percent < 0
        ):
            raise ValueError(
                "threshold_percent sonlu ve negatif olmayan "
                "bir sayı olmalıdır."
            )


class SolutionProgressTracker:
    def __init__(
        self,
        *,
        provider: str,
        reference_distance: float | None,
        reference_type: str | None,
        reference_is_proven_optimal: bool,
        early_stop_policy: EarlyStopPolicy | None = None,
    ) -> None:
        normalized_reference_distance = _finite_float(
            reference_distance
        )

        if (
            normalized_reference_distance is not None
            and normalized_reference_distance <= 0
        ):
            normalized_reference_distance = None

        self.provider = str(provider).strip().casefold()
        self.reference_distance = normalized_reference_distance
        self.reference_type = reference_type
        self.reference_is_proven_optimal = bool(
            reference_is_proven_optimal
        )
        self.early_stop_policy = (
            early_stop_policy or EarlyStopPolicy()
        )

        self._system_gbest: SolutionObservation | None = None
        self._observed_candidate_gbest: (
            SolutionObservation | None
        ) = None
        self._initializer: SolutionObservation | None = None
        self._history: list[dict[str, Any]] = []
        self._latest_early_stop = self._evaluate_early_stop()

    @property
    def system_gbest(self) -> SolutionObservation | None:
        return self._system_gbest

    @property
    def observed_candidate_gbest(
        self,
    ) -> SolutionObservation | None:
        return self._observed_candidate_gbest

    @property
    def should_stop(self) -> bool:
        return bool(
            self._latest_early_stop.get("should_stop")
        )

    @property
    def latest_early_stop(self) -> dict[str, Any]:
        return deepcopy(self._latest_early_stop)

    def _calculated_gap(
        self,
        observation: SolutionObservation,
    ) -> float | None:
        if observation.gap_to_reference_percent is not None:
            return observation.gap_to_reference_percent

        if (
            observation.distance is None
            or self.reference_distance is None
            or self.reference_distance <= 0
        ):
            return None

        return (
            (
                observation.distance
                - self.reference_distance
            )
            / self.reference_distance
        ) * 100.0

    @staticmethod
    def _better(
        candidate: SolutionObservation,
        current: SolutionObservation | None,
    ) -> bool:
        if not candidate.is_comparable:
            return False

        if current is None or not current.is_comparable:
            return True

        assert candidate.distance is not None
        assert current.distance is not None

        return candidate.distance < current.distance

    def _update_system_gbest(
        self,
        observation: SolutionObservation | None,
    ) -> None:
        if observation is None:
            return

        if self._better(observation, self._system_gbest):
            self._system_gbest = observation

    def _update_observed_candidate_gbest(
        self,
        observation: SolutionObservation,
    ) -> None:
        if self._better(
            observation,
            self._observed_candidate_gbest,
        ):
            self._observed_candidate_gbest = observation

    def _evaluate_early_stop(self) -> dict[str, Any]:
        policy = self.early_stop_policy
        gbest = self._system_gbest

        base = {
            "enabled": policy.enabled,
            "eligible": False,
            "should_stop": False,
            "reason": None,
            "threshold_percent": policy.threshold_percent,
            "provider": self.provider,
            "reference_type": self.reference_type,
            "reference_is_proven_optimal": (
                self.reference_is_proven_optimal
            ),
            "system_gbest_source": (
                gbest.source
                if gbest is not None
                else None
            ),
            "system_gbest_iteration": (
                gbest.iteration
                if gbest is not None
                else None
            ),
            "system_gbest_gap_percent": None,
        }

        if not policy.enabled:
            base["reason"] = "disabled"
            return base

        allowed_providers = {
            provider.strip().casefold()
            for provider in policy.allowed_providers
        }

        if self.provider not in allowed_providers:
            base["reason"] = "provider_not_enabled"
            return base

        if (
            policy.require_proven_optimum
            and not self.reference_is_proven_optimal
            and not policy.allow_heuristic_reference
        ):
            base["reason"] = "reference_not_proven_optimal"
            return base

        base["eligible"] = True

        if gbest is None or not gbest.is_comparable:
            base["reason"] = "valid_system_gbest_missing"
            return base

        gap = self._calculated_gap(gbest)
        base["system_gbest_gap_percent"] = gap

        if gap is None:
            base["reason"] = "gap_unavailable"
            return base

        # Kanıtlanmış optimumdan anlamlı biçimde daha kısa sonuç,
        # metrik/referans uyuşmazlığına işaret eder.
        if gap < -1e-9:
            base["reason"] = "negative_gap_inconsistent"
            return base

        if gap <= policy.threshold_percent:
            base["should_stop"] = True
            base["reason"] = "gap_threshold_reached"
            return base

        base["reason"] = "gap_above_threshold"
        return base

    def seed_initializer(
        self,
        solution: Mapping[str, Any],
        *,
        source: str = "zero_shot",
        iteration: int = 0,
    ) -> dict[str, Any]:
        observation = SolutionObservation.from_mapping(
            solution,
            source=source,
            iteration=iteration,
        )

        self._initializer = observation
        self._update_system_gbest(observation)
        self._latest_early_stop = self._evaluate_early_stop()

        return {
            "initializer": observation.to_dict(),
            "system_gbest": (
                self._system_gbest.to_dict()
                if self._system_gbest is not None
                else None
            ),
            "early_stop": self.latest_early_stop,
        }

    def record_iteration(
        self,
        *,
        iteration: int,
        selected_solution: Mapping[str, Any] | None,
        candidates: Iterable[Mapping[str, Any]],
        selected_source: str = "selected",
        candidate_source: str = "critic",
    ) -> dict[str, Any]:
        candidate_observations = [
            SolutionObservation.from_mapping(
                candidate,
                source=candidate_source,
                iteration=iteration,
                candidate_index=index,
            )
            for index, candidate in enumerate(
                candidates,
                start=1,
            )
        ]

        iteration_best = best_valid_observation(
            candidate_observations
        )

        for observation in candidate_observations:
            self._update_observed_candidate_gbest(observation)

        selected_observation = (
            SolutionObservation.from_mapping(
                selected_solution,
                source=selected_source,
                iteration=iteration,
            )
            if selected_solution is not None
            else None
        )

        self._update_system_gbest(selected_observation)
        self._latest_early_stop = self._evaluate_early_stop()

        record = {
            "iteration": int(iteration),
            "valid_candidate_count": sum(
                1
                for observation in candidate_observations
                if observation.is_comparable
            ),
            "candidate_count": len(candidate_observations),
            "selected_solution": (
                selected_observation.to_dict()
                if selected_observation is not None
                else None
            ),
            "iteration_best": (
                iteration_best.to_dict()
                if iteration_best is not None
                else None
            ),
            "system_gbest": (
                self._system_gbest.to_dict()
                if self._system_gbest is not None
                else None
            ),
            "observed_candidate_gbest": (
                self._observed_candidate_gbest.to_dict()
                if self._observed_candidate_gbest is not None
                else None
            ),
            "selection_regret_percent": (
                selection_regret_percent(
                    selected_observation,
                    iteration_best,
                )
            ),
            "selected_is_iteration_best": (
                selected_is_iteration_best(
                    selected_observation,
                    iteration_best,
                )
            ),
            "early_stop": self.latest_early_stop,
        }

        self._history.append(deepcopy(record))
        return record

    def record_multi_agent2_iteration(
        self,
        *,
        iteration: int,
        solution: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.record_iteration(
            iteration=iteration,
            selected_solution=solution,
            candidates=[solution],
            selected_source="critic",
            candidate_source="critic",
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "initializer": (
                self._initializer.to_dict()
                if self._initializer is not None
                else None
            ),
            "system_gbest": (
                self._system_gbest.to_dict()
                if self._system_gbest is not None
                else None
            ),
            "observed_candidate_gbest": (
                self._observed_candidate_gbest.to_dict()
                if self._observed_candidate_gbest is not None
                else None
            ),
            "early_stop": self.latest_early_stop,
            "history": deepcopy(self._history),
        }