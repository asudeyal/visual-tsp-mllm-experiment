"""AVMA-CVRP orchestration with compact, resumable experiment persistence."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

from ..agents import CriticAgent, DiversityAgent, HybridAgent, InitializerAgent, RepairAgent, ScorerAgent
from ..agents.base import ModelOutputError, parse_hybrid, parse_cvrp_routes, parse_scorer
from ..config import ExperimentConfig
from ..evaluation import evaluate_cvrp_routes
from ..experiment.compact import (
    TraceStore,
    checkpoint_from_payload,
    checkpoint_payload,
    compact_call_record,
    read_state,
    relative_artifact,
    resolve_artifact,
    update_state,
)
from ..providers.base import ProviderAdapter
from ..prompts import PromptSet
from ..rendering import render_problem, render_routes
from ..schemas import CheckpointState, ObserverEvaluation, ProblemInstance, RouteCandidate
from ..search import (
    canonicalize_routes,
    detect_structural_stagnation,
    edge_similarity,
    is_exact_two_opt_transition,
    undirected_edge_set,
)
from .state_machine import EscapeStateMachine


T = TypeVar("T")


class AdaptiveVisualCVRPOrchestrator:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        problem: ProblemInstance,
        provider: ProviderAdapter,
        prompts: PromptSet,
        run_dir: str | Path,
        problem_image: str | Path | None = None,
    ) -> None:
        self.config = config
        self.problem = problem
        self.provider = provider
        self.prompts = prompts
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.routes_dir = self.run_dir / "routes"
        self.routes_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.trace = TraceStore(self.run_dir / "trace.jsonl")

        self.initializer = InitializerAgent(provider, prompts, config.initializer)
        self.critic = CriticAgent(provider, prompts, config.critic)
        self.scorer = ScorerAgent(provider, prompts, config.scorer)
        self.repair = RepairAgent(provider, prompts, config.repair)
        self.hybrid = HybridAgent(provider, prompts, config.hybrid)
        self.diversity = DiversityAgent(provider, prompts, config.diversity)

        self.state_machine = EscapeStateMachine()
        self.structural_history: list[tuple[tuple[int, ...], ...]] = []
        self.restart_count = 0
        self.observed_oracle_best_distance: float | None = None
        self.observed_oracle_best_route: tuple[tuple[int, ...], ...] | None = None
        self.selected_best_distance: float | None = None
        self.selected_best_route: tuple[tuple[int, ...], ...] | None = None
        self.problem_image = (
            Path(problem_image)
            if problem_image is not None
            else self.run_dir / "problem.png"
        )
        if not self.problem_image.exists():
            render_problem(self.problem, self.problem_image, self.config.render, demand_encoding=self.config.demand_encoding)

    @property
    def allowed_node_ids(self) -> set[int]:
        return set(self.problem.node_ids)

    @staticmethod
    def _status_code(error: Exception) -> int | str | None:
        for attribute in ("status_code", "code"):
            value = getattr(error, attribute, None)
            if value is not None and not callable(value):
                return value
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
        if value is not None:
            return value
        return None

    def _context(self, **values: Any) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}

    def _record_output_attempts(self, context: dict[str, Any], attempts: Any) -> None:
        if not attempts:
            return
        for index, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                continue
            self.trace.append(
                {
                    "event": "model_output_attempt",
                    **context,
                    "output_attempt": int(attempt.get("attempt") or index),
                    "error_type": attempt.get("error_type"),
                    "error_message": attempt.get("error_message"),
                    "raw_response": str(attempt.get("raw_response") or ""),
                    "provider_response": {
                        "phase": attempt.get("phase"),
                        "provider": attempt.get("provider"),
                        "model": attempt.get("model"),
                        "latency_seconds": attempt.get("latency_seconds"),
                        "usage": attempt.get("usage") or {},
                        "raw_metadata": attempt.get("raw_metadata") or {},
                    },
                }
            )

    def _record_call(self, context: dict[str, Any], call) -> None:
        self._record_output_attempts(context, call.failed_output_attempts)
        self.trace.append(
            {
                "event": "agent_call",
                **context,
                "agent": call.agent,
                "call": compact_call_record(call),
            }
        )

    def _last_call(self, agent: str, **context: Any) -> dict[str, Any] | None:
        for event in reversed(self.trace.events):
            if event.get("event") != "agent_call" or event.get("agent") != agent:
                continue
            if all(event.get(key) == value for key, value in context.items()):
                return event
        return None

    def _invoke_with_error_record(
        self,
        context: dict[str, Any],
        phase: str,
        operation: Callable[[], T],
    ) -> T:
        update_state(
            self.state_path,
            status="running",
            current={"phase": phase, **context},
            last_error=None,
        )
        try:
            return operation()
        except Exception as exc:
            if isinstance(exc, ModelOutputError):
                self._record_output_attempts(context, exc.attempts)
                event_name = "model_output_failure"
            else:
                event_name = "provider_error"
            status = self._status_code(exc)
            wait_metadata = getattr(exc, "_avma_wait_metadata", {}) or {}
            error = {
                "phase": phase,
                **context,
                "provider": self.config.provider.name,
                "model": self.config.provider.model,
                "error_type": type(exc).__name__,
                "status_code": status,
                "message": str(exc),
                "transient_http_status": status in {408, 429, 500, 502, 503, 504},
                "request_delay_wait_seconds": wait_metadata.get("request_delay_wait_seconds", 0.0),
                "retry_backoff_wait_seconds": wait_metadata.get("retry_backoff_wait_seconds", 0.0),
                "provider_wait_seconds": wait_metadata.get("provider_wait_seconds", 0.0),
            }
            self.trace.append({"event": event_name, **error})
            update_state(
                self.state_path,
                status="partial",
                current={"phase": phase, **context},
                last_error=error,
            )
            raise

    def _is_recoverable_agent_failure(self, error: Exception) -> bool:
        return isinstance(error, ModelOutputError)

    def _invoke_recoverable(
        self,
        context: dict[str, Any],
        phase: str,
        operation: Callable[[], T],
        *,
        resume: bool = False,
    ) -> T | None:
        cached_failure = None
        if resume:
            cached_failure = self.trace.find_last(
                "recoverable_agent_failure",
                phase=phase,
                **context,
            )
            if cached_failure is None:
                cached_failure = self.trace.find_last(
                    "model_output_failure",
                    phase=phase,
                    **context,
                )
        if cached_failure is not None:
            return None

        try:
            return self._invoke_with_error_record(context, phase, operation)
        except Exception as exc:
            if not self._is_recoverable_agent_failure(exc):
                raise
            self.trace.append(
                {
                    "event": "recoverable_agent_failure",
                    "phase": phase,
                    **context,
                    "error_type": type(exc).__name__,
                    "status_code": self._status_code(exc),
                    "message": str(exc),
                }
            )
            return None

    def _observe(self, route: tuple[tuple[int, ...], ...]) -> ObserverEvaluation:
        evaluation = evaluate_cvrp_routes(self.problem, route)
        if evaluation.validation.valid and evaluation.distance is not None:
            if self.observed_oracle_best_distance is None or evaluation.distance < self.observed_oracle_best_distance:
                self.observed_oracle_best_distance = evaluation.distance
                self.observed_oracle_best_route = route
        return evaluation

    def _observe_selected(self, route: tuple[tuple[int, ...], ...], evaluation: ObserverEvaluation) -> None:
        if evaluation.validation.valid and evaluation.distance is not None:
            if self.selected_best_distance is None or evaluation.distance < self.selected_best_distance:
                self.selected_best_distance = evaluation.distance
                self.selected_best_route = route

    def _render_and_observe(
        self,
        candidate: RouteCandidate,
        image_path: Path,
    ) -> tuple[RouteCandidate, ObserverEvaluation]:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        render_routes(self.problem, candidate.routes, image_path, self.config.render, demand_encoding=self.config.demand_encoding, route_rendering=self.config.route_rendering)
        candidate = replace(candidate, image_path=str(image_path))
        evaluation = self._observe(candidate.routes)
        return candidate, evaluation

    def _candidate_from_event(
        self,
        event: dict[str, Any],
        *,
        source: str,
        candidate_id: int,
        route_key: str,
        image_key: str = "image",
        raw_text: str = "",
    ) -> tuple[RouteCandidate, ObserverEvaluation, Path]:
        routes = tuple(tuple(int(node) for node in r) for r in event[route_key])
        image = resolve_artifact(event.get(image_key), self.run_dir)
        if image is None:
            raise ValueError(f"Compact trace image alanı eksik: {image_key}")
        candidate = RouteCandidate(candidate_id, routes, source, raw_text, str(image))
        if not image.exists():
            render_routes(
                self.problem, 
                routes, 
                image, 
                self.config.render,
                demand_encoding=self.config.demand_encoding,
                route_rendering=self.config.route_rendering
            )
        evaluation = self._observe(routes)
        return candidate, evaluation, image

    def _repair_until_valid(
        self,
        candidate: RouteCandidate,
        candidate_image: Path,
        *,
        scope: str,
        image_dir: Path,
        image_prefix: str,
        resume: bool = False,
    ) -> tuple[RouteCandidate, ObserverEvaluation, list[dict[str, Any]]] | None:
        traces: list[dict[str, Any]] = []
        current = candidate
        current_image = candidate_image

        for attempt in range(1, self.config.repair.max_attempts + 1):
            context = self._context(scope=scope, attempt=attempt)
            image_path = image_dir / f"{image_prefix}_{attempt:02d}.png"
            cached = self.trace.find_last("repair_result", scope=scope, attempt=attempt) if resume else None

            if cached is not None:
                call = self._last_call("repair", scope=scope, attempt=attempt)
                raw_text = ((call or {}).get("call") or {}).get("raw_response", "")
                repaired, evaluation, image_path = self._candidate_from_event(
                    cached,
                    source="repair_resume",
                    candidate_id=1,
                    route_key="output_route",
                    raw_text=raw_text,
                )
                trace = dict(cached.get("result") or {})
            else:
                cached_call = self._last_call("repair", scope=scope, attempt=attempt) if resume else None
                if cached_call is not None:
                    raw_text = (cached_call.get("call") or {}).get("raw_response", "")
                    repaired = RouteCandidate(1, parse_cvrp_routes(raw_text), "repair_resume_raw", raw_text)
                    repaired, evaluation = self._render_and_observe(repaired, image_path)
                else:
                    result = self._invoke_recoverable(
                        context,
                        f"repair_attempt_{attempt:02d}",
                        lambda: self.repair.run(
                            self.problem_image,
                            current_image,
                            allowed_node_ids=self.allowed_node_ids,
                            attempt=attempt,
                        ),
                        resume=resume,
                    )
                    if result is None:
                        trace = {
                            "attempt": attempt,
                            "input_route": [list(r) for r in current.routes],
                            "output_route": None,
                            "evaluation": None,
                            "status": "agent_failure",
                        }
                        traces.append(trace)
                        continue
                    self._record_call(context, result.call)
                    repaired, evaluation = self._render_and_observe(result.candidate, image_path)

                trace = {
                    "attempt": attempt,
                    "input_route": [list(r) for r in current.routes],
                    "output_route": [list(r) for r in repaired.routes],
                    "evaluation": evaluation.to_dict(),
                }
                self.trace.append(
                    {
                        "event": "repair_result",
                        **context,
                        "output_route": [list(r) for r in repaired.routes],
                        "evaluation": evaluation.to_dict(),
                        "image": relative_artifact(repaired.image_path, self.run_dir),
                        "result": trace,
                    }
                )

            traces.append(trace)
            if evaluation.validation.valid:
                return repaired, evaluation, traces
            current = repaired
            current_image = Path(repaired.image_path or image_path)
        return None

    def _restart_until_valid(
        self,
        *,
        scope: str,
        image_dir: Path,
        image_prefix: str,
        resume: bool = False,
        incumbent: tuple[RouteCandidate, ObserverEvaluation] | None = None,
    ) -> tuple[RouteCandidate, ObserverEvaluation, dict[str, Any]]:
        restart_trace: dict[str, Any] = {"attempts": []}
        for restart_attempt in range(1, self.config.max_restart_attempts + 1):
            context = self._context(scope=scope, restart_attempt=restart_attempt)
            cached = self.trace.find_last(
                "diversity_result", scope=scope, restart_attempt=restart_attempt
            ) if resume else None
            image_path = image_dir / f"{image_prefix}_{restart_attempt:02d}.png"

            if cached is not None:
                self.restart_count = max(self.restart_count, int(cached.get("global_restart_count") or 0))
                call = self._last_call("diversity", scope=scope, restart_attempt=restart_attempt)
                raw_text = ((call or {}).get("call") or {}).get("raw_response", "")
                candidate, evaluation, image_path = self._candidate_from_event(
                    cached,
                    source="diversity_resume",
                    candidate_id=1,
                    route_key="route",
                    raw_text=raw_text,
                )
                item = dict(cached.get("result") or {})
            else:
                cached_failure = (
                    self.trace.find_last(
                        "recoverable_agent_failure",
                        phase="diversity_restart",
                        **context,
                    )
                    if resume
                    else None
                )
                if cached_failure is not None:
                    restart_trace["attempts"].append(
                        {
                            "restart_attempt": restart_attempt,
                            "global_restart_count": self.restart_count,
                            "route": None,
                            "evaluation": None,
                            "status": "agent_failure",
                        }
                    )
                    continue

                self.restart_count += 1
                cached_call = self._last_call("diversity", scope=scope, restart_attempt=restart_attempt) if resume else None
                if cached_call is not None:
                    raw_text = (cached_call.get("call") or {}).get("raw_response", "")
                    candidate = RouteCandidate(1, parse_cvrp_routes(raw_text), "diversity_resume_raw", raw_text)
                    candidate, evaluation = self._render_and_observe(candidate, image_path)
                else:
                    result = self._invoke_recoverable(
                        context,
                        "diversity_restart",
                        lambda: self.diversity.run(
                            self.problem_image,
                            allowed_node_ids=self.allowed_node_ids,
                        ),
                        resume=resume,
                    )
                    if result is None:
                        restart_trace["attempts"].append(
                            {
                                "restart_attempt": restart_attempt,
                                "global_restart_count": self.restart_count,
                                "route": None,
                                "evaluation": None,
                                "status": "agent_failure",
                            }
                        )
                        continue
                    self._record_call(context, result.call)
                    candidate, evaluation = self._render_and_observe(result.candidate, image_path)

                item = {
                    "restart_attempt": restart_attempt,
                    "global_restart_count": self.restart_count,
                    "route": [list(r) for r in candidate.routes],
                    "evaluation": evaluation.to_dict(),
                }
                self.trace.append(
                    {
                        "event": "diversity_result",
                        **context,
                        "global_restart_count": self.restart_count,
                        "route": [list(r) for r in candidate.routes],
                        "evaluation": evaluation.to_dict(),
                        "image": relative_artifact(candidate.image_path, self.run_dir),
                        "result": item,
                    }
                )

            if evaluation.validation.valid:
                # DIVERSITY KONTROLÜ: Eğer incumbent (mevcut iyi rota) ile yeni rota tamamen aynıysa 
                # bu diversity (çeşitlilik) sağlamaz, bu yüzden es geçilip bir sonraki restart'a devam edilir.
                if (
                    incumbent is not None
                    and canonicalize_routes(candidate.routes, self.problem.depot)
                    == canonicalize_routes(incumbent[0].routes, self.problem.depot)
                ):
                    item["repair"] = "skipped_identical_to_incumbent"
                    restart_trace["attempts"].append(item)
                    continue
                
                restart_trace["attempts"].append(item)
                self.state_machine.mark_restart()
                return candidate, evaluation, restart_trace

            repaired = self._repair_until_valid(
                candidate,
                Path(candidate.image_path or image_path),
                scope=f"{scope}.restart_{restart_attempt:02d}",
                image_dir=image_dir,
                image_prefix=f"{image_prefix}_{restart_attempt:02d}_repair",
                resume=resume,
            )
            if repaired is not None:
                repaired_candidate, repaired_eval, repair_trace = repaired
                item["repair"] = repair_trace
                restart_trace["attempts"].append(item)
                self.state_machine.mark_restart()
                return repaired_candidate, repaired_eval, restart_trace
            item["repair"] = "failed"
            restart_trace["attempts"].append(item)

        restart_trace["exhausted"] = True
        restart_trace["max_attempts"] = self.config.max_restart_attempts
        restart_trace["fallback_action"] = (
            "retain_incumbent" if incumbent is not None else "hard_fail_no_incumbent"
        )

        if not (resume and self.trace.find_last("restart_exhausted", scope=scope)):
            self.trace.append(
                {
                    "event": "restart_exhausted",
                    "scope": scope,
                    "max_attempts": self.config.max_restart_attempts,
                    "fallback_action": restart_trace["fallback_action"],
                }
            )

        if incumbent is not None:
            retained, retained_eval = incumbent
            if not retained_eval.validation.valid:
                raise RuntimeError("Restart fallback incumbent geçerli değil")
            restart_trace["retained_route"] = [list(r) for r in retained.routes]
            return retained, retained_eval, restart_trace

        raise RuntimeError(
            f"Diversity Restart {self.config.max_restart_attempts} denemede geçerli rota üretemedi"
        )

    def _initial_route(
        self,
        *,
        resume: bool = False,
    ) -> tuple[RouteCandidate, ObserverEvaluation, dict[str, Any]]:
        final_event = self.trace.find_last("initializer_result") if resume else None
        if final_event is not None:
            candidate, evaluation, _ = self._candidate_from_event(
                final_event,
                source="initializer_resume_final",
                candidate_id=1,
                route_key="accepted_route",
            )
            return candidate, evaluation, dict(final_event.get("result") or {})

        image_dir = self.routes_dir / "initializer"
        image_path = image_dir / "candidate.png"
        cached = self.trace.find_last("initializer_candidate") if resume else None
        candidate: RouteCandidate | None = None
        evaluation: ObserverEvaluation | None = None

        if cached is not None:
            call = self._last_call("initializer")
            raw_text = ((call or {}).get("call") or {}).get("raw_response", "")
            candidate, evaluation, image_path = self._candidate_from_event(
                cached,
                source="initializer_resume",
                candidate_id=1,
                route_key="route",
                raw_text=raw_text,
            )
        else:
            cached_call = self._last_call("initializer") if resume else None
            if cached_call is not None:
                raw_text = (cached_call.get("call") or {}).get("raw_response", "")
                candidate = RouteCandidate(1, parse_cvrp_routes(raw_text), "initializer_resume_raw", raw_text)
                candidate, evaluation = self._render_and_observe(candidate, image_path)
            else:
                context: dict[str, Any] = {}
                result = self._invoke_recoverable(
                    context,
                    "initializer",
                    lambda: self.initializer.run(
                        self.problem_image,
                        allowed_node_ids=self.allowed_node_ids,
                    ),
                    resume=resume,
                )
                if result is not None:
                    self._record_call(context, result.call)
                    candidate, evaluation = self._render_and_observe(result.candidate, image_path)

            if candidate is not None and evaluation is not None:
                self.trace.append(
                    {
                        "event": "initializer_candidate",
                        "route": [list(r) for r in candidate.routes],
                        "evaluation": evaluation.to_dict(),
                        "image": relative_artifact(candidate.image_path, self.run_dir),
                    }
                )

        if candidate is None or evaluation is None:
            trace: dict[str, Any] = {
                "route": None,
                "evaluation": None,
                "resumed": resume,
                "initializer_output_failure": True,
            }
            accepted, accepted_eval, restart_trace = self._restart_until_valid(
                scope="initializer.fallback",
                image_dir=image_dir,
                image_prefix="restart",
                resume=resume,
            )
            trace["restart"] = restart_trace
        else:
            trace = {
                "route": [list(r) for r in candidate.routes],
                "evaluation": evaluation.to_dict(),
                "resumed": resume,
            }
            accepted = candidate
            accepted_eval = evaluation

            if not evaluation.validation.valid:
                repaired = self._repair_until_valid(
                    candidate,
                    Path(candidate.image_path or image_path),
                    scope="initializer",
                    image_dir=image_dir,
                    image_prefix="repair",
                    resume=resume,
                )
                if repaired is not None:
                    accepted, accepted_eval, repair_trace = repaired
                    trace["repair"] = repair_trace
                else:
                    accepted, accepted_eval, restart_trace = self._restart_until_valid(
                        scope="initializer.fallback",
                        image_dir=image_dir,
                        image_prefix="restart",
                        resume=resume,
                    )
                    trace["restart"] = restart_trace

        self.trace.append(
            {
                "event": "initializer_result",
                "accepted_route": [list(r) for r in accepted.routes],
                "evaluation": accepted_eval.to_dict(),
                "image": relative_artifact(accepted.image_path, self.run_dir),
                "result": trace,
            }
        )
        return accepted, accepted_eval, trace

    def _hybrid_audit(
        self,
        old_routes: tuple[tuple[int, ...], ...],
        new_routes: tuple[tuple[int, ...], ...],
        selected_edges: tuple[tuple[int, int], tuple[int, int]],
    ) -> dict[str, Any]:
        """Audit that a Hybrid output is exactly one intra-route 2-opt move.

        The selected edges must be two distinct, non-adjacent edges from the
        same input route, and they must be exactly the two edges removed by
        the observed transition.  The generic transition audit additionally
        verifies that the resulting route set can be obtained by reversing
        one contiguous segment of that same route, with no inter-route
        customer exchange.
        """
        old = tuple(tuple(route) for route in old_routes)
        new = tuple(tuple(route) for route in new_routes)
        selected = tuple(
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in selected_edges
        )

        old_edges = undirected_edge_set(old)
        new_edges = undirected_edge_set(new)

        selected_exist = all(edge in old_edges for edge in selected)
        selected_distinct = len(set(selected)) == 2

        same_input_route = False
        non_adjacent = False
        if selected_distinct:
            for route in old:
                route_edges = {
                    tuple(sorted((a, b)))
                    for a, b in zip(route, route[1:])
                    if a != b
                }
                if all(edge in route_edges for edge in selected):
                    same_input_route = True
                    positions = []
                    for edge in selected:
                        for index, (a, b) in enumerate(zip(route, route[1:])):
                            if tuple(sorted((a, b))) == edge:
                                positions.append(index)
                                break
                    non_adjacent = (
                        len(positions) == 2
                        and abs(positions[0] - positions[1]) > 1
                    )
                    break

        removed_edges = old_edges - new_edges
        added_edges = new_edges - old_edges
        selected_are_exactly_removed = (
            selected_distinct
            and frozenset(selected) == removed_edges
            and len(removed_edges) == 2
            and len(added_edges) == 2
        )

        transition_is_exact = is_exact_two_opt_transition(old, new)
        is_exact = (
            transition_is_exact
            and selected_exist
            and same_input_route
            and non_adjacent
            and selected_are_exactly_removed
        )

        return {
            "selected_edges": [list(edge) for edge in selected_edges],
            "selected_edges_exist_in_input_route": selected_exist,
            "selected_edges_distinct": selected_distinct,
            "selected_edges_same_input_route": same_input_route,
            "selected_edges_non_adjacent": non_adjacent,
            "removed_edges": [list(edge) for edge in sorted(removed_edges)],
            "added_edges": [list(edge) for edge in sorted(added_edges)],
            "selected_edges_are_exactly_removed": selected_are_exactly_removed,
            "transition_is_exact_single_two_opt": transition_is_exact,
            "exact_single_two_opt_transition": is_exact,
        }

    def _run_hybrid_escape(
        self,
        working: RouteCandidate,
        working_image: Path,
        *,
        iteration: int,
        image_dir: Path,
        resume: bool = False,
    ) -> tuple[RouteCandidate, ObserverEvaluation, dict[str, Any]]:
        scope = f"iteration_{iteration:03d}.hybrid"
        context = self._context(scope=scope, iteration=iteration)
        image_path = image_dir / "hybrid.png"
        cached = self.trace.find_last("hybrid_result", iteration=iteration) if resume else None

        if cached is not None:
            call = self._last_call("hybrid", scope=scope, iteration=iteration)
            raw_text = ((call or {}).get("call") or {}).get("raw_response", "")
            candidate, evaluation, image_path = self._candidate_from_event(
                cached,
                source="hybrid_resume",
                candidate_id=1,
                route_key="output_route",
                raw_text=raw_text,
            )
            edge_lists = (cached.get("two_opt_audit") or {}).get("selected_edges") or []
            selected_edges = (
                (int(edge_lists[0][0]), int(edge_lists[0][1])),
                (int(edge_lists[1][0]), int(edge_lists[1][1])),
            )
        else:
            cached_call = self._last_call("hybrid", scope=scope, iteration=iteration) if resume else None
            if cached_call is not None:
                raw_text = (cached_call.get("call") or {}).get("raw_response", "")
                routes, selected_edges = parse_hybrid(raw_text)
                candidate = RouteCandidate(1, routes, "hybrid_resume_raw", raw_text)
                candidate, evaluation = self._render_and_observe(candidate, image_path)
            else:
                result = self._invoke_recoverable(
                    context,
                    "hybrid_visual_two_opt",
                    lambda: self.hybrid.run(
                        working_image,
                        allowed_node_ids=self.allowed_node_ids,
                    ),
                    resume=resume,
                )
                if result is None:
                    trace: dict[str, Any] = {
                        "input_route": [list(r) for r in working.routes],
                        "output_route": None,
                        "evaluation": None,
                        "two_opt_audit": None,
                        "hybrid_agent_failure": True,
                    }
                    restarted, restarted_eval, restart_trace = self._restart_until_valid(
                        scope=f"{scope}.fallback",
                        image_dir=image_dir,
                        image_prefix="hybrid_restart",
                        resume=resume,
                        incumbent=(working, evaluate_cvrp_routes(self.problem, working.routes)),
                    )
                    trace["restart"] = restart_trace
                    if restart_trace.get("fallback_action") == "retain_incumbent":
                        trace["fallback_action"] = "retain_incumbent"
                    return restarted, restarted_eval, trace
                self._record_call(context, result.call)
                candidate, evaluation = self._render_and_observe(result.candidate, image_path)
                selected_edges = result.selected_edges

            audit = self._hybrid_audit(working.routes, candidate.routes, selected_edges)
            self.trace.append(
                {
                    "event": "hybrid_result",
                    **context,
                    "input_route": [list(r) for r in working.routes],
                    "output_route": [list(r) for r in candidate.routes],
                    "evaluation": evaluation.to_dict(),
                    "two_opt_audit": audit,
                    "image": relative_artifact(candidate.image_path, self.run_dir),
                }
            )

        audit = self._hybrid_audit(working.routes, candidate.routes, selected_edges)
        trace: dict[str, Any] = {
            "input_route": [list(r) for r in working.routes],
            "output_route": [list(r) for r in candidate.routes],
            "evaluation": evaluation.to_dict(),
            "two_opt_audit": audit,
        }

        if evaluation.validation.valid and audit["exact_single_two_opt_transition"]:
            return candidate, evaluation, trace

        # A feasible candidate that violates the Hybrid operator contract
        # must not be repaired: Repair could produce a different operator
        # transition and would therefore invalidate the Hybrid protocol.
        if not evaluation.validation.valid:
            repaired = self._repair_until_valid(
                candidate,
                Path(candidate.image_path or image_path),
                scope=scope,
                image_dir=image_dir,
                image_prefix="hybrid_repair",
                resume=resume,
            )
            if repaired is not None:
                repaired_candidate, repaired_eval, repair_trace = repaired
                trace["repair"] = repair_trace
                return repaired_candidate, repaired_eval, trace

        trace["hybrid_contract_violation"] = (
            "feasible_but_not_exact_single_intra_route_two_opt"
            if evaluation.validation.valid
            else "invalid_hybrid_candidate"
        )

        restarted, restarted_eval, restart_trace = self._restart_until_valid(
            scope=f"{scope}.fallback",
            image_dir=image_dir,
            image_prefix="hybrid_restart",
            resume=resume,
            incumbent=(working, evaluate_cvrp_routes(self.problem, working.routes)),
        )
        trace["restart"] = restart_trace
        if restart_trace.get("fallback_action") == "retain_incumbent":
            trace["fallback_action"] = "retain_incumbent"
        return restarted, restarted_eval, trace

    def _checkpoint(self, iteration: int, working: RouteCandidate) -> None:
        state = CheckpointState(
            completed_iteration=iteration,
            working_routes=[list(r) for r in working.routes],
            structural_history=[[list(r) for r in routes] for routes in self.structural_history],
            hybrid_used_since_restart=self.state_machine.hybrid_used_since_restart,
            restart_count=self.restart_count,
            observed_oracle_best_distance=self.observed_oracle_best_distance,
            observed_oracle_best_routes=([list(r) for r in self.observed_oracle_best_route] if self.observed_oracle_best_route else None),
            selected_best_distance=self.selected_best_distance,
            selected_best_routes=([list(r) for r in self.selected_best_route] if self.selected_best_route else None),
            config_sha256=self.config.sha256,
            instance_sha256=self.problem.source_sha256,
        )
        update_state(
            self.state_path,
            status="running",
            checkpoint=checkpoint_payload(state),
            working_image=relative_artifact(working.image_path, self.run_dir),
            completed_iterations=iteration,
            current={"phase": "iteration_complete", "iteration": iteration},
            last_error=None,
        )

    def _restore(self) -> tuple[int, RouteCandidate]:
        state_data = read_state(self.state_path)
        checkpoint_data = state_data.get("checkpoint")
        if not isinstance(checkpoint_data, dict):
            raise ValueError("Resume reddedildi: state.json checkpoint içermiyor")
        checkpoint = checkpoint_from_payload(checkpoint_data)
        if checkpoint.config_sha256 != self.config.sha256:
            raise ValueError("Resume reddedildi: config SHA256 checkpoint ile eşleşmiyor")
        if checkpoint.instance_sha256 != self.problem.source_sha256:
            raise ValueError("Resume reddedildi: instance SHA256 checkpoint ile eşleşmiyor")
            
        self.structural_history = [tuple(tuple(r) for r in routes) for routes in checkpoint.structural_history]
        self.state_machine.hybrid_used_since_restart = checkpoint.hybrid_used_since_restart
        self.restart_count = checkpoint.restart_count
        self.observed_oracle_best_distance = checkpoint.observed_oracle_best_distance
        self.observed_oracle_best_route = (
            tuple(tuple(r) for r in checkpoint.observed_oracle_best_routes) if checkpoint.observed_oracle_best_routes else None
        )
        self.selected_best_distance = checkpoint.selected_best_distance
        self.selected_best_route = tuple(tuple(r) for r in checkpoint.selected_best_routes) if checkpoint.selected_best_routes else None
        
        image = resolve_artifact(state_data.get("working_image"), self.run_dir)
        working_routes_tup = tuple(tuple(r) for r in checkpoint.working_routes)
        if image is None or not image.exists():
            image = self.routes_dir / "resume_working.png"
            render_routes(self.problem, working_routes_tup, image, self.config.render, demand_encoding=self.config.demand_encoding, route_rendering=self.config.route_rendering)
            update_state(self.state_path, working_image=relative_artifact(image, self.run_dir))
        working = RouteCandidate(1, working_routes_tup, "resume", "", str(image))
        return checkpoint.completed_iteration + 1, working

    def _validate_precheckpoint_resume(self) -> None:
        state = read_state(self.state_path)
        if not state:
            raise ValueError("Resume reddedildi: state.json bulunamadı")
        if state.get("config_sha256") != self.config.sha256:
            raise ValueError("Resume reddedildi: config SHA256 state ile eşleşmiyor")
        if state.get("instance_sha256") not in {None, self.problem.source_sha256}:
            raise ValueError("Resume reddedildi: instance SHA256 state ile eşleşmiyor")

    def _critic_candidates(
        self,
        iteration: int,
        input_image: Path,
        *,
        resume: bool,
    ) -> tuple[list[dict[str, Any]], dict[int, tuple[RouteCandidate, ObserverEvaluation, Path]]]:
        critic_records: list[dict[str, Any]] = []
        rendered_candidates: dict[int, tuple[RouteCandidate, ObserverEvaluation, Path]] = {}
        image_dir = self.routes_dir / f"iteration_{iteration:03d}"

        for candidate_id in range(1, self.config.critic.candidates + 1):
            context = self._context(iteration=iteration, candidate=candidate_id)
            image_path = image_dir / f"C{candidate_id}.png"
            cached = self.trace.find_last(
                "critic_result", iteration=iteration, candidate=candidate_id
            ) if resume else None

            if cached is not None:
                call = self._last_call("critic", iteration=iteration, candidate=candidate_id)
                raw_text = ((call or {}).get("call") or {}).get("raw_response", "")
                candidate, evaluation, image_path = self._candidate_from_event(
                    cached,
                    source="critic_resume",
                    candidate_id=candidate_id,
                    route_key="route",
                    raw_text=raw_text,
                )
            else:
                cached_call = self._last_call("critic", iteration=iteration, candidate=candidate_id) if resume else None
                if cached_call is not None:
                    raw_text = (cached_call.get("call") or {}).get("raw_response", "")
                    candidate = RouteCandidate(candidate_id, parse_cvrp_routes(raw_text), "critic_resume_raw", raw_text)
                    candidate, evaluation = self._render_and_observe(candidate, image_path)
                else:
                    result = self._invoke_recoverable(
                        context,
                        f"critic_candidate_{candidate_id:02d}",
                        lambda candidate_id=candidate_id: self.critic.run_one(
                            input_image,
                            allowed_node_ids=self.allowed_node_ids,
                            candidate_id=candidate_id,
                        ),
                        resume=resume,
                    )
                    if result is None:
                        critic_records.append(
                            {
                                "candidate_id": candidate_id,
                                "route": None,
                                "evaluation": None,
                                "output_failure": True,
                            }
                        )
                        continue
                    self._record_call(context, result.call)
                    candidate, evaluation = self._render_and_observe(result.candidate, image_path)

                self.trace.append(
                    {
                        "event": "critic_result",
                        **context,
                        "route": [list(r) for r in candidate.routes],
                        "evaluation": evaluation.to_dict(),
                        "image": relative_artifact(candidate.image_path, self.run_dir),
                    }
                )

            record = {
                "candidate_id": candidate_id,
                "route": [list(r) for r in candidate.routes],
                "evaluation": evaluation.to_dict(),
            }
            rendered_candidates[candidate_id] = (candidate, evaluation, image_path)
            critic_records.append(record)

        return critic_records, rendered_candidates

    def _scorer_selection(
        self,
        iteration: int,
        rendered_candidates: dict[int, tuple[RouteCandidate, ObserverEvaluation, Path]],
        *,
        resume: bool,
    ) -> tuple[list[int], list[int], int | None]:
        shuffled_ids = list(rendered_candidates)
        if not shuffled_ids:
            return [], [], None
        random.Random(self.config.seed + iteration).shuffle(shuffled_ids)
        expected = set(shuffled_ids)
        cached = self.trace.find_last("scorer_result", iteration=iteration) if resume else None

        if cached is not None:
            cached_order = [int(value) for value in cached["display_order"]]
            if cached_order != shuffled_ids:
                raise ValueError("Resume reddedildi: scorer display order değişmiş")
            ranking = [int(value) for value in cached["ranking"]]
            best_id = int(cached["best_id"])
            if set(ranking) != expected or len(ranking) != len(expected) or ranking[0] != best_id:
                raise ValueError("Resume reddedildi: cached scorer sonucu candidate set ile uyumsuz")
            return shuffled_ids, ranking, best_id

        context = self._context(iteration=iteration)
        cached_call = self._last_call("scorer", iteration=iteration) if resume else None
        if cached_call is not None:
            ranking, best_id = parse_scorer(
                (cached_call.get("call") or {}).get("raw_response", ""), expected
            )
        else:
            scorer_input = [
                (candidate_id, rendered_candidates[candidate_id][2])
                for candidate_id in shuffled_ids
            ]
            scorer_result = self._invoke_recoverable(
                context,
                "visual_scorer",
                lambda: self.scorer.run(self.problem_image, scorer_input),
                resume=resume,
            )
            if scorer_result is None:
                return shuffled_ids, [], None
            self._record_call(context, scorer_result.call)
            ranking = scorer_result.ranking
            best_id = scorer_result.best_id

        self.trace.append(
            {
                "event": "scorer_result",
                **context,
                "display_order": shuffled_ids,
                "ranking": ranking,
                "best_id": best_id,
            }
        )
        return shuffled_ids, ranking, best_id

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        state = read_state(self.state_path)
        if resume and isinstance(state.get("checkpoint"), dict):
            start_iteration, working = self._restore()
            working_eval = self._observe(working.routes)
            if not working_eval.validation.valid:
                raise RuntimeError("Checkpoint working route geçerli değil")
        elif resume:
            self._validate_precheckpoint_resume()
            start_iteration = 1
            working, working_eval, _ = self._initial_route(resume=True)
            self.structural_history = [working.routes]
            self._observe_selected(working.routes, working_eval)
            self._checkpoint(0, working)
        else:
            start_iteration = 1
            working, working_eval, _ = self._initial_route(resume=False)
            self.structural_history = [working.routes]
            self._observe_selected(working.routes, working_eval)
            self._checkpoint(0, working)

        for iteration in range(start_iteration, self.config.iterations + 1):
            if not working.image_path:
                raise RuntimeError("Working route image bulunamadı")
            input_image = Path(working.image_path)

            critic_records, rendered_candidates = self._critic_candidates(
                iteration,
                input_image,
                resume=resume,
            )

            shuffled_ids, scorer_ranking, scorer_best_id = self._scorer_selection(
                iteration,
                rendered_candidates,
                resume=resume,
            )

            repair_trace = None
            restart_trace = None
            feasibility_fallback = None
            iteration_image_dir = self.routes_dir / f"iteration_{iteration:03d}"
            transition_accepted = True

            if scorer_best_id is None:
                selected = working
                selected_eval = working_eval
                selected_image = input_image
                selected_before_repair: dict[str, Any] = {
                    "candidate_id": None,
                    "route": None,
                    "evaluation": None,
                    "selection_status": "agent_failure",
                }
                transition_accepted = False
                feasibility_fallback = {
                    "triggered": True,
                    "reason": (
                        "all_critic_outputs_failed"
                        if not rendered_candidates
                        else "scorer_output_failure"
                    ),
                    "action": "retain_previous_working_route",
                }
            else:
                selected, selected_eval, selected_image = rendered_candidates[scorer_best_id]
                selected_before_repair = {
                    "candidate_id": selected.candidate_id,
                    "route": [list(r) for r in selected.routes],
                    "evaluation": selected_eval.to_dict(),
                }

                if not selected_eval.validation.valid:
                    repaired = self._repair_until_valid(
                        selected,
                        selected_image,
                        scope=f"iteration_{iteration:03d}.selected",
                        image_dir=iteration_image_dir,
                        image_prefix="repair",
                        resume=resume,
                    )
                    if repaired is not None:
                        selected, selected_eval, repair_trace = repaired
                        selected_image = Path(selected.image_path or selected_image)
                    else:
                        selected, selected_eval, restart_trace = self._restart_until_valid(
                            scope=f"iteration_{iteration:03d}.selected_restart",
                            image_dir=iteration_image_dir,
                            image_prefix="restart",
                            resume=resume,
                            incumbent=(working, working_eval),
                        )
                        selected_image = Path(selected.image_path or selected_image)
                        if restart_trace.get("fallback_action") == "retain_incumbent":
                            transition_accepted = False
                            feasibility_fallback = {
                                "triggered": True,
                                "reason": "restart_attempts_exhausted",
                                "action": "retain_previous_working_route",
                                "restart_attempts": self.config.max_restart_attempts,
                            }

            working = selected
            working_eval = selected_eval
            self._observe_selected(working.routes, working_eval)
            if transition_accepted:
                self.structural_history.append(working.routes)

            stagnation = detect_structural_stagnation(
                self.structural_history,
                depot=self.problem.depot,
                window=self.config.stagnation.window,
                similarity_threshold=self.config.stagnation.similarity_threshold,
                max_unique_routes=self.config.stagnation.max_unique_routes,
            )
            escape: dict[str, Any] | None = None

            if transition_accepted and stagnation.stagnated:
                action = self.state_machine.action_for_stagnation()
                if action == "hybrid":
                    before_hybrid = working
                    working, working_eval, hybrid_trace = self._run_hybrid_escape(
                        working,
                        selected_image,
                        iteration=iteration,
                        image_dir=iteration_image_dir,
                        resume=resume,
                    )
                    self._observe_selected(working.routes, working_eval)
                    hybrid_retained = hybrid_trace.get("fallback_action") == "retain_incumbent"
                    if not hybrid_retained:
                        self.structural_history = [working.routes]
                    escape = {
                        "action": (
                            "hybrid_failed_retain_incumbent"
                            if hybrid_retained
                            else "hybrid"
                        ),
                        "input_route": [list(r) for r in before_hybrid.routes],
                        "trace": hybrid_trace,
                    }
                else:
                    working, working_eval, diversity_trace = self._restart_until_valid(
                        scope=f"iteration_{iteration:03d}.escape_restart",
                        image_dir=iteration_image_dir,
                        image_prefix="escape_restart",
                        resume=resume,
                        incumbent=(working, working_eval),
                    )
                    self._observe_selected(working.routes, working_eval)
                    restart_retained = diversity_trace.get("fallback_action") == "retain_incumbent"
                    if not restart_retained:
                        self.structural_history = [working.routes]
                    escape = {
                        "action": (
                            "restart_failed_retain_incumbent"
                            if restart_retained
                            else "restart"
                        ),
                        "trace": diversity_trace,
                    }

            iteration_result = {
                "iteration": iteration,
                "critic_candidates": critic_records,
                "scorer": {
                    "display_order": shuffled_ids,
                    "ranking": scorer_ranking,
                    "best_id": scorer_best_id,
                },
                "selected_before_repair": selected_before_repair,
                "repair": repair_trace,
                "restart_after_failed_repair": restart_trace,
                "feasibility_fallback": feasibility_fallback,
                "working_route_after_iteration": [list(r) for r in working.routes],
                "working_evaluation": working_eval.to_dict(),
                "structural_stagnation": stagnation.to_dict(),
                "escape": escape,
                "observer_only": {
                    "observed_oracle_best_distance": self.observed_oracle_best_distance,
                    "observed_oracle_best_route": (
                        [list(r) for r in self.observed_oracle_best_route] if self.observed_oracle_best_route else None
                    ),
                    "selected_best_distance": self.selected_best_distance,
                    "selected_best_route": [list(r) for r in self.selected_best_route] if self.selected_best_route else None,
                },
            }
            self.trace.append(
                {
                    "event": "iteration_result",
                    "iteration": iteration,
                    "result": iteration_result,
                }
            )
            self._checkpoint(iteration, working)

        summary = {
            "completed_iterations": self.config.iterations,
            "final_working_route": [list(r) for r in working.routes],
            "final_working_evaluation": evaluate_cvrp_routes(self.problem, working.routes).to_dict(),
            "selected_best_distance": self.selected_best_distance,
            "selected_best_route": [list(r) for r in self.selected_best_route] if self.selected_best_route else None,
            "observed_oracle_best_distance": self.observed_oracle_best_distance,
            "observed_oracle_best_route": (
                [list(r) for r in self.observed_oracle_best_route] if self.observed_oracle_best_route else None
            ),
            "restart_count": self.restart_count,
        }
        update_state(
            self.state_path,
            status="completed",
            summary=summary,
            completed_iterations=self.config.iterations,
            current={"phase": "completed", "iteration": self.config.iterations},
            last_error=None,
        )
        return summary