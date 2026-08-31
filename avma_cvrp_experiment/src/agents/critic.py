from __future__ import annotations

from pathlib import Path

from ..schemas import CriticResult, RouteCandidate, RouteAgentResult
from .base import BaseAgent, ModelOutputError, output_attempt_record, parse_cvrp_routes


class CriticAgent(BaseAgent):
    role_name = "critic"

    def run_one(
        self,
        route_image: str | Path,
        *,
        allowed_node_ids: set[int],
        candidate_id: int,
    ) -> RouteAgentResult:
        """Generate one independently sampled, renderable critic candidate.

        The orchestrator persists each successful candidate immediately. This
        makes a partially completed ensemble resumable without regenerating the
        candidates that already succeeded.
        """

        parts = self._parts([("CURRENT CVRP ROUTES IMAGE", route_image)])
        last_error: Exception | None = None
        failed_attempts: list[dict[str, object]] = []
        
        for retry in range(self.config.max_output_retries + 1):
            response = self.provider.generate(
                parts,
                phase=f"critic_candidate_{candidate_id:02d}_output_{retry + 1:02d}",
                temperature=self.config.temperature,
                thinking_level=self.config.thinking_level,
            )
            try:
                routes = parse_cvrp_routes(response.text)
                self.ensure_renderable(routes, allowed_node_ids)
                return RouteAgentResult(
                    RouteCandidate(candidate_id, routes, "critic", response.text),
                    self._call_record(response, parts, failed_output_attempts=failed_attempts),
                )
            except ModelOutputError as exc:
                last_error = exc
                failed_attempts.append(
                    output_attempt_record(response, exc, attempt=retry + 1)
                )
                
        raise ModelOutputError(
            f"Critic candidate {candidate_id} kullanılır çıktı üretemedi: {last_error}",
            attempts=failed_attempts,
        )

    def run(
        self,
        route_image: str | Path,
        *,
        allowed_node_ids: set[int],
        candidate_strategy: str,
    ) -> CriticResult:
        # Kept for compatibility. AVMA's orchestrator intentionally uses
        # run_one() so each of the three ensemble members is independently
        # sampled and can be checkpointed at candidate granularity.
        results = [
            self.run_one(
                route_image,
                allowed_node_ids=allowed_node_ids,
                candidate_id=candidate_id,
            )
            for candidate_id in range(1, self.config.candidates + 1)
        ]
        return CriticResult(
            candidates=[result.candidate for result in results],
            calls=[result.call for result in results],
        )