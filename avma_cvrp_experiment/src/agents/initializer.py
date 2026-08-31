from __future__ import annotations

from pathlib import Path

from ..schemas import RouteAgentResult, RouteCandidate
from .base import BaseAgent, ModelOutputError, output_attempt_record, parse_cvrp_routes


class InitializerAgent(BaseAgent):
    role_name = "initializer"

    def run(self, problem_image: str | Path, *, allowed_node_ids: set[int]) -> RouteAgentResult:
        parts = self._parts([("ORIGINAL CVRP PROBLEM IMAGE", problem_image)])
        last_error: Exception | None = None
        failed_attempts: list[dict[str, object]] = []
        
        for retry in range(self.config.max_output_retries + 1):
            response = self.provider.generate(
                parts,
                phase="initializer",
                temperature=self.config.temperature,
                thinking_level=self.config.thinking_level,
            )
            try:
                routes = parse_cvrp_routes(response.text)
                self.ensure_renderable(routes, allowed_node_ids)
                return RouteAgentResult(
                    RouteCandidate(1, routes, "initializer", response.text),
                    self._call_record(response, parts, failed_output_attempts=failed_attempts),
                )
            except ModelOutputError as exc:
                last_error = exc
                failed_attempts.append(
                    output_attempt_record(response, exc, attempt=retry + 1)
                )
                
        raise ModelOutputError(f"Initializer kullanılır çıktı üretemedi: {last_error}", attempts=failed_attempts)