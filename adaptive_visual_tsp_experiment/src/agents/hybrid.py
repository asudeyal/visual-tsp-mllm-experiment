from __future__ import annotations

from pathlib import Path

from ..schemas import HybridResult, RouteCandidate
from .base import BaseAgent, ModelOutputError, output_attempt_record, parse_hybrid


class HybridAgent(BaseAgent):
    role_name = "hybrid"

    def run(self, route_image: str | Path, *, allowed_node_ids: set[int]) -> HybridResult:
        parts = self._parts([("STRUCTURALLY STAGNANT CURRENT ROUTE IMAGE", route_image)])
        last_error: Exception | None = None
        failed_attempts: list[dict[str, object]] = []
        for retry in range(self.config.max_output_retries + 1):
            response = self.provider.generate(
                parts,
                phase="hybrid_visual_two_opt",
                temperature=self.config.temperature,
                thinking_level=self.config.thinking_level,
            )
            try:
                route, edges = parse_hybrid(response.text)
                self.ensure_renderable(route, allowed_node_ids)
                return HybridResult(
                    RouteCandidate(1, route, "hybrid", response.text),
                    edges,
                    self._call_record(response, parts, failed_output_attempts=failed_attempts),
                )
            except ModelOutputError as exc:
                last_error = exc
                failed_attempts.append(
                    output_attempt_record(response, exc, attempt=retry + 1)
                )
        raise ModelOutputError(f"Hybrid kullanılır çıktı üretemedi: {last_error}", attempts=failed_attempts)
