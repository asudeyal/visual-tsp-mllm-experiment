from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..schemas import ScorerResult
from .base import BaseAgent, ModelOutputError, output_attempt_record, parse_scorer


class ScorerAgent(BaseAgent):
    role_name = "scorer"

    def run(
        self,
        problem_image: str | Path,
        candidate_images: Sequence[tuple[int, str | Path]],
    ) -> ScorerResult:
        labeled: list[tuple[str, str | Path]] = [("REFERENCE ORIGINAL CVRP PROBLEM IMAGE", problem_image)]
        for candidate_id, path in candidate_images:
            labeled.append((f"CANDIDATE ID {candidate_id}", path))
        parts = self._parts(labeled)
        expected = {candidate_id for candidate_id, _ in candidate_images}
        last_error: Exception | None = None
        failed_attempts: list[dict[str, object]] = []
        for retry in range(self.config.max_output_retries + 1):
            response = self.provider.generate(
                parts,
                phase="visual_scorer",
                temperature=self.config.temperature,
                thinking_level=self.config.thinking_level,
            )
            try:
                ranking, best_id = parse_scorer(response.text, expected)
                return ScorerResult(ranking, best_id, self._call_record(response, parts, failed_output_attempts=failed_attempts))
            except ModelOutputError as exc:
                last_error = exc
                failed_attempts.append(
                    output_attempt_record(response, exc, attempt=retry + 1)
                )
        raise ModelOutputError(f"Scorer kullanılır çıktı üretemedi: {last_error}", attempts=failed_attempts)