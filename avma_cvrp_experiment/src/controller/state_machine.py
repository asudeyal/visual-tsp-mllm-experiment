"""Minimal escape-state machine: first stagnation -> Hybrid, next -> Restart."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EscapeStateMachine:
    hybrid_used_since_restart: bool = False

    def action_for_stagnation(self) -> str:
        if not self.hybrid_used_since_restart:
            self.hybrid_used_since_restart = True
            return "hybrid"
        return "restart"

    def mark_restart(self) -> None:
        self.hybrid_used_since_restart = False
