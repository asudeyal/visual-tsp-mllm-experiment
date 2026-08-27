from __future__ import annotations

from pathlib import Path

import pytest

from src.schemas import ProblemInstance


@pytest.fixture
def square_problem() -> ProblemInstance:
    return ProblemInstance(
        name="square4",
        dimension=4,
        node_ids=(1, 2, 3, 4),
        coordinates={
            1: (0.0, 0.0),
            2: (1.0, 0.0),
            3: (1.0, 1.0),
            4: (0.0, 1.0),
        },
        depot=1,
        edge_weight_type="EUC_2D",
        source_path=None,
        source_sha256="test",
        reference_optimum=4.0,
    )
