from pathlib import Path

import numpy as np

from lucida_mini.expert import ExpertThresholds
from lucida_mini.pose import Pose
from lucida_mini.trajectory import PerturbationConfig, generate_expert_trajectory


def test_trajectory_is_deterministic_and_masks_injected_error():
    kwargs = dict(
        context_id="front_01",
        index=0,
        seed=7,
        target=Pose.identity(),
        observation_dir=Path("contexts/front_01/trajectories/00"),
        thresholds=ExpertThresholds(),
        config=PerturbationConfig(inject_error_probability=1.0),
    )
    first = generate_expert_trajectory(**kwargs)
    second = generate_expert_trajectory(**kwargs)
    assert first.model_dump() == second.model_dump()
    errors = [turn for turn in first.turns if turn.injected_error]
    assert len(errors) == 1
    assert not errors[0].supervise
    assert first.termination == "stop"
