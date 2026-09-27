import numpy as np
import pytest
from pydantic import ValidationError

from lucida_mini.schema import DatasetManifest, PoseRecord, TurnRecord


def pose_record() -> PoseRecord:
    return PoseRecord(
        position_m=[0.0, 0.0, 0.0],
        rotation_object_to_world=np.eye(3).tolist(),
        size_m=[1.0, 1.0, 1.0],
    )


def test_injected_error_cannot_be_supervised() -> None:
    with pytest.raises(ValidationError):
        TurnRecord(
            step=0,
            observation_paths=["observation.png"],
            action="<stop>{}</stop>",
            injected_error=True,
            supervise=True,
            state_before=pose_record(),
            state_after=pose_record(),
        )


def test_manifest_rejects_non_fixed_dataset() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest(name="gizmoact_overfit50", frozen=False, contexts=[], trajectories=[])
