"""Static optimisation at the first samples of the five-sample motion (#12827).

Diagnosis: at the first two samples ``rectus_abdominis_r`` of the invented
fixture is shorter than its rigid-tendon minimum fibre length, so its force is
zero at every activation (an all-zero column of the activation gain).  That
column made SLSQP stop with "Inequality constraints incompatible".  Every
number is invented (see ``test_study``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.drive import (
    FIXTURE_PELVIS,
    DriveOptions,
    drive_segmental_model,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import (
    read_golf_humanoid_motion,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    SpineIDOptions,
    spinal_generalized_forces,
)
from src.engines.physics_engines.opensim.python.spine.static_optimization import (
    ReserveSpec,
    SpineMuscleBasis,
    SpineSOOptions,
    run_spine_static_optimization,
)
from tests.helpers.opensim_support import require_opensim
from tests.opensim.spine.test_study import GOLF_MODEL, _law, _retune, _write_motion

_N_SAMPLES = 5
_SLACK_MUSCLE = "rectus_abdominis_r"


@pytest.fixture(scope="module")
def five_sample(tmp_path_factory):
    require_opensim()
    root = tmp_path_factory.mktemp("so_early")
    model = write_fixture_osim(
        FixtureSpec(), root / "fixture.osim", coupling=_law(), muscles=True
    )
    _retune(model)
    golf = read_golf_humanoid_motion(_write_motion(root / "swing.sto"), GOLF_MODEL)
    drive = drive_segmental_model(
        golf,
        model_path=model,
        model_id="fixture",
        law=None,
        options=DriveOptions(tolerance_rad=1e-6, pelvis=FIXTURE_PELVIS),
    )
    efforts = spinal_generalized_forces(
        model, drive, "fixture", options=SpineIDOptions(pelvis=FIXTURE_PELVIS)
    )
    return model, efforts


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_the_early_samples_have_a_muscle_with_no_force_capacity(five_sample) -> None:
    model, efforts = five_sample
    basis = SpineMuscleBasis(model, "fixture")
    column = basis.muscle_names.index(_SLACK_MUSCLE)
    capacity = [
        basis.evaluate(efforts.model_coordinates, efforts.q[f], efforts.qd[f])[0][
            column
        ]
        for f in range(_N_SAMPLES)
    ]
    assert capacity[0] == 0.0 and capacity[1] == 0.0
    assert all(c > 0.0 for c in capacity[2:])


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("exponent", [2, 3])
def test_every_sample_of_the_five_sample_motion_converges(
    five_sample, exponent
) -> None:
    model, efforts = five_sample
    options = SpineSOOptions(
        exponent=exponent,
        reserves=ReserveSpec(limit_nm=1e6, max_share=1e3),
        stride=1,
        max_frames=_N_SAMPLES,
    )
    result = run_spine_static_optimization(model, efforts, "fixture", options=options)
    assert result.frame_index.tolist() == list(range(_N_SAMPLES))
    assert np.all(result.success), result.success
    assert not any("solver" in r for rs in result.flag_reasons for r in rs)
    column = result.muscle_names.index(_SLACK_MUSCLE)
    assert np.all(result.activation[:2, column] == 0.0)
    np.testing.assert_allclose(
        result.muscle_moment_nm + result.reserve_nm, result.required_nm, atol=1e-9
    )
