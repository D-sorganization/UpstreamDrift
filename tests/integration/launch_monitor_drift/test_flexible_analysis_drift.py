"""ADR-0046 G0.1 drift gate: flexible analysis, UD stack vs vendored Tools stack.

ADR-0048 row 1 of the five ``needs-decision`` rows. ADR-0046 lists flexible
analysis as a UD-only capability; it is not. Both stacks carry a full
correlation + OLS + group analysis over an arbitrary outcome/predictor
selection, and both define identically named frozen dataclasses.

    UD    ``launch_monitor/flexible_analysis.py`` (415 lines) —
          ``FlexibleAnalysisRequest`` / ``FlexibleAnalysisResult``,
          ``analyze_variables``.
    Tools ``rate_of_closure/launch_monitor_analysis.py`` (228) plus
          ``_launch_monitor_analysis_statistics.py`` (200) and
          ``_launch_monitor_analysis_types.py`` (137) — ``AnalysisRequest`` /
          ``AnalysisResult``, ``analyze_launch_monitor_data``.

AGREE — asserted to bit equality on the 160-shot clean session
    * ``CONTRACT_VERSION`` is ``"1.0.0"`` on both sides.
    * ``DatasetSummary`` is field-for-field identical **and the
      ``fingerprint_sha256`` is the same digest**,
      ``64a5a550eeb7c8007d620d7b6c127b2c24e8668e372f466bcd45a1602b631b16``.
      The two stacks independently implement the same identity-column +
      selection JSON canonicalisation.
    * All three Pearson correlations agree with delta exactly ``0.0`` —
      coefficient, raw p, Benjamini-Hochberg adjusted p, and both Fisher-z
      interval bounds.
    * The four-parameter OLS agrees with delta exactly ``0.0`` on every
      estimate, standard error, t statistic, p value and interval bound, plus
      ``r_squared`` 0.004214133185394764 and ``adjusted_r_squared``
      -0.014935595022578463.
    * The six shared residual diagnostics agree exactly, including
      ``durbin_watson`` 0.7496775937342075 and ``influential_count`` 5.
    * The four ``group_by="player_id"`` groups agree in value, row count, and
      per-group ``r_squared`` (0.19222385643053186 / 0.11468147468979084 /
      0.03183901589114291 / 0.029543614100479454).
    * Radix and underscore text (``"0x10"``, ``"1_0"``) is rejected by both,
      leaving the same 158 complete rows.

DIFFER — documented and pinned below
    D15. **The multiplicity denominator differs.** An under-sampled predictor
         (n=5 against ``min_samples=10``) still counts toward UD's
         Benjamini-Hochberg correction and does not count toward Tools'. On the
         same four-predictor request the three fully sampled predictors come
         back adjusted to 0.9217169029997262 from UD and 0.8646154865187129
         from Tools — a 6.6% difference in a reported FDR value, from identical
         raw p values.
    D16. **UD fails closed on an unknown enum; Tools silently degrades.** UD
         validates ``analysis_mode``/``correlation_method``/``missing_policy``
         in ``__post_init__``. Tools validates none of them: an unknown
         ``correlation_method`` falls through to Kendall (bit-identical result
         0.010163995292918303) while still reporting ``method="bogus"``, and an
         unknown ``analysis_mode`` behaves as ``"comprehensive"``.
    D17. **Boolean columns.** UD's ``pd.to_numeric`` projects ``True``/``False``
         to 1.0/0.0 and analyses the column (r = 0.01854649955664114 over 160
         rows); Tools' ``finite_launch_monitor_scalar`` refuses booleans, so the
         column reads as all-null and Tools raises "Constant variables cannot
         be analyzed".
    D18. **Blank provenance strings.** UD drops whitespace-only identifiers from
         ``DatasetSummary.session_ids`` (20 entries); Tools keeps them (21).
    D19. **Result surfaces are near-identical but not equal.** UD alone reports
         ``ResidualDiagnostics.jarque_bera_p_value`` (0.16558673882669894) and
         a per-column ``units`` map; Tools alone carries ``contract_version`` as
         a result field, a camelCase ``to_wire()``, and the
         ``numeric_columns()`` eligibility scan.
    D20. **Missing-estimate sentinel.** UD uses ``float("nan")`` inside the
         dataclass, Tools uses ``None``. Both serialise to JSON ``null``, so
         this bites Python callers only.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.shared.python.launch_monitor.flexible_analysis import (
    CONTRACT_VERSION as UD_CONTRACT_VERSION,
)
from src.shared.python.launch_monitor.flexible_analysis import (
    AnalysisMode,
    CorrelationMethod,
    FlexibleAnalysisRequest,
    MissingPolicy,
    analyze_variables,
)
from tests.integration.launch_monitor_drift.conftest import (
    require_vendored_tools_stack,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

require_vendored_tools_stack()

from rate_of_closure._launch_monitor_analysis_types import (  # noqa: E402
    CONTRACT_VERSION as TOOLS_CONTRACT_VERSION,
)
from rate_of_closure._launch_monitor_analysis_types import (  # noqa: E402
    AnalysisRequest,
)
from rate_of_closure.launch_monitor_analysis import (  # noqa: E402
    analyze_launch_monitor_data,
    numeric_columns,
)

OUTCOME = "carry_distance_metres"
PREDICTORS = ("start_distance_yards", "lateral_carry_metres", "session_order")
GROUP_COLUMN = "player_id"
EXPECTED_SAMPLE_COUNT = 160

# AGREE pins.
SHARED_FINGERPRINT = "64a5a550eeb7c8007d620d7b6c127b2c24e8668e372f466bcd45a1602b631b16"
SHARED_SESSION_ID_COUNT = 20
SHARED_R_SQUARED = 0.004214133185394764
SHARED_ADJUSTED_R_SQUARED = -0.014935595022578463
SHARED_DURBIN_WATSON = 0.7496775937342075
SHARED_INFLUENTIAL_COUNT = 5
SHARED_RMSE = 8.368148642488878
SHARED_GROUP_R_SQUARED = {
    "P1": 0.19222385643053186,
    "P2": 0.11468147468979084,
    "P3": 0.03183901589114291,
    "P4": 0.029543614100479454,
}
SHARED_CORRELATIONS = {
    "start_distance_yards": (
        0.06009330623288146,
        0.4503393090638395,
        -0.09596017196902239,
        0.2132635815386259,
    ),
    "lateral_carry_metres": (
        -0.013585250457195438,
        0.8646154865187129,
        -0.1683890535193738,
        0.14187254384742634,
    ),
    "session_order": (
        0.021129392182954454,
        0.7908538119679481,
        -0.13447020163483708,
        0.17571208130827173,
    ),
}
SHARED_COEFFICIENTS = {
    "intercept": (136.64793588182673, 4.03654565409335),
    "start_distance_yards": (0.018785298936902558, 0.024953166767414856),
    "lateral_carry_metres": (-0.012896264043859839, 0.08787358859457287),
    "session_order": (0.12343602933506102, 0.4747551984681337),
}
SHARED_TEXT_COMPLETE_ROWS = 158

# D15 pins: the same three raw p values, two different FDR denominators.
SPARSE_PREDICTORS = (*PREDICTORS, "sparse_metric")
SPARSE_SAMPLE_COUNT = 5
UD_ADJUSTED_WITH_SPARSE = 0.9217169029997262
TOOLS_ADJUSTED_WITH_SPARSE = 0.8646154865187129

# D16 pin: Tools' unknown-method fall-through equals its Kendall branch.
TOOLS_BOGUS_METHOD_COEFFICIENT = 0.010163995292918303

# D17 pin: the boolean column UD analyses and Tools refuses.
UD_BOOLEAN_COEFFICIENT = 0.01854649955664114

# D19 pin: the diagnostic only UD computes.
UD_ONLY_JARQUE_BERA_P_VALUE = 0.16558673882669894
TOOLS_ONLY_NUMERIC_COLUMNS = (
    "carry_distance_metres",
    "finish_distance_metres",
    "lateral_carry_metres",
    "session_order",
    "shot_index",
    "start_distance_yards",
)


def _ud_request(**overrides: object) -> FlexibleAnalysisRequest:
    settings: dict[str, object] = {
        "outcome": OUTCOME,
        "predictors": PREDICTORS,
        "group_by": GROUP_COLUMN,
    }
    settings.update(overrides)
    return FlexibleAnalysisRequest(**settings)  # type: ignore[arg-type]


def _tools_request(**overrides: object) -> AnalysisRequest:
    settings: dict[str, object] = {
        "outcome": OUTCOME,
        "predictors": PREDICTORS,
        "group_by": GROUP_COLUMN,
    }
    settings.update(overrides)
    return AnalysisRequest(**settings)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def ud_result(session_frame: pd.DataFrame):
    return analyze_variables(session_frame, _ud_request())


@pytest.fixture(scope="module")
def tools_result(session_frame: pd.DataFrame):
    return analyze_launch_monitor_data(session_frame, _tools_request())


def test_contract_versions_agree() -> None:
    """AGREE: both stacks stamp the flexible-analysis contract ``1.0.0``."""
    assert UD_CONTRACT_VERSION == TOOLS_CONTRACT_VERSION == "1.0.0"


def test_dataset_summary_and_fingerprint_agree_exactly(ud_result, tools_result) -> None:
    """AGREE: identical provenance summary, including the sha256 fingerprint."""
    ud_dataset = ud_result.dataset
    tools_dataset = tools_result.dataset

    assert ud_dataset.row_count == tools_dataset.row_count == EXPECTED_SAMPLE_COUNT
    assert (
        ud_dataset.complete_row_count
        == tools_dataset.complete_row_count
        == EXPECTED_SAMPLE_COUNT
    )
    assert ud_dataset.selected_columns == tools_dataset.selected_columns
    assert ud_dataset.monitor_vendors == tools_dataset.monitor_vendors == ()
    assert ud_dataset.observation_kinds == tools_dataset.observation_kinds == ("shot",)
    assert ud_dataset.session_ids == tools_dataset.session_ids
    assert len(ud_dataset.session_ids) == SHARED_SESSION_ID_COUNT
    assert ud_dataset.fingerprint_sha256 == SHARED_FINGERPRINT
    assert tools_dataset.fingerprint_sha256 == SHARED_FINGERPRINT


def test_correlations_agree_bit_for_bit(ud_result, tools_result) -> None:
    """AGREE: coefficient, raw p, adjusted p and both bounds, delta 0.0."""
    tools_by_predictor = {item.predictor: item for item in tools_result.correlations}

    for ud_item in ud_result.correlations:
        tools_item = tools_by_predictor[ud_item.predictor]
        coefficient, p_value, ci_lower, ci_upper = SHARED_CORRELATIONS[
            ud_item.predictor
        ]

        assert ud_item.sample_count == tools_item.sample_count
        assert ud_item.sample_count == EXPECTED_SAMPLE_COUNT
        assert ud_item.method == tools_item.method == "pearson"
        assert ud_item.coefficient == pytest.approx(coefficient, rel=1e-12)
        assert ud_item.coefficient - tools_item.coefficient == 0.0
        assert ud_item.p_value == pytest.approx(p_value, rel=1e-12)
        assert ud_item.p_value - tools_item.p_value == 0.0
        assert ud_item.adjusted_p_value - tools_item.adjusted_p_value == 0.0
        assert ud_item.ci_lower == pytest.approx(ci_lower, rel=1e-12)
        assert ud_item.ci_lower - tools_item.ci_lower == 0.0
        assert ud_item.ci_upper == pytest.approx(ci_upper, rel=1e-12)
        assert ud_item.ci_upper - tools_item.ci_upper == 0.0


def test_regression_agrees_bit_for_bit(ud_result, tools_result) -> None:
    """AGREE: the four-parameter OLS is identical, coefficient by coefficient."""
    ud_regression = ud_result.regression
    tools_regression = tools_result.regression
    assert ud_regression is not None
    assert tools_regression is not None

    assert (
        ud_regression.sample_count
        == tools_regression.sample_count
        == EXPECTED_SAMPLE_COUNT
    )
    assert ud_regression.r_squared == pytest.approx(SHARED_R_SQUARED, rel=1e-12)
    assert ud_regression.r_squared - tools_regression.r_squared == 0.0
    assert ud_regression.adjusted_r_squared == pytest.approx(
        SHARED_ADJUSTED_R_SQUARED, rel=1e-12
    )
    assert ud_regression.adjusted_r_squared - tools_regression.adjusted_r_squared == 0.0

    for name, (estimate, standard_error) in SHARED_COEFFICIENTS.items():
        ud_coefficient = ud_regression.coefficients[name]
        tools_coefficient = tools_regression.coefficients[name]
        assert ud_coefficient.estimate == pytest.approx(estimate, rel=1e-12)
        assert ud_coefficient.standard_error == pytest.approx(standard_error, rel=1e-12)
        for field in (
            "estimate",
            "standard_error",
            "t_statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
        ):
            delta = getattr(ud_coefficient, field) - getattr(tools_coefficient, field)
            assert delta == 0.0, f"{name}.{field} drifted by {delta!r}"


def test_shared_residual_diagnostics_agree_bit_for_bit(ud_result, tools_result) -> None:
    """AGREE: the six diagnostics both stacks define return the same values."""
    ud_diagnostics = ud_result.regression.residual_diagnostics
    tools_diagnostics = tools_result.regression.residual_diagnostics

    assert ud_diagnostics.rmse == pytest.approx(SHARED_RMSE, rel=1e-12)
    assert ud_diagnostics.durbin_watson == pytest.approx(
        SHARED_DURBIN_WATSON, rel=1e-12
    )
    assert ud_diagnostics.influential_count == SHARED_INFLUENTIAL_COUNT
    for field in ("rmse", "mae", "residual_mean", "residual_std", "durbin_watson"):
        delta = getattr(ud_diagnostics, field) - getattr(tools_diagnostics, field)
        assert delta == 0.0, f"{field} drifted by {delta!r}"
    assert ud_diagnostics.influential_count == tools_diagnostics.influential_count


def test_group_analyses_agree_bit_for_bit(ud_result, tools_result) -> None:
    """AGREE: the same four ``player_id`` groups with the same fits."""
    assert len(ud_result.groups) == len(tools_result.groups) == 4

    tools_by_value = {item.group_value: item for item in tools_result.groups}
    for ud_group in ud_result.groups:
        tools_group = tools_by_value[ud_group.group_value]
        assert ud_group.row_count == tools_group.row_count == 40
        assert ud_group.warnings == tools_group.warnings == ()
        assert ud_group.regression is not None
        assert tools_group.regression is not None
        assert ud_group.regression.r_squared == pytest.approx(
            SHARED_GROUP_R_SQUARED[ud_group.group_value], rel=1e-12
        )
        assert ud_group.regression.r_squared - tools_group.regression.r_squared == 0.0


def test_non_numeric_text_is_rejected_identically(session_frame: pd.DataFrame) -> None:
    """AGREE: radix and underscore literals fail both numeric projections."""
    frame = session_frame.copy()
    frame["texty"] = [f"{value:.4f}" for value in frame["session_order"]]
    frame.loc[frame.index[0], "texty"] = "0x10"
    frame.loc[frame.index[1], "texty"] = "1_0"
    predictors = ("session_order", "texty")

    ud = analyze_variables(
        frame,
        _ud_request(predictors=predictors, analysis_mode="correlation", group_by=None),
    )
    tools = analyze_launch_monitor_data(
        frame,
        _tools_request(
            predictors=predictors, analysis_mode="correlation", group_by=None
        ),
    )

    assert ud.dataset.complete_row_count == SHARED_TEXT_COMPLETE_ROWS
    assert tools.dataset.complete_row_count == SHARED_TEXT_COMPLETE_ROWS
    ud_text = next(item for item in ud.correlations if item.predictor == "texty")
    tools_text = next(item for item in tools.correlations if item.predictor == "texty")
    assert ud_text.sample_count == tools_text.sample_count
    assert ud_text.coefficient - tools_text.coefficient == 0.0


def test_divergence_d15_multiplicity_denominator_differs(
    session_frame: pd.DataFrame,
) -> None:
    """DIFFER (D15): an under-sampled predictor counts in UD's FDR, not Tools'.

    The raw p values are identical on both sides. UD keeps the under-sampled
    predictor's raw p in the Benjamini-Hochberg pool and only blanks the
    reported values afterwards, so it corrects against k=4; Tools drops it
    before correcting, so it corrects against k=3.
    """
    frame = session_frame.copy()
    frame["sparse_metric"] = np.nan
    frame.loc[frame.index[:SPARSE_SAMPLE_COUNT], "sparse_metric"] = np.arange(
        float(SPARSE_SAMPLE_COUNT)
    )

    ud = analyze_variables(
        frame,
        _ud_request(
            predictors=SPARSE_PREDICTORS,
            analysis_mode="correlation",
            group_by=None,
        ),
    )
    tools = analyze_launch_monitor_data(
        frame,
        _tools_request(
            predictors=SPARSE_PREDICTORS,
            analysis_mode="correlation",
            group_by=None,
        ),
    )
    tools_by_predictor = {item.predictor: item for item in tools.correlations}

    for ud_item in ud.correlations:
        tools_item = tools_by_predictor[ud_item.predictor]
        assert ud_item.sample_count == tools_item.sample_count
        if ud_item.predictor == "sparse_metric":
            assert ud_item.sample_count == SPARSE_SAMPLE_COUNT
            continue
        # Same raw p value ...
        assert ud_item.p_value - tools_item.p_value == 0.0
        # ... different adjusted p value.
        assert ud_item.adjusted_p_value == pytest.approx(
            UD_ADJUSTED_WITH_SPARSE, rel=1e-12
        )
        assert tools_item.adjusted_p_value == pytest.approx(
            TOOLS_ADJUSTED_WITH_SPARSE, rel=1e-12
        )
        assert ud_item.adjusted_p_value > tools_item.adjusted_p_value


def test_divergence_d16_unknown_enum_values_fail_closed_only_in_ud(
    session_frame: pd.DataFrame,
) -> None:
    """DIFFER (D16): Tools silently computes Kendall for an unknown method."""
    for field, value in (
        ("analysis_mode", "bogus"),
        ("correlation_method", "bogus"),
        ("missing_policy", "bogus"),
    ):
        with pytest.raises(ValueError, match="Unknown"):
            _ud_request(**{field: value})

    bogus = analyze_launch_monitor_data(
        session_frame,
        _tools_request(
            correlation_method=cast(CorrelationMethod, "bogus"),
            analysis_mode="correlation",
            group_by=None,
            predictors=("session_order",),
        ),
    )
    kendall = analyze_launch_monitor_data(
        session_frame,
        _tools_request(
            correlation_method="kendall",
            analysis_mode="correlation",
            group_by=None,
            predictors=("session_order",),
        ),
    )
    assert bogus.correlations[0].coefficient == pytest.approx(
        TOOLS_BOGUS_METHOD_COEFFICIENT, rel=1e-12
    )
    assert bogus.correlations[0].coefficient == kendall.correlations[0].coefficient
    # The result still advertises the method the caller asked for.
    assert bogus.correlations[0].method == "bogus"

    degraded = analyze_launch_monitor_data(
        session_frame,
        _tools_request(analysis_mode=cast(AnalysisMode, "bogus"), group_by=None),
    )
    assert len(degraded.correlations) == len(PREDICTORS)
    assert degraded.regression is not None

    pairwise = analyze_launch_monitor_data(
        session_frame,
        _tools_request(
            missing_policy=cast(MissingPolicy, "bogus"),
            analysis_mode="correlation",
            group_by=None,
        ),
    )
    assert len(pairwise.correlations) == len(PREDICTORS)


def test_divergence_d17_boolean_columns_are_analysed_only_by_ud(
    session_frame: pd.DataFrame,
) -> None:
    """DIFFER (D17): UD projects booleans to 1/0; Tools reads them as null."""
    frame = session_frame.copy()
    frame["flagged"] = frame["shot_index"] % 2 == 0
    predictors = ("session_order", "flagged")

    ud = analyze_variables(
        frame,
        _ud_request(predictors=predictors, analysis_mode="correlation", group_by=None),
    )
    flagged = next(item for item in ud.correlations if item.predictor == "flagged")
    assert flagged.sample_count == EXPECTED_SAMPLE_COUNT
    assert flagged.coefficient == pytest.approx(UD_BOOLEAN_COEFFICIENT, rel=1e-12)
    assert ud.dataset.complete_row_count == EXPECTED_SAMPLE_COUNT

    with pytest.raises(ValueError, match=r"Constant variables cannot be analyzed"):
        analyze_launch_monitor_data(
            frame,
            _tools_request(
                predictors=predictors, analysis_mode="correlation", group_by=None
            ),
        )


def test_divergence_d18_blank_provenance_strings_are_dropped_only_by_ud(
    session_frame: pd.DataFrame,
) -> None:
    """DIFFER (D18): a whitespace-only ``session_id`` survives into Tools."""
    frame = session_frame.copy()
    frame.loc[frame.index[0], "session_id"] = "   "
    predictors = ("session_order", "lateral_carry_metres")

    ud = analyze_variables(
        frame,
        _ud_request(predictors=predictors, analysis_mode="correlation", group_by=None),
    )
    tools = analyze_launch_monitor_data(
        frame,
        _tools_request(
            predictors=predictors, analysis_mode="correlation", group_by=None
        ),
    )

    assert len(ud.dataset.session_ids) == SHARED_SESSION_ID_COUNT
    assert "   " not in ud.dataset.session_ids
    assert len(tools.dataset.session_ids) == SHARED_SESSION_ID_COUNT + 1
    assert "   " in tools.dataset.session_ids


def test_divergence_d19_result_surfaces_are_not_equal(
    session_frame: pd.DataFrame, ud_result, tools_result
) -> None:
    """DIFFER (D19): one extra diagnostic and a units map on one side; a
    contract field, a camelCase wire form and a column scan on the other."""
    ud_diagnostic_fields = set(
        type(ud_result.regression.residual_diagnostics).__dataclass_fields__
    )
    tools_diagnostic_fields = set(
        type(tools_result.regression.residual_diagnostics).__dataclass_fields__
    )
    assert ud_diagnostic_fields - tools_diagnostic_fields == {"jarque_bera_p_value"}
    assert tools_diagnostic_fields - ud_diagnostic_fields == set()
    assert ud_result.regression.residual_diagnostics.jarque_bera_p_value == (
        pytest.approx(UD_ONLY_JARQUE_BERA_P_VALUE, rel=1e-12)
    )

    ud_result_fields = set(type(ud_result).__dataclass_fields__)
    tools_result_fields = set(type(tools_result).__dataclass_fields__)
    assert ud_result_fields - tools_result_fields == {"units"}
    assert tools_result_fields - ud_result_fields == {"contract_version"}
    assert ud_result.units == dict.fromkeys((OUTCOME, *PREDICTORS), "source")
    assert tools_result.contract_version == TOOLS_CONTRACT_VERSION

    # Same payload, two key conventions; ``units`` exists on one side only.
    assert set(ud_result.to_dict()) == {
        "contract_version",
        "correlations",
        "dataset",
        "groups",
        "regression",
        "request",
        "units",
        "warnings",
    }
    assert set(tools_result.to_wire()) == {
        "contractVersion",
        "correlations",
        "dataset",
        "groups",
        "regression",
        "request",
        "warnings",
    }

    assert tuple(numeric_columns(session_frame)) == TOOLS_ONLY_NUMERIC_COLUMNS


def test_divergence_d20_missing_estimate_sentinel_differs(
    session_frame: pd.DataFrame,
) -> None:
    """DIFFER (D20): NaN versus None in Python, both ``null`` on the wire."""
    frame = session_frame.copy()
    frame["sparse_metric"] = np.nan
    frame.loc[frame.index[:SPARSE_SAMPLE_COUNT], "sparse_metric"] = np.arange(
        float(SPARSE_SAMPLE_COUNT)
    )
    request_overrides = {
        "predictors": SPARSE_PREDICTORS,
        "analysis_mode": "correlation",
        "group_by": None,
    }

    ud = analyze_variables(frame, _ud_request(**request_overrides))
    tools = analyze_launch_monitor_data(frame, _tools_request(**request_overrides))

    ud_sparse = next(
        item for item in ud.correlations if item.predictor == "sparse_metric"
    )
    tools_sparse = next(
        item for item in tools.correlations if item.predictor == "sparse_metric"
    )
    assert isinstance(ud_sparse.coefficient, float)
    assert np.isnan(ud_sparse.coefficient)
    assert tools_sparse.coefficient is None

    ud_wire = ud.to_dict()["correlations"][-1]
    tools_wire = tools.to_wire()["correlations"][-1]
    assert ud_wire["coefficient"] is None
    assert tools_wire["coefficient"] is None
    assert ud_wire["adjusted_p_value"] is None
    assert tools_wire["adjustedPValue"] is None
