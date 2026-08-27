from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.run_event_topology_channel_matrix import (
    BASE_DT_S,
    COMMON_HORIZON_S,
    PREREGISTRATION_COMMENT,
    registered_horizons_s,
    registered_step_sizes_s,
    summarize_horizon_controls,
    summarize_step_refinement,
    validate_report,
)


def test_phase_c_numerical_matrix_is_the_preregistered_fixed_design() -> None:
    assert PREREGISTRATION_COMMENT.endswith("issuecomment-5431439586")
    assert pytest.approx(0.002) == BASE_DT_S
    assert pytest.approx(0.60) == COMMON_HORIZON_S
    assert registered_step_sizes_s() == pytest.approx((0.001, 0.002, 0.004))
    assert registered_horizons_s() == pytest.approx((0.40, 0.60, 0.80))


def test_phase_c_validator_rejects_implicit_work_or_coaching_promotion() -> None:
    report = {
        "schema_version": "proximal-distal-event-topology-channel-matrix/v1",
        "source_identity": {},
        "registration": {
            "preregistration_comment": PREREGISTRATION_COMMENT,
            "base_dt_s": BASE_DT_S,
            "common_horizon_s": COMMON_HORIZON_S,
            "step_sizes_s": list(registered_step_sizes_s()),
            "horizons_s": list(registered_horizons_s()),
            "fixed_stop_rule_completed": True,
        },
        "channel_maps": [],
        "step_refinement": [],
        "horizon_controls": [],
        "availability": {
            "work_power": "available",
            "coaching_recommendation": "unavailable",
        },
        "inference_boundary": "synthetic channel masks are not anatomical or coaching evidence",
    }

    with pytest.raises(ValueError, match="work/power"):
        validate_report(report, verify_source=False)


def test_phase_c_validator_requires_complete_registered_matrix() -> None:
    report = {
        "schema_version": "proximal-distal-event-topology-channel-matrix/v1",
        "source_identity": {},
        "registration": {
            "preregistration_comment": PREREGISTRATION_COMMENT,
            "base_dt_s": BASE_DT_S,
            "common_horizon_s": COMMON_HORIZON_S,
            "step_sizes_s": list(registered_step_sizes_s()),
            "horizons_s": list(registered_horizons_s()),
            "fixed_stop_rule_completed": True,
        },
        "channel_maps": [],
        "step_refinement": [],
        "horizon_controls": [],
        "availability": {
            "work_power": "unavailable",
            "coaching_recommendation": "unavailable",
        },
        "inference_boundary": "synthetic channel masks are not anatomical or coaching evidence",
    }

    with pytest.raises(ValueError, match="channel map"):
        validate_report(report, verify_source=False)


def test_step_summary_distinguishes_identity_stability_from_metric_residuals() -> None:
    records = [
        {
            "channel": "both",
            "dt_s": dt_s,
            "outcomes": [
                {
                    "delay_s": 0.0,
                    "status": "unique_transverse",
                    "crossing_count": 1,
                    "events": [
                        {
                            "direction": "negative_to_nonnegative",
                            "event_time_s": event_time,
                            "event_state": [0.1, -0.1, 1.0, 2.0],
                            "clubhead_speed_m_s": 4.0,
                            "transversality_per_s": 3.0,
                        }
                    ],
                }
            ],
        }
        for dt_s, event_time in ((0.001, 0.300001), (0.002, 0.3), (0.004, 0.300004))
    ]

    summary = summarize_step_refinement(records)

    assert summary[0]["channel"] == "both"
    assert summary[0]["topology_identity_all_match"] is True
    assert summary[0]["maximum_event_time_residual_s"] == pytest.approx(4e-6)


def test_horizon_summary_types_original_horizon_truncation() -> None:
    records = [
        {
            "channel": "wrist_only",
            "horizon_s": horizon,
            "status": status,
            "crossing_count": count,
            "events": events,
        }
        for horizon, status, count, events in (
            (0.4, "absent", 0, []),
            (
                0.6,
                "unique_transverse",
                1,
                [{"direction": "negative_to_nonnegative"}],
            ),
            (
                0.8,
                "unique_transverse",
                1,
                [{"direction": "negative_to_nonnegative"}],
            ),
        )
    ]

    summary = summarize_horizon_controls(records)

    assert summary[0]["expanded_horizon_identity_stable"] is True
    assert summary[0]["original_horizon_differs"] is True
    assert summary[0]["interpretation"] == "original_horizon_truncation"
