"""Tests for sidekick.process_calculators.psa_package.psa_model (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from sidekick.process_calculators.psa_package.psa_model import (
    DEFAULT_COMPONENTS,
    PSAModel,
    PSAResults,
    StreamFlows,
)


class TestPSAModelConstruction:
    def test_psa_model_default_construction(self) -> None:
        model = PSAModel()
        assert model.total_feed_scfm == pytest.approx(1100.0)

    def test_default_s2_tail_recycle_frac(self) -> None:
        model = PSAModel()
        assert model.s2_tail_recycle_frac == pytest.approx(1.0)

    def test_default_product_recycle_frac(self) -> None:
        model = PSAModel()
        assert model.product_recycle_frac == pytest.approx(0.0)

    def test_default_components_nonempty(self) -> None:
        model = PSAModel()
        assert len(model.components) > 0

    def test_custom_feed_scfm(self) -> None:
        model = PSAModel(total_feed_scfm=500.0)
        assert model.total_feed_scfm == pytest.approx(500.0)


class TestDefaultComponents:
    def test_has_h2(self) -> None:
        names = [c["name"] for c in DEFAULT_COMPONENTS]
        assert "H2" in names

    def test_has_co(self) -> None:
        names = [c["name"] for c in DEFAULT_COMPONENTS]
        assert "CO" in names

    def test_feed_percentages_sum_to_100(self) -> None:
        total = sum(c["feed_pct"] for c in DEFAULT_COMPONENTS)
        assert total == pytest.approx(100.0, abs=1.0)

    def test_removal_pct_in_valid_range(self) -> None:
        for comp in DEFAULT_COMPONENTS:
            assert 0.0 <= comp["stage1_removal_pct"] <= 100.0
            assert 0.0 <= comp["stage2_removal_pct"] <= 100.0


class TestPSAModelCalculate:
    def setup_method(self) -> None:
        self.model = PSAModel(total_feed_scfm=1100.0)
        self.result = self.model.calculate()

    def test_returns_psa_results(self) -> None:
        assert isinstance(self.result, PSAResults)

    def test_h2_recovery_in_valid_range(self) -> None:
        assert 0.0 <= self.result.h2_recovery_pct <= 100.0

    def test_h2_purity_in_valid_range(self) -> None:
        assert 0.0 <= self.result.h2_purity_pct <= 100.0

    def test_total_net_product_positive(self) -> None:
        assert self.result.total_net_product_scfm >= 0.0

    def test_total_exhaust_positive(self) -> None:
        assert self.result.total_exhaust_scfm >= 0.0

    def test_mass_balance_small_error(self) -> None:
        # Mass balance error should be close to zero for a well-formulated model
        assert abs(self.result.mass_balance_error) < 10.0

    def test_component_names_match(self) -> None:
        assert "H2" in self.result.component_names
        assert "CO" in self.result.component_names

    def test_flows_is_stream_flows(self) -> None:
        assert isinstance(self.result.flows, StreamFlows)

    def test_fresh_feed_sums_to_total(self) -> None:
        total = np.sum(self.result.flows.fresh_feed)
        assert total == pytest.approx(1100.0, rel=1e-3)

    def test_high_s2_tail_recycle_improves_h2_recovery(self) -> None:
        model_full = PSAModel(s2_tail_recycle_frac=1.0)
        model_zero = PSAModel(s2_tail_recycle_frac=0.0)
        result_full = model_full.calculate()
        result_zero = model_zero.calculate()
        # Full recycle should give higher or equal H2 recovery
        assert result_full.h2_recovery_pct >= result_zero.h2_recovery_pct - 0.1

    def test_s2_tail_h2_pct_in_range(self) -> None:
        assert 0.0 <= self.result.s2_tail_h2_pct <= 100.0

    def test_s2_tail_o2_pct_in_range(self) -> None:
        assert 0.0 <= self.result.s2_tail_o2_pct <= 100.0
