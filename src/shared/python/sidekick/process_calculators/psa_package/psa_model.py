"""
Two-Stage PSA (Pressure Swing Adsorption) Model.

This module provides calculations for a two-stage PSA system with recycle streams.
The model solves the mass balance algebraically to avoid circular references.

Stream numbering (per PFD):
    1   - Fresh Feed (from gasifier)
    2   - Exhaust (PSA 1 tail)
    3G  - Gross Product (PSA 2 output before split)
    3N  - Net Product (final product)
    3R  - Product Recycle (back to feed)
    4   - Stage 2 Tail Recycle
    5A  - Mixed Feed (Fresh + Recycles)
    5B  - Mixed Feed (after combining)
    5C  - Feed to PSA 1 (after compressor)
    6   - Interstage (PSA 1 product to PSA 2)
"""

import logging
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ComponentData",
    "DEFAULT_COMPONENTS",
    "PSAModel",
    "PSAResults",
    "StreamCompositions",
    "StreamFlows",
    "calculate_o2_safety_analysis",
    "calculate_sensitivity",
    "get_flammability_status",
]

_logger = logging.getLogger(__name__)


class ComponentData(TypedDict):
    """Type definition for component feed data."""

    name: str
    feed_pct: float
    stage1_removal_pct: float
    stage2_removal_pct: float


# Default component data matching the Excel model
DEFAULT_COMPONENTS: list[ComponentData] = [
    {
        "name": "H2",
        "feed_pct": 32.08,
        "stage1_removal_pct": 18.0,
        "stage2_removal_pct": 15.0,
    },
    {
        "name": "CO",
        "feed_pct": 38.22,
        "stage1_removal_pct": 98.0,
        "stage2_removal_pct": 99.99,
    },
    {
        "name": "CO2",
        "feed_pct": 21.98,
        "stage1_removal_pct": 98.0,
        "stage2_removal_pct": 99.99,
    },
    {
        "name": "H2O",
        "feed_pct": 4.85,
        "stage1_removal_pct": 99.0,
        "stage2_removal_pct": 99.99,
    },
    {
        "name": "N2",
        "feed_pct": 0.50,
        "stage1_removal_pct": 95.0,
        "stage2_removal_pct": 99.99,
    },
    {
        "name": "O2",
        "feed_pct": 0.50,
        "stage1_removal_pct": 81.0,
        "stage2_removal_pct": 99.99,
    },
    {
        "name": "CH4",
        "feed_pct": 1.88,
        "stage1_removal_pct": 99.0,
        "stage2_removal_pct": 99.99,
    },
]


@dataclass
class StreamFlows:
    """Mass flows (SCFM) for each stream in the PSA system."""

    fresh_feed: NDArray[np.float64]  # Stream 1
    s2_tail_recycle: NDArray[np.float64]  # Stream 4 (C)
    product_recycle: NDArray[np.float64]  # Stream 3R (D)
    mixed_feed: NDArray[np.float64]  # Stream 5A (E)
    exhaust: NDArray[np.float64]  # Stream 2 (F)
    s2_tail_vent: NDArray[np.float64]  # Stream G
    interstage: NDArray[np.float64]  # Stream 6 (H)
    gross_product: NDArray[np.float64]  # Stream 3G (I)
    s2_tail: NDArray[np.float64]  # Stream J
    net_product: NDArray[np.float64]  # Stream 3N (K)


@dataclass
class StreamCompositions:
    """Compositions (%) for each stream in the PSA system."""

    fresh_feed: NDArray[np.float64]
    s2_tail_recycle: NDArray[np.float64]
    product_recycle: NDArray[np.float64]
    mixed_feed: NDArray[np.float64]
    exhaust: NDArray[np.float64]
    s2_tail_vent: NDArray[np.float64]
    interstage: NDArray[np.float64]
    gross_product: NDArray[np.float64]
    s2_tail: NDArray[np.float64]
    net_product: NDArray[np.float64]


@dataclass
class PSAResults:
    """Complete results from PSA model calculation."""

    component_names: list[str]
    flows: StreamFlows
    compositions: StreamCompositions
    h2_recovery_pct: float
    h2_purity_pct: float
    total_feed_scfm: float
    total_net_product_scfm: float
    total_exhaust_scfm: float
    total_s2_tail_vent_scfm: float
    mass_balance_error: float
    s2_tail_h2_pct: float
    s2_tail_o2_pct: float


@dataclass
class PSAModel:
    """
    Two-stage PSA model with algebraic solution for recycle streams.

    The key equation for mixed feed (eliminating circular references):
        M_i = F_i / [1 - (1-R1_i) * (R2_i * r_tail + (1-R2_i) * r_prod)]

    Where:
        F_i = Fresh feed flow of component i
        M_i = Mixed feed flow of component i
        R1_i = Stage 1 removal fraction for component i
        R2_i = Stage 2 removal fraction for component i
        r_tail = Stage 2 tail recycle fraction (0 to 1)
        r_prod = Product recycle fraction (0 to 1)
    """

    total_feed_scfm: float = 1100.0
    s2_tail_recycle_frac: float = 1.0
    product_recycle_frac: float = 0.0
    components: list[ComponentData] = field(
        default_factory=lambda: list(DEFAULT_COMPONENTS)
    )

    def _compute_stream_flows(
        self,
    ) -> StreamFlows:
        """Solve the algebraic mass balance for all PSA streams.

        Returns:
            StreamFlows with computed flow arrays for each stream.
        """
        feed_pct = np.array([c["feed_pct"] for c in self.components], dtype=np.float64)
        r1 = np.array(
            [c["stage1_removal_pct"] / 100.0 for c in self.components], dtype=np.float64
        )
        r2 = np.array(
            [c["stage2_removal_pct"] / 100.0 for c in self.components], dtype=np.float64
        )

        r_tail = self.s2_tail_recycle_frac
        r_prod = self.product_recycle_frac

        fresh_feed = self.total_feed_scfm * feed_pct / np.sum(feed_pct)
        denominator = 1.0 - (1.0 - r1) * (r2 * r_tail + (1.0 - r2) * r_prod)
        mixed_feed = fresh_feed / denominator

        exhaust = mixed_feed * r1
        interstage = mixed_feed - exhaust
        s2_tail = interstage * r2
        s2_tail_recycle = s2_tail * r_tail
        s2_tail_vent = s2_tail * (1.0 - r_tail)
        gross_product = interstage - s2_tail
        product_recycle = gross_product * r_prod
        net_product = gross_product * (1.0 - r_prod)

        return StreamFlows(
            fresh_feed=fresh_feed,
            s2_tail_recycle=s2_tail_recycle,
            product_recycle=product_recycle,
            mixed_feed=mixed_feed,
            exhaust=exhaust,
            s2_tail_vent=s2_tail_vent,
            interstage=interstage,
            gross_product=gross_product,
            s2_tail=s2_tail,
            net_product=net_product,
        )

    @staticmethod
    def _compute_performance_metrics(
        component_names: list[str],
        flows: StreamFlows,
        n_components: int,
    ) -> tuple[
        StreamCompositions, float, float, float, float, float, float, float, float
    ]:
        """Compute compositions and key performance metrics from stream flows.

        Returns:
            Tuple of (compositions, h2_recovery_pct, h2_purity_pct,
            total_net_product, total_exhaust, total_s2_tail_vent,
            mass_balance_error, s2_tail_h2_pct, s2_tail_o2_pct)
        """

        if component_names is None:
            raise ValueError("component_names must be provided")

        def calc_composition(flow_array: NDArray[np.float64]) -> NDArray[np.float64]:
            total: float = float(np.sum(flow_array))
            if total == 0:
                return np.zeros(n_components, dtype=np.float64)
            return flow_array / total * 100.0

        compositions = StreamCompositions(
            fresh_feed=calc_composition(flows.fresh_feed),
            s2_tail_recycle=calc_composition(flows.s2_tail_recycle),
            product_recycle=calc_composition(flows.product_recycle),
            mixed_feed=calc_composition(flows.mixed_feed),
            exhaust=calc_composition(flows.exhaust),
            s2_tail_vent=calc_composition(flows.s2_tail_vent),
            interstage=calc_composition(flows.interstage),
            gross_product=calc_composition(flows.gross_product),
            s2_tail=calc_composition(flows.s2_tail),
            net_product=calc_composition(flows.net_product),
        )

        total_net_product = float(np.sum(flows.net_product))
        total_exhaust = float(np.sum(flows.exhaust))
        total_s2_tail_vent = float(np.sum(flows.s2_tail_vent))

        mass_balance_error = float(
            np.sum(flows.fresh_feed)
            - np.sum(flows.exhaust)
            - np.sum(flows.s2_tail_vent)
            - np.sum(flows.net_product)
        )

        h2_idx = component_names.index("H2")
        h2_recovery_pct = float(
            flows.net_product[h2_idx] / flows.fresh_feed[h2_idx] * 100.0
        )
        h2_purity_pct = float(compositions.net_product[h2_idx])

        s2_tail_h2_pct = float(compositions.s2_tail[h2_idx])
        o2_idx = component_names.index("O2")
        s2_tail_o2_pct = float(compositions.s2_tail[o2_idx])

        return (
            compositions,
            h2_recovery_pct,
            h2_purity_pct,
            total_net_product,
            total_exhaust,
            total_s2_tail_vent,
            mass_balance_error,
            s2_tail_h2_pct,
            s2_tail_o2_pct,
        )

    def calculate(self) -> PSAResults:
        """
        Perform the PSA mass balance calculation.

        Returns:
            PSAResults containing all stream flows, compositions, and metrics.
        """
        component_names = [c["name"] for c in self.components]
        n_components = len(self.components)

        flows = self._compute_stream_flows()
        (
            compositions,
            h2_recovery_pct,
            h2_purity_pct,
            total_net_product,
            total_exhaust,
            total_s2_tail_vent,
            mass_balance_error,
            s2_tail_h2_pct,
            s2_tail_o2_pct,
        ) = self._compute_performance_metrics(component_names, flows, n_components)

        return PSAResults(
            component_names=component_names,
            flows=flows,
            compositions=compositions,
            h2_recovery_pct=h2_recovery_pct,
            h2_purity_pct=h2_purity_pct,
            total_feed_scfm=self.total_feed_scfm,
            total_net_product_scfm=total_net_product,
            total_exhaust_scfm=total_exhaust,
            total_s2_tail_vent_scfm=total_s2_tail_vent,
            mass_balance_error=mass_balance_error,
            s2_tail_h2_pct=s2_tail_h2_pct,
            s2_tail_o2_pct=s2_tail_o2_pct,
        )


def calculate_sensitivity(
    total_feed: float = 1100.0,
    s2_tail_recycle_range: NDArray[np.float64] | None = None,
    product_recycle_range: NDArray[np.float64] | None = None,
    components: list[ComponentData] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Calculate sensitivity analysis over range of recycle fractions.

    Args:
        total_feed: Total feed flow rate (SCFM)
        s2_tail_recycle_range: Array of S2 tail recycle fractions to evaluate
        product_recycle_range: Array of product recycle fractions to evaluate
        components: Component data (uses defaults if None)

    Returns:
        Dictionary with arrays for each metric vs recycle fractions
    """
    if total_feed is None:
        raise ValueError("total_feed must be provided")

    _s2_tail = s2_tail_recycle_range
    if _s2_tail is None:
        _s2_tail = np.linspace(0.0, 1.0, 11, dtype=np.float64)

    _prod_recycle = product_recycle_range
    if _prod_recycle is None:
        _prod_recycle = np.array([0.0], dtype=np.float64)

    _components = components
    if _components is None:
        _components = list(DEFAULT_COMPONENTS)

    n_tail = len(_s2_tail)
    n_prod = len(_prod_recycle)

    h2_recovery: NDArray[np.float64] = np.zeros((n_tail, n_prod), dtype=np.float64)
    h2_purity: NDArray[np.float64] = np.zeros((n_tail, n_prod), dtype=np.float64)
    net_product: NDArray[np.float64] = np.zeros((n_tail, n_prod), dtype=np.float64)
    s2_tail_o2: NDArray[np.float64] = np.zeros((n_tail, n_prod), dtype=np.float64)

    for i, r_tail in enumerate(_s2_tail):
        for j, r_prod in enumerate(_prod_recycle):
            model = PSAModel(
                total_feed_scfm=total_feed,
                s2_tail_recycle_frac=float(r_tail),
                product_recycle_frac=float(r_prod),
                components=_components,
            )
            results = model.calculate()
            h2_recovery[i, j] = results.h2_recovery_pct
            h2_purity[i, j] = results.h2_purity_pct
            net_product[i, j] = results.total_net_product_scfm
            s2_tail_o2[i, j] = results.s2_tail_o2_pct

    return {
        "s2_tail_recycle": _s2_tail,
        "product_recycle": _prod_recycle,
        "h2_recovery": h2_recovery,
        "h2_purity": h2_purity,
        "net_product": net_product,
        "s2_tail_o2": s2_tail_o2,
    }


def calculate_o2_safety_analysis(
    inlet_o2_pcts: NDArray[np.float64] | None = None,
    stage1_o2_removal_range: NDArray[np.float64] | None = None,
    total_feed: float = 1100.0,
    components: list[ComponentData] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Calculate O2 safety analysis - S2 Tail O2% vs Stage 1 O2 removal efficiency.

    Args:
        inlet_o2_pcts: Array of inlet O2 percentages to evaluate
        stage1_o2_removal_range: Array of Stage 1 O2 removal percentages
        total_feed: Total feed flow rate (SCFM)
        components: Base component data (O2 feed % will be varied)

    Returns:
        Dictionary with S2 Tail O2% for each inlet O2% and S1 removal%
    """
    if total_feed is None:
        raise ValueError("total_feed must be provided")
    if inlet_o2_pcts is None:
        inlet_o2_pcts = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    if stage1_o2_removal_range is None:
        stage1_o2_removal_range = np.arange(50.0, 100.0, 5.0, dtype=np.float64)
    if components is None:
        components = list(DEFAULT_COMPONENTS)

    # Create local variables that are guaranteed to be non-None
    _inlet_o2_pcts = inlet_o2_pcts
    _stage1_o2_removal_range = stage1_o2_removal_range

    n_inlet = len(_inlet_o2_pcts)
    n_removal = len(_stage1_o2_removal_range)

    s2_tail_o2: NDArray[np.float64] = np.zeros((n_removal, n_inlet), dtype=np.float64)

    for i, s1_removal in enumerate(_stage1_o2_removal_range):
        for j, inlet_o2 in enumerate(_inlet_o2_pcts):
            # Create modified components with new O2 feed% and S1 removal%
            modified_components: list[ComponentData] = []
            for c in components:
                # Explicitly construct a new ComponentData to preserve typing
                new_c: ComponentData = {
                    "name": c["name"],
                    "feed_pct": float(c["feed_pct"]),
                    "stage1_removal_pct": float(c["stage1_removal_pct"]),
                    "stage2_removal_pct": float(c["stage2_removal_pct"]),
                }
                if c["name"] == "O2":
                    new_c["feed_pct"] = float(inlet_o2)
                    new_c["stage1_removal_pct"] = float(s1_removal)
                modified_components.append(new_c)

            model = PSAModel(
                total_feed_scfm=total_feed,
                s2_tail_recycle_frac=1.0,  # Full recycle for worst case
                product_recycle_frac=0.0,
                components=modified_components,
            )
            results = model.calculate()
            s2_tail_o2[i, j] = results.s2_tail_o2_pct

    return {
        "inlet_o2_pcts": _inlet_o2_pcts,
        "stage1_o2_removal": _stage1_o2_removal_range,
        "s2_tail_o2": s2_tail_o2,
    }


def get_flammability_status(h2_pct: float, o2_pct: float) -> str:
    """
    Determine flammability status based on H2 and O2 concentrations.

    H2 flammability limits: LFL 4%, UFL 75%
    Critical O2 threshold: 2% for H2/O2 mixtures

    Args:
        h2_pct: Hydrogen concentration (%)
        o2_pct: Oxygen concentration (%)

    Returns:
        Status string indicating safety level
    """
    if h2_pct is None:
        raise ValueError("h2_pct must be provided")
    if o2_pct < 0.1:
        return "Safe-Low O2"
    if h2_pct > 4 and o2_pct > 2:
        return "CRITICAL"
    if h2_pct < 4:
        return "Safe-Below LFL"
    if h2_pct > 75:
        return "Caution-Rich"
    return "FLAMMABLE"


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    model = PSAModel()
    results = model.calculate()
    _logger.info(f"H2 Recovery: {results.h2_recovery_pct:.2f}%")
    _logger.info(f"H2 Purity: {results.h2_purity_pct:.5f}%")
    _logger.info(f"Net Product: {results.total_net_product_scfm:.2f} SCFM")
    _logger.info(f"Mass Balance Error: {results.mass_balance_error:.2e}")
