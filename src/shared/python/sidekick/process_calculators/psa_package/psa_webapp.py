# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""
Streamlit Web App for Two-Stage PSA System Analysis.

This web application provides an interactive interface for PSA analysis
that can be shared with users who don't have Python installed.

Run with: streamlit run psa_webapp.py
"""

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from numpy.typing import NDArray

__all__ = [
    "ComponentData",
    "DEFAULT_COMPONENTS",
    "PSAModel",
    "PSAResults",
    "StreamCompositions",
    "StreamFlows",
    "get_flammability_status",
    "main",
]

# ============== PSA Model (embedded for standalone operation) ==============


class ComponentData(TypedDict):
    """Type definition for component feed data."""

    name: str
    feed_pct: float
    stage1_removal_pct: float
    stage2_removal_pct: float


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
    """Mass flows (SCFM) for each stream."""

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
class StreamCompositions:
    """Compositions (%) for each stream."""

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
    """Complete results from PSA model."""

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
    """Two-stage PSA model."""

    total_feed_scfm: float = 1100.0
    s2_tail_recycle_frac: float = 1.0
    product_recycle_frac: float = 0.0
    components: list[ComponentData] = field(
        default_factory=lambda: list(DEFAULT_COMPONENTS)
    )

    def calculate(self) -> PSAResults:
        """Perform PSA mass balance calculation."""
        n_components = len(self.components)
        component_names = [c["name"] for c in self.components]
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

        flows = StreamFlows(
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

        def calc_composition(flow_array: NDArray[np.float64]) -> NDArray[np.float64]:
            total: float = float(np.sum(flow_array))
            if total == 0:
                return np.zeros(n_components, dtype=np.float64)
            return flow_array / total * 100.0

        compositions = StreamCompositions(
            fresh_feed=calc_composition(fresh_feed),
            s2_tail_recycle=calc_composition(s2_tail_recycle),
            product_recycle=calc_composition(product_recycle),
            mixed_feed=calc_composition(mixed_feed),
            exhaust=calc_composition(exhaust),
            s2_tail_vent=calc_composition(s2_tail_vent),
            interstage=calc_composition(interstage),
            gross_product=calc_composition(gross_product),
            s2_tail=calc_composition(s2_tail),
            net_product=calc_composition(net_product),
        )

        h2_idx = component_names.index("H2")
        o2_idx = component_names.index("O2")

        return PSAResults(
            component_names=component_names,
            flows=flows,
            compositions=compositions,
            h2_recovery_pct=float(net_product[h2_idx] / fresh_feed[h2_idx] * 100.0),
            h2_purity_pct=float(compositions.net_product[h2_idx]),
            total_feed_scfm=self.total_feed_scfm,
            total_net_product_scfm=float(np.sum(net_product)),
            total_exhaust_scfm=float(np.sum(exhaust)),
            total_s2_tail_vent_scfm=float(np.sum(s2_tail_vent)),
            mass_balance_error=float(
                np.sum(fresh_feed)
                - np.sum(exhaust)
                - np.sum(s2_tail_vent)
                - np.sum(net_product)
            ),
            s2_tail_h2_pct=float(compositions.s2_tail[h2_idx]),
            s2_tail_o2_pct=float(compositions.s2_tail[o2_idx]),
        )


def get_flammability_status(h2_pct: float, o2_pct: float) -> tuple[str, str]:
    """Return status and color for flammability."""
    if h2_pct is None:
        raise ValueError("h2_pct must be provided")
    if o2_pct < 0.1:
        return "Safe-Low O2", "green"
    if h2_pct > 4 and o2_pct > 2:
        return "CRITICAL", "red"
    if h2_pct < 4:
        return "Safe-Below LFL", "green"
    if h2_pct > 75:
        return "Caution-Rich", "orange"
    return "FLAMMABLE", "red"


# ============== Streamlit App Sections ==============


def _resolve_plot_mode(show_lines: bool, show_markers: bool) -> str:
    """Resolve Plotly trace mode from boolean flags."""
    if show_lines is None:
        raise ValueError("show_lines must be provided")
    if show_lines and show_markers:
        return "lines+markers"
    if show_markers:
        return "markers"
    return "lines"


def _render_sidebar(
    components_template: list[ComponentData],
) -> tuple[float, int, int, list[ComponentData]]:
    """Render sidebar inputs and return operating parameters.

    Returns:
        Tuple of (total_feed, s2_recycle, prod_recycle, components).
    """
    st.sidebar.title("🔧 Operating Parameters")

    total_feed = st.sidebar.slider("Total Feed (SCFM)", 500, 2000, 1100, 50)
    s2_recycle = st.sidebar.slider("S2 Tail Recycle (%)", 0, 100, 100, 5)
    prod_recycle = st.sidebar.slider("Product Recycle (%)", 0, 50, 0, 5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Feed Composition")

    feed_h2 = st.sidebar.number_input("H2 (%)", 0.0, 100.0, 32.08, 0.1)
    feed_o2 = st.sidebar.number_input("O2 (%)", 0.0, 10.0, 0.50, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Stage 1 Removal")

    s1_h2_removal = st.sidebar.slider("H2 Removal (%)", 0, 50, 18)
    s1_o2_removal = st.sidebar.slider("O2 Removal (%)", 50, 99, 81)

    # Build component list with user overrides
    components: list[ComponentData] = [
        {
            "name": "H2",
            "feed_pct": feed_h2,
            "stage1_removal_pct": float(s1_h2_removal),
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
            "feed_pct": feed_o2,
            "stage1_removal_pct": float(s1_o2_removal),
            "stage2_removal_pct": 99.99,
        },
        {
            "name": "CH4",
            "feed_pct": 1.88,
            "stage1_removal_pct": 99.0,
            "stage2_removal_pct": 99.99,
        },
    ]

    return float(total_feed), s2_recycle, prod_recycle, components


def _render_key_metrics(results: PSAResults) -> None:
    """Render the top-level KPI metric tiles."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("H2 Recovery", f"{results.h2_recovery_pct:.2f}%")
    with col2:
        st.metric("H2 Purity", f"{results.h2_purity_pct:.4f}%")
    with col3:
        st.metric("Net Product", f"{results.total_net_product_scfm:.1f} SCFM")
    with col4:
        status, color = get_flammability_status(
            results.s2_tail_h2_pct, results.s2_tail_o2_pct
        )
        if color == "red":
            st.error(f"⚠️ {status}")
        elif color == "orange":
            st.warning(f"⚠️ {status}")
        else:
            st.success(f"✅ {status}")


def _render_results_tab(results: PSAResults) -> None:
    """Render Tab 1 — mass balance summary and Sankey flow diagram."""
    st.subheader("Mass Balance Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Flow Summary (SCFM)**")
        flow_summary = {
            "Stream": ["Feed", "Exhaust", "Net Product", "S2 Tail Vent"],
            "Flow (SCFM)": [
                results.total_feed_scfm,
                results.total_exhaust_scfm,
                results.total_net_product_scfm,
                results.total_s2_tail_vent_scfm,
            ],
        }
        st.dataframe(pd.DataFrame(flow_summary), hide_index=True)

    with col2:
        st.markdown("**Safety Metrics**")
        st.write(f"S2 Tail H2: {results.s2_tail_h2_pct:.2f}%")
        st.write(f"S2 Tail O2: {results.s2_tail_o2_pct:.2f}%")
        st.write(f"Mass Balance Error: {results.mass_balance_error:.2e}")

    # Sankey diagram
    st.subheader("Flow Diagram")

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                    "label": [
                        "Feed",
                        "PSA 1",
                        "PSA 2",
                        "Exhaust",
                        "Net Product",
                        "S2 Recycle",
                    ],
                    "color": ["blue", "gray", "gray", "red", "green", "orange"],
                },
                link={
                    "source": [0, 1, 1, 2, 2],
                    "target": [1, 3, 2, 4, 5],
                    "value": [
                        results.total_feed_scfm,
                        results.total_exhaust_scfm,
                        float(np.sum(results.flows.interstage)),
                        results.total_net_product_scfm,
                        float(np.sum(results.flows.s2_tail_recycle)),
                    ],
                },
            )
        ]
    )
    fig.update_layout(title_text="PSA System Flow", font_size=12)
    st.plotly_chart(fig, use_container_width=True)


def _render_sensitivity_tab(
    total_feed: float,
    s2_recycle: int,
    prod_recycle: int,
    components: list[ComponentData],
) -> None:
    """Render Tab 2 — S2 tail recycle sensitivity analysis."""
    if total_feed is None:
        raise ValueError("total_feed must be provided")
    st.subheader("Sensitivity Analysis")

    # Plot options
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        show_lines = st.checkbox("Show Lines", value=True, key="sens_lines")
    with opt_col2:
        show_markers = st.checkbox("Show Markers", value=False, key="sens_markers")
    with opt_col3:
        num_points = st.slider("Number of Points", 11, 101, 51, 10, key="sens_points")

    plot_mode = _resolve_plot_mode(show_lines, show_markers)

    # Calculate sensitivity
    s2_range = np.linspace(0, 1, num_points)
    h2_recovery_data = []
    net_product_data = []

    for r in s2_range:
        m = PSAModel(
            total_feed_scfm=total_feed,
            s2_tail_recycle_frac=float(r),
            product_recycle_frac=prod_recycle / 100.0,
            components=components,
        )
        res = m.calculate()
        h2_recovery_data.append(res.h2_recovery_pct)
        net_product_data.append(res.total_net_product_scfm)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=s2_range * 100,
                y=h2_recovery_data,
                mode=plot_mode,
                name="H2 Recovery",
                line={"width": 2},
                marker={"size": 6},
            )
        )
        fig1.add_vline(x=s2_recycle, line_dash="dash", line_color="red")
        fig1.update_layout(
            title="H2 Recovery vs S2 Tail Recycle",
            xaxis_title="S2 Tail Recycle (%)",
            yaxis_title="H2 Recovery (%)",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=s2_range * 100,
                y=net_product_data,
                mode=plot_mode,
                name="Net Product",
                line={"width": 2},
                marker={"size": 6},
            )
        )
        fig2.add_vline(x=s2_recycle, line_dash="dash", line_color="red")
        fig2.update_layout(
            title="Net Product vs S2 Tail Recycle",
            xaxis_title="S2 Tail Recycle (%)",
            yaxis_title="Net Product (SCFM)",
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_o2_safety_tab(
    total_feed: float,
    components: list[ComponentData],
) -> None:
    """Render Tab 3 — O2 flammability / safety analysis."""
    if total_feed is None:
        raise ValueError("total_feed must be provided")
    st.subheader("O2 Safety Analysis")
    st.markdown("""
    **Critical Thresholds:**
    - H2 LFL: 4%, UFL: 75%
    - O2 Danger Level: >2% with H2 >4%
    """)

    # Plot options for O2 safety
    o2_col1, o2_col2, o2_col3 = st.columns(3)
    with o2_col1:
        o2_show_lines = st.checkbox("Show Lines", value=True, key="o2_lines")
    with o2_col2:
        o2_show_markers = st.checkbox("Show Markers", value=False, key="o2_markers")
    with o2_col3:
        o2_num_points = st.slider("Number of Points", 11, 51, 21, 5, key="o2_points")

    o2_plot_mode = _resolve_plot_mode(o2_show_lines, o2_show_markers)

    # O2 analysis
    inlet_o2_values = [0.5, 1.0, 2.0, 5.0]
    s1_removal_range = np.linspace(50, 95, o2_num_points)

    o2_data = []
    for s1_rem in s1_removal_range:
        row: dict[str, float] = {"S1 O2 Removal (%)": float(s1_rem)}
        for inlet_o2 in inlet_o2_values:
            mod_components: list[ComponentData] = [
                ComponentData(
                    name=c["name"],
                    feed_pct=c["feed_pct"],
                    stage1_removal_pct=c["stage1_removal_pct"],
                    stage2_removal_pct=c["stage2_removal_pct"],
                )
                for c in components
            ]
            for c in mod_components:
                if c["name"] == "O2":
                    c["feed_pct"] = inlet_o2
                    c["stage1_removal_pct"] = float(s1_rem)
            m = PSAModel(
                total_feed_scfm=total_feed,
                s2_tail_recycle_frac=1.0,
                product_recycle_frac=0.0,
                components=mod_components,
            )
            row[f"{inlet_o2}% Inlet"] = m.calculate().s2_tail_o2_pct
        o2_data.append(row)

    df_o2 = pd.DataFrame(o2_data)

    # Line plot with options
    fig_line = go.Figure()
    for inlet_o2 in inlet_o2_values:
        fig_line.add_trace(
            go.Scatter(
                x=s1_removal_range,
                y=df_o2[f"{inlet_o2}% Inlet"],
                mode=o2_plot_mode,
                name=f"{inlet_o2}% Inlet O2",
                line={"width": 2},
                marker={"size": 6},
            )
        )
    fig_line.add_hline(
        y=2.0, line_dash="dash", line_color="red", annotation_text="Danger (2%)"
    )
    fig_line.update_layout(
        title="S2 Tail O2% vs Stage 1 O2 Removal",
        xaxis_title="Stage 1 O2 Removal (%)",
        yaxis_title="S2 Tail O2 (%)",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Heatmap
    fig = px.imshow(
        df_o2.set_index("S1 O2 Removal (%)").T,
        labels={
            "x": "S1 O2 Removal (%)",
            "y": "Inlet O2 (%)",
            "color": "S2 Tail O2 %",
        },
        title="S2 Tail O2% Heatmap",
        color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table with highlighting
    st.markdown("**Detailed Values (Red = Dangerous >2%)**")

    def highlight_danger(val: float) -> str:
        if isinstance(val, int | float) and val > 2.0:
            return "background-color: #ffcccc"
        elif isinstance(val, int | float) and val > 1.5:
            return "background-color: #ffffcc"
        return ""

    st.dataframe(
        df_o2.style.map(highlight_danger, subset=df_o2.columns[1:]).format(precision=2)
    )


def _render_data_tables_tab(results: PSAResults) -> None:
    """Render Tab 4 — detailed mass balance and composition tables."""
    st.subheader("Detailed Data Tables")

    # Mass balance table
    st.markdown("**Mass Balance (SCFM)**")
    mass_df = pd.DataFrame(
        {
            "Component": results.component_names,
            "Fresh Feed": results.flows.fresh_feed,
            "Mixed Feed": results.flows.mixed_feed,
            "Exhaust": results.flows.exhaust,
            "Interstage": results.flows.interstage,
            "Net Product": results.flows.net_product,
        }
    )
    st.dataframe(mass_df.style.format(precision=4))

    # Composition table
    st.markdown("**Compositions (%)**")
    comp_df = pd.DataFrame(
        {
            "Component": results.component_names,
            "Fresh Feed": results.compositions.fresh_feed,
            "Mixed Feed": results.compositions.mixed_feed,
            "Exhaust": results.compositions.exhaust,
            "Interstage": results.compositions.interstage,
            "Net Product": results.compositions.net_product,
        }
    )
    st.dataframe(comp_df.style.format(precision=4))


# ============== Streamlit App ==============


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PSA System Analysis",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar — collect operating parameters
    total_feed, s2_recycle, prod_recycle, components = _render_sidebar(
        DEFAULT_COMPONENTS
    )

    # Calculate
    model = PSAModel(
        total_feed_scfm=total_feed,
        s2_tail_recycle_frac=s2_recycle / 100.0,
        product_recycle_frac=prod_recycle / 100.0,
        components=components,
    )
    results = model.calculate()

    # Main content
    st.title("🔬 Two-Stage PSA System Analysis")

    _render_key_metrics(results)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Results", "📈 Sensitivity", "⚠️ O2 Safety", "📋 Data Tables"]
    )

    with tab1:
        _render_results_tab(results)

    with tab2:
        _render_sensitivity_tab(total_feed, s2_recycle, prod_recycle, components)

    with tab3:
        _render_o2_safety_tab(total_feed, components)

    with tab4:
        _render_data_tables_tab(results)

    # Footer
    st.markdown("---")
    st.markdown(
        "*PSA System Analysis Tool - "
        "All calculations validated against Excel reference model.*"
    )


if __name__ == "__main__":
    main()
