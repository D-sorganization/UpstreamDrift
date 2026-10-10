"""Swing comparison report rendering module (Issue #11164).

Provides:
- comparison_to_markdown: Render a ComparisonReport into Title Case GitHub-flavored Markdown.
- comparison_to_dict: Convert a ComparisonReport into a pure JSON-serializable dictionary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.python.swing_comparison.metrics import ComparisonReport


def comparison_to_dict(report: ComparisonReport) -> dict[str, Any]:
    """Convert ComparisonReport into a JSON-serializable dictionary.

    Args:
        report: ComparisonReport instance.

    Returns:
        Dictionary containing all report metrics, events, differences, and trajectory RMS.
    """
    clean_diffs: dict[str, Any] = {}
    for k, v in report.differences.items():
        if v is None:
            clean_diffs[k] = None
        elif isinstance(v, (int, float)):
            clean_diffs[k] = float(v)
        else:
            clean_diffs[k] = v

    return {
        "motion_a_events": report.motion_a_events.to_dict(),
        "motion_b_events": report.motion_b_events.to_dict(),
        "metrics_a": report.metrics_a.to_dict(),
        "metrics_b": report.metrics_b.to_dict(),
        "differences": clean_diffs,
        "shared_marker_rms": {k: float(v) for k, v in report.shared_marker_rms.items()},
        "mean_marker_rms": float(report.mean_marker_rms),
    }


def _fmt(val: float | int | None, unit: str = "", decimals: int = 2) -> str:
    """Format an optional numerical value with unit."""
    if val is None or val != val:
        return "N/A"
    return f"{val:.{decimals}f}{unit}"


def comparison_to_markdown(
    report: ComparisonReport,
    title: str = "Swing Comparison Report",
) -> str:
    """Render ComparisonReport into GitHub-flavored Markdown with Title Case headings.

    Args:
        report: ComparisonReport instance.
        title: Title of the document.

    Returns:
        Formatted Markdown string.
    """
    ev_a = report.motion_a_events
    ev_b = report.motion_b_events
    m_a = report.metrics_a
    m_b = report.metrics_b
    diff = report.differences

    lines: list[str] = [
        f"# {title}",
        "",
        "## Key Swing Events and Tempo",
        "",
        "| Event / Metric | Swing A (Ref) | Swing B (Cand) | Difference (B - A) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Address Time | {_fmt(ev_a.address_time, ' s')} | {_fmt(ev_b.address_time, ' s')} | {_fmt(ev_b.address_time - ev_a.address_time, ' s')} |",
        f"| Top of Backswing Time | {_fmt(ev_a.top_time, ' s')} | {_fmt(ev_b.top_time, ' s')} | {_fmt(ev_b.top_time - ev_a.top_time, ' s')} |",
        f"| Impact Time | {_fmt(ev_a.impact_time, ' s')} | {_fmt(ev_b.impact_time, ' s')} | {_fmt(ev_b.impact_time - ev_a.impact_time, ' s')} |",
        f"| Finish Time | {_fmt(ev_a.finish_time, ' s')} | {_fmt(ev_b.finish_time, ' s')} | {_fmt(ev_b.finish_time - ev_a.finish_time, ' s')} |",
        f"| Backswing Time | {_fmt(m_a.tempo.backswing_time, ' s')} | {_fmt(m_b.tempo.backswing_time, ' s')} | {_fmt(diff.get('backswing_time_s'), ' s')} |",
        f"| Downswing Time | {_fmt(m_a.tempo.downswing_time, ' s')} | {_fmt(m_b.tempo.downswing_time, ' s')} | {_fmt(diff.get('downswing_time_s'), ' s')} |",
        f"| Tempo Ratio | {_fmt(m_a.tempo.tempo_ratio, ':1')} | {_fmt(m_b.tempo.tempo_ratio, ':1')} | {_fmt(diff.get('tempo_ratio'))} |",
        "",
        "## Segment Rotations and X-Factor",
        "",
        "| Metric | Swing A (Ref) | Swing B (Cand) | Difference (B - A) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Pelvis Yaw at Top | {_fmt(m_a.segment_rotation.pelvis_yaw_top, '°')} | {_fmt(m_b.segment_rotation.pelvis_yaw_top, '°')} | {_fmt(diff.get('pelvis_yaw_top_deg'), '°')} |",
        f"| Pelvis Yaw at Impact | {_fmt(m_a.segment_rotation.pelvis_yaw_impact, '°')} | {_fmt(m_b.segment_rotation.pelvis_yaw_impact, '°')} | {_fmt(diff.get('pelvis_yaw_impact_deg'), '°')} |",
        f"| Shoulder-Girdle Yaw at Top | {_fmt(m_a.segment_rotation.shoulder_girdle_yaw_top, '°')} | {_fmt(m_b.segment_rotation.shoulder_girdle_yaw_top, '°')} | {_fmt(diff.get('shoulder_girdle_yaw_top_deg'), '°')} |",
        f"| Upper-Trunk Yaw at Top | {_fmt(m_a.segment_rotation.upper_trunk_yaw_top, '°')} | {_fmt(m_b.segment_rotation.upper_trunk_yaw_top, '°')} | {_fmt(diff.get('upper_trunk_yaw_top_deg'), '°')} |",
        f"| Shoulder-Girdle Yaw at Impact | {_fmt(m_a.segment_rotation.shoulder_girdle_yaw_impact, '°')} | {_fmt(m_b.segment_rotation.shoulder_girdle_yaw_impact, '°')} | {_fmt(diff.get('shoulder_girdle_yaw_impact_deg'), '°')} |",
        f"| Upper-Trunk Yaw at Impact | {_fmt(m_a.segment_rotation.upper_trunk_yaw_impact, '°')} | {_fmt(m_b.segment_rotation.upper_trunk_yaw_impact, '°')} | {_fmt(diff.get('upper_trunk_yaw_impact_deg'), '°')} |",
        f"| X-Factor at Address | {_fmt(m_a.segment_rotation.x_factor_address, '°')} | {_fmt(m_b.segment_rotation.x_factor_address, '°')} | {_fmt(diff.get('x_factor_address_deg'), '°')} |",
        f"| X-Factor at Top | {_fmt(m_a.segment_rotation.x_factor_top, '°')} | {_fmt(m_b.segment_rotation.x_factor_top, '°')} | {_fmt(diff.get('x_factor_top_deg'), '°')} |",
        f"| X-Factor at Impact | {_fmt(m_a.segment_rotation.x_factor_impact, '°')} | {_fmt(m_b.segment_rotation.x_factor_impact, '°')} | {_fmt(diff.get('x_factor_impact_deg'), '°')} |",
        f"| X-Factor Stretch | {_fmt(m_a.segment_rotation.x_factor_stretch, '°')} | {_fmt(m_b.segment_rotation.x_factor_stretch, '°')} | {_fmt(diff.get('x_factor_stretch_deg'), '°')} |",
        "",
        "## Kinematic Sequence",
        "",
        "| Segment | Swing A Peak Speed | Swing A Peak Time | Swing B Peak Speed | Swing B Peak Time |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| Pelvis | {_fmt(m_a.peak_speed('pelvis'), ' °/s')} | {_fmt(m_a.peak_time('pelvis'), ' s')} | {_fmt(m_b.peak_speed('pelvis'), ' °/s')} | {_fmt(m_b.peak_time('pelvis'), ' s')} |",
        f"| Thorax | {_fmt(m_a.peak_speed('thorax'), ' °/s')} | {_fmt(m_a.peak_time('thorax'), ' s')} | {_fmt(m_b.peak_speed('thorax'), ' °/s')} | {_fmt(m_b.peak_time('thorax'), ' s')} |",
        f"| Lead Arm | {_fmt(m_a.peak_speed('lead_arm'), ' °/s')} | {_fmt(m_a.peak_time('lead_arm'), ' s')} | {_fmt(m_b.peak_speed('lead_arm'), ' °/s')} | {_fmt(m_b.peak_time('lead_arm'), ' s')} |",
        f"| Club | {_fmt(m_a.peak_speed('club'), ' °/s')} | {_fmt(m_a.peak_time('club'), ' s')} | {_fmt(m_b.peak_speed('club'), ' °/s')} | {_fmt(m_b.peak_time('club'), ' s')} |",
        "",
        f"- **Swing A Sequence Order:** {' → '.join(m_a.kinematic_sequence.order)}",
        f"- **Swing B Sequence Order:** {' → '.join(m_b.kinematic_sequence.order)}",
        "",
        "## Arm and Wrist Mechanics",
        "",
        "| Metric | Swing A (Ref) | Swing B (Cand) | Difference (B - A) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Elbow Included Angle at Top | {_fmt(m_a.lead_arm.elbow_angle_top, '°')} | {_fmt(m_b.lead_arm.elbow_angle_top, '°')} | {_fmt(m_b.lead_arm.elbow_angle_top - m_a.lead_arm.elbow_angle_top, '°')} |",
        f"| Elbow Included Angle at Impact | {_fmt(m_a.lead_arm.elbow_angle_impact, '°')} | {_fmt(m_b.lead_arm.elbow_angle_impact, '°')} | {_fmt(m_b.lead_arm.elbow_angle_impact - m_a.lead_arm.elbow_angle_impact, '°')} |",
        f"| Minimum Included Elbow Angle | {_fmt(m_a.lead_arm.min_included_angle, '°')} | {_fmt(m_b.lead_arm.min_included_angle, '°')} | {_fmt(diff.get('elbow_min_included_angle_deg'), '°')} |",
        f"| Maximum Elbow Flexion | {_fmt(m_a.lead_arm.max_flexion_deg, '°')} | {_fmt(m_b.lead_arm.max_flexion_deg, '°')} | {_fmt(diff.get('elbow_max_flexion_deg'), '°')} |",
        f"| Wrist Hinge at Top | {_fmt(m_a.wrist.hinge_angle_top, '°')} | {_fmt(m_b.wrist.hinge_angle_top, '°')} | {_fmt(diff.get('wrist_hinge_top_deg'), '°')} |",
        f"| Wrist Hinge at Impact | {_fmt(m_a.wrist.hinge_angle_impact, '°')} | {_fmt(m_b.wrist.hinge_angle_impact, '°')} | {_fmt(diff.get('wrist_hinge_impact_deg'), '°')} |",
        "",
        "## Club Delivery and Hand Path",
        "",
        "| Metric | Swing A (Ref) | Swing B (Cand) | Difference (B - A) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Impact Club Head Speed | {_fmt(m_a.club.impact_club_head_speed_m_s, ' m/s')} | {_fmt(m_b.club.impact_club_head_speed_m_s, ' m/s')} | {_fmt(diff.get('impact_club_head_speed_m_s'), ' m/s')} |",
        f"| Peak Club Head Speed | {_fmt(m_a.club.peak_club_head_speed_m_s, ' m/s')} | {_fmt(m_b.club.peak_club_head_speed_m_s, ' m/s')} | {_fmt(diff.get('peak_club_head_speed_m_s'), ' m/s')} |",
        f"| Shaft Lean at Impact | {_fmt(m_a.club.impact_shaft_lean_deg, '°')} | {_fmt(m_b.club.impact_shaft_lean_deg, '°')} | {_fmt(diff.get('impact_shaft_lean_deg'), '°')} |",
        f"| Face Angle at Impact | {_fmt(m_a.club.impact_face_angle_deg, '°')} | {_fmt(m_b.club.impact_face_angle_deg, '°')} | {_fmt(diff.get('impact_face_angle_deg'), '°')} |",
        f"| Hand Path Length (Swing) | {_fmt(m_a.hand_path.path_length_swing_m, ' m')} | {_fmt(m_b.hand_path.path_length_swing_m, ' m')} | {_fmt(diff.get('hand_path_length_swing_m'), ' m')} |",
        f"| Hand Maximum Height at Top | {_fmt(m_a.hand_path.max_height_top_m, ' m')} | {_fmt(m_b.hand_path.max_height_top_m, ' m')} | {_fmt(diff.get('hand_max_height_top_m'), ' m')} |",
        "",
        "## Trajectory Alignment and Marker RMS",
        "",
        f"- **Mean Marker Trajectory RMS:** {_fmt(report.mean_marker_rms, ' m', decimals=4)}",
        f"- **Shared Markers Count:** {len(report.shared_marker_rms)}",
        "",
    ]

    if report.shared_marker_rms:
        lines.extend(
            [
                "| Marker Label | Trajectory RMS Error |",
                "| :--- | :--- |",
            ]
        )
        for lbl, rms in sorted(report.shared_marker_rms.items()):
            lines.append(f"| `{lbl}` | {_fmt(rms, ' m', decimals=4)} |")
        lines.append("")

    return "\n".join(lines)
