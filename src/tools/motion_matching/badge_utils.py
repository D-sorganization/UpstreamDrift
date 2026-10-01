"""Presentation helpers and badge formatting for motion matching GUI."""

from __future__ import annotations

from typing import Any


def resolve_acceptance_badge_style(verdict: str) -> tuple[str, str]:
    """Return text and stylesheet for acceptance badge."""
    if verdict in ("PASSED", "QUALIFIED"):
        return verdict, "font-weight: bold; color: green;"
    if verdict == "REJECTED":
        return verdict, "font-weight: bold; color: red;"
    return verdict, "font-weight: bold; color: gray;"


def resolve_neural_badge(
    neural_info: dict[str, Any] | None,
    active_model: str = "",
) -> dict[str, Any]:
    """Resolve fail-closed neural verification badge state and metadata (R06, #11146)."""
    if not isinstance(neural_info, dict):
        return {
            "badge_text": "-",
            "badge_style": "font-weight: bold; color: gray;",
            "confidence_text": "-",
            "timing_text": "-",
            "status": "NONE",
            "is_preview": False,
        }

    raw_status = neural_info.get("status")
    is_preview = bool(
        neural_info.get("is_preview") or neural_info.get("is_preview_only", False)
    )
    if not raw_status:
        status = "UNVERIFIED"
    else:
        status = str(raw_status).strip()
        info_model = str(neural_info.get("model_id", "")).strip()
        if info_model and active_model and info_model != active_model:
            status = "UNVERIFIED"

    status_upper = status.upper()
    if is_preview:
        badge_text = f"PREVIEW ({status})"
        badge_style = (
            "font-weight: bold; color: darkorange; background-color: cornsilk; "
            "border-radius: 4px; padding: 2px 6px;"
        )
    elif status_upper in ("VERIFIED", "NEURAL_ACCEPTED"):
        badge_text = "VERIFIED"
        badge_style = (
            "font-weight: bold; color: forestgreen; background-color: honeydew; "
            "border-radius: 4px; padding: 2px 6px;"
        )
    elif status_upper == "CLASSICAL_FALLBACK":
        badge_text = "CLASSICAL FALLBACK"
        badge_style = (
            "font-weight: bold; color: royalblue; background-color: aliceblue; "
            "border-radius: 4px; padding: 2px 6px;"
        )
    elif status_upper == "REJECTED":
        badge_text = "REJECTED"
        badge_style = (
            "font-weight: bold; color: crimson; background-color: mistyrose; "
            "border-radius: 4px; padding: 2px 6px;"
        )
    else:
        badge_text = status
        badge_style = "font-weight: bold; color: gray;"

    conf = neural_info.get("confidence")
    confidence_text = f"{conf:.3f} (domain support)" if conf is not None else "-"

    t_neural_s = float(neural_info.get("t_neural_s", 0.0))
    t_polish_s = float(neural_info.get("t_polish_s", 0.0))
    t_total_s = float(neural_info.get("t_total_s", 0.0))
    if t_total_s > 0.0:
        timing_text = (
            f"Neural: {t_neural_s * 1000:.1f}ms | "
            f"Polish: {t_polish_s * 1000:.1f}ms | "
            f"Total: {t_total_s * 1000:.1f}ms"
        )
    else:
        timing_text = "-"

    return {
        "badge_text": badge_text,
        "badge_style": badge_style,
        "confidence_text": confidence_text,
        "timing_text": timing_text,
        "status": status,
        "is_preview": is_preview,
    }
