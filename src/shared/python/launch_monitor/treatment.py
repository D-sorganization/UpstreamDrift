"""Reproducible launch-monitor data-treatment pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterRule:
    """One structured, auditable subset rule."""

    column: str
    operator: str
    value: object

    def __post_init__(self) -> None:
        allowed = {"eq", "ne", "lt", "le", "gt", "ge", "contains", "in"}
        if not self.column.strip():
            raise ValueError("filter column must be non-empty")
        if self.operator not in allowed:
            raise ValueError(f"Unsupported filter operator: {self.operator}")


@dataclass(frozen=True)
class TreatmentConfig:
    """Configuration for non-destructive quality flagging and filtering."""

    required_metrics: tuple[str, ...] = ()
    duplicate_columns: tuple[str, ...] = ("shot_id",)
    outlier_metrics: tuple[str, ...] = ()
    robust_z_threshold: float = 4.5
    exclude_flagged: bool = False
    derive_metrics: bool = True
    filters: tuple[FilterRule, ...] = ()

    def __post_init__(self) -> None:
        if self.robust_z_threshold <= 0 or not np.isfinite(self.robust_z_threshold):
            raise ValueError("robust_z_threshold must be finite and positive")


@dataclass(frozen=True)
class TreatmentResult:
    """Treated analysis view, row-level flags, and audit records."""

    data: pd.DataFrame
    flags: pd.DataFrame
    audit_log: tuple[dict[str, object], ...]


def _robust_outlier_mask(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    absolute = (numeric - median).abs()
    mad = absolute.median()
    if not np.isfinite(mad) or mad == 0:
        std = numeric.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(False, index=values.index)
        return absolute / std > threshold
    modified_z = 0.67448975 * absolute / mad
    return modified_z > threshold


def _filter_mask(frame: pd.DataFrame, rule: FilterRule) -> pd.Series:
    if rule.column not in frame:
        raise ValueError(f"Filter column not present: {rule.column}")
    values = frame[rule.column]
    if rule.operator in {"lt", "le", "gt", "ge"}:
        numeric = pd.to_numeric(values, errors="coerce")
        try:
            target = float(str(rule.value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Filter value for {rule.operator} must be numeric: {rule.value}"
            ) from exc
        return {
            "lt": numeric < target,
            "le": numeric <= target,
            "gt": numeric > target,
            "ge": numeric >= target,
        }[rule.operator]
    if rule.operator == "contains":
        return values.astype(str).str.contains(str(rule.value), case=False, na=False)
    if rule.operator == "in":
        accepted = [item.strip() for item in str(rule.value).split(",")]
        return values.astype(str).isin(accepted)
    if rule.operator == "eq":
        return values.astype(str) == str(rule.value)
    return values.astype(str) != str(rule.value)


def _derive_available_metrics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    output = frame.copy()
    actions: list[dict[str, object]] = []
    recipes = {
        "smash_factor": ("ball_speed", "club_speed", lambda a, b: a / b),
        "face_to_path": ("face_angle", "club_path", lambda a, b: a - b),
        "roll_distance": ("total_distance", "carry_distance", lambda a, b: a - b),
    }
    for target, (left, right, operation) in recipes.items():
        if left not in output or right not in output:
            continue
        derived = operation(
            pd.to_numeric(output[left], errors="coerce"),
            pd.to_numeric(output[right], errors="coerce"),
        ).replace([np.inf, -np.inf], np.nan)
        if target in output:
            mask = output[target].isna() & derived.notna()
            output.loc[mask, target] = derived.loc[mask]
        else:
            mask = derived.notna()
            output[target] = derived
        status_column = f"status::{target}"
        if status_column not in output:
            output[status_column] = "unknown"
        output.loc[mask, status_column] = "derived"
        count = int(mask.sum())
        if count:
            actions.append(
                {
                    "action": "derive_metric",
                    "metric": target,
                    "inputs": [left, right],
                    "row_count": count,
                }
            )
    return output, actions


def apply_treatment(frame: pd.DataFrame, config: TreatmentConfig) -> TreatmentResult:
    """Flag data-quality conditions and optionally exclude affected rows."""
    if frame.empty:
        raise ValueError("frame must contain at least one shot")
    working = frame.copy(deep=True)
    records: list[dict[str, object]] = []
    flagged_indices: set[object] = set()

    if config.derive_metrics:
        working, derivation_actions = _derive_available_metrics(working)
    else:
        derivation_actions = []

    filter_actions: list[dict[str, object]] = []
    for rule in config.filters:
        before = len(working)
        working = working.loc[_filter_mask(working, rule)].copy()
        filter_actions.append(
            {
                "action": "filter",
                "column": rule.column,
                "operator": rule.operator,
                "value": rule.value,
                "rows_before": before,
                "rows_after": len(working),
            }
        )

    missing_columns = set(config.required_metrics) - set(working.columns)
    if missing_columns:
        raise ValueError(f"Required metrics not present: {sorted(missing_columns)}")
    if config.required_metrics:
        mask = working[list(config.required_metrics)].isna().any(axis=1)
        for index in working.index[mask]:
            records.append(
                {"row_index": index, "flag_type": "missing_required", "metric": None}
            )
            flagged_indices.add(index)

    duplicates = [name for name in config.duplicate_columns if name in working]
    if duplicates:
        mask = working.duplicated(subset=duplicates, keep="first")
        for index in working.index[mask]:
            records.append(
                {
                    "row_index": index,
                    "flag_type": "duplicate",
                    "metric": ",".join(duplicates),
                }
            )
            flagged_indices.add(index)

    for metric in config.outlier_metrics:
        if metric not in working:
            raise ValueError(f"Outlier metric not present: {metric}")
        mask = _robust_outlier_mask(working[metric], config.robust_z_threshold)
        for index in working.index[mask]:
            records.append(
                {"row_index": index, "flag_type": "robust_outlier", "metric": metric}
            )
            flagged_indices.add(index)

    flags = pd.DataFrame(records, columns=["row_index", "flag_type", "metric"])
    if config.exclude_flagged and flagged_indices:
        working = working.drop(index=list(flagged_indices)).copy()
    audit: list[dict[str, object]] = [
        *derivation_actions,
        *filter_actions,
        *[
            {
                "action": "flag",
                "flag_type": record["flag_type"],
                "row_index": record["row_index"],
                "metric": record["metric"],
            }
            for record in records
        ],
    ]
    if config.exclude_flagged:
        audit.append({"action": "exclude_flagged", "row_count": len(flagged_indices)})
    return TreatmentResult(working.reset_index(drop=True), flags, tuple(audit))
