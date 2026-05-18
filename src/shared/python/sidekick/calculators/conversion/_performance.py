from __future__ import annotations


class PerformanceMixin:
    def gasifier_performance(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        metric_type: str = "efficiency",
    ) -> float:
        metric_type = metric_type.lower()
        from_key = from_unit.lower()
        to_key = to_unit.lower()

        if metric_type in {"efficiency", "carbon_conversion"}:
            if from_key == to_key:
                return value
            if from_key == "%" and to_key == "fraction":
                return value / 100.0
            if from_key == "fraction" and to_key == "%":
                return value * 100.0
            msg = f"Unknown conversion for {metric_type}"
            raise ValueError(msg)

        if metric_type == "specific_production":
            if from_key == to_key:
                return value
            if from_key in {"nm³/kg", "nm3/kg"} and to_key == "scf/lb":
                return value / 0.0624
            if from_key == "scf/lb" and to_key in {"nm³/kg", "nm3/kg"}:
                return value * 0.0624
            msg = "Unknown specific production conversion"
            raise ValueError(msg)

        msg = f"Unknown metric type: {metric_type}"
        raise ValueError(msg)
