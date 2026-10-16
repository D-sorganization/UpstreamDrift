"""Relationship, treatment, multivariate and model analysis handlers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.async_action import WorkerContext
from src.tools.launch_monitor_analytics._gui_params import (
    _ModelParams,
    _RelationshipParams,
)
from src.tools.launch_monitor_analytics.widgets import (
    selected_text as _selected_text,
)
from src.tools.launch_monitor_model import (
    CorrelationResult,
    PCAResult,
    PredictiveModelResult,
    TreatmentConfig,
    TreatmentResult,
    VIFResult,
    apply_treatment,
    compute_correlations,
    compute_pca,
    compute_vif,
    fit_predictive_model,
)

logger = get_logger(__name__)


class _CoreAnalysisMixin:
    """Core statistical analysis handlers for ``MainWidget`` (#9470)."""

    if TYPE_CHECKING:
        # Host attributes/methods live on ``MainWidget`` (composed mixins).
        def __getattr__(self, name: str) -> Any: ...

    def run_relationship_analysis(self) -> CorrelationResult:
        """Run the relationship analysis synchronously and return its result.

        The synchronous path (``present(compute())``); the button runs
        :meth:`_run_relationship_async` on the shared action bar.
        """
        result = self._compute_relationship(self._read_relationship_params())
        self._present_relationship(result)
        return result

    def _read_relationship_params(self) -> _RelationshipParams:
        """Read the relationship controls on the GUI thread.

        The worker must not touch widgets, so every value is captured here
        before the work callable is built (#9470).
        """
        metrics = _selected_text(self.relationship_metrics)
        return _RelationshipParams(
            metrics=metrics,
            controls=tuple(
                item
                for item in _selected_text(self.relationship_controls)
                if item not in metrics
            ),
            method=self.relationship_method.currentText(),
            edge_threshold=float(self.edge_threshold_spin.value()),
        )

    def _compute_relationship(
        self,
        params: _RelationshipParams,
        ctx: WorkerContext | None = None,
    ) -> CorrelationResult:
        """Compute the correlation matrix as plain data (#9470).

        Args:
            params: Widget-derived inputs read on the GUI thread.
            ctx: Cancellation/progress seam; ``None`` keeps the synchronous
                path identical to the pre-migration behaviour.

        Returns:
            The correlation result, safe to hand to the GUI thread.

        Raises:
            ValueError: If fewer than two relationship metrics are selected.
        """
        if len(params.metrics) < 2:
            raise ValueError("Select at least two relationship metrics")
        if ctx is not None:
            ctx.report(
                None,
                "correlating metrics; cancel takes effect when the step ends",
            )
        return compute_correlations(
            self.analysis_frame,
            metrics=params.metrics,
            method=params.method,
            controls=params.controls,
            edge_threshold=params.edge_threshold,
        )

    def _present_relationship(self, result: CorrelationResult) -> None:
        """Render the correlation matrix. GUI-thread only (#9470)."""
        metrics = list(result.coefficients.index)
        self.relationship_table.set_frame(
            result.coefficients.reset_index(names="metric")
        )
        axes = self.relationship_plot.reset_axes()
        image = axes.imshow(
            result.coefficients.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm"
        )
        axes.set_xticks(range(len(metrics)), metrics, rotation=45, ha="right")
        axes.set_yticks(range(len(metrics)), metrics)
        axes.set_title(self._title_with_project(f"{result.method.title()} Correlation"))
        self.relationship_plot.figure.colorbar(image, ax=axes, fraction=0.046)
        self.relationship_plot.draw_idle()
        self.status_label.setText(
            f"Mapped {len(result.edges)} screened dependency edges across "
            f"{len(metrics)} metrics. Interpret as association, not causation."
        )

    def _run_relationship_ui(self) -> None:
        """Synchronous relationship analysis (kept pre-migration path)."""
        try:
            self.run_relationship_analysis()
        except ValueError as exc:
            self._show_error("Relationship Analysis Failed", exc)

    def _run_relationship_async(self) -> None:
        """Run the relationship analysis off the GUI thread (#9470)."""
        params = self._read_relationship_params()
        self._start_action(
            "Relationship analysis",
            lambda ctx: self._compute_relationship(params, ctx),
            self._present_relationship,
            "Relationship Analysis Failed",
        )

    # ---- data treatment ---------------------------------------------------

    def _read_treatment_config(self) -> TreatmentConfig:
        """Read the treatment controls on the GUI thread (#9470)."""
        required = tuple(
            item.strip()
            for item in self.required_metrics_edit.text().split(",")
            if item.strip()
        )
        outliers = tuple(
            item.strip()
            for item in self.outlier_metrics_edit.text().split(",")
            if item.strip()
        )
        return TreatmentConfig(
            required_metrics=required,
            outlier_metrics=outliers,
            robust_z_threshold=float(self.outlier_threshold_spin.value()),
            exclude_flagged=self.exclude_flagged_check.isChecked(),
            filters=self._filter_rules(),
        )

    def _compute_treatment(
        self,
        config: TreatmentConfig,
        ctx: WorkerContext | None = None,
    ) -> TreatmentResult:
        """Apply the treatment to the combined shots as plain data (#9470).

        ``apply_treatment`` is one opaque call with no loop boundary, so the
        worker keeps the window painting but Cancel only takes effect when
        the current step ends; the progress line says so.
        """
        if ctx is not None:
            ctx.report(
                None,
                "screening shots; cancel takes effect when the step ends",
            )
        return apply_treatment(self.project.combined_shots(), config)

    def _present_treatment(self, result: TreatmentResult) -> None:
        """Apply treated data to the workspace. GUI-thread only (#9470)."""
        self.analysis_frame = result.data
        self.project.record_actions(result.audit_log)
        self.flags_table.set_frame(result.flags)
        self.audit_text.setPlainText(
            json.dumps(self.project.audit_log, indent=2, default=str)
        )
        self._dirty = True
        self._refresh_all()
        self.status_label.setText(
            f"Treatment complete: {len(result.flags)} flags; "
            f"{len(result.data)} shots in the analysis view. Raw sessions are unchanged."
        )

    def _run_treatment_ui(self) -> None:
        """Synchronous data treatment (kept pre-migration path)."""
        try:
            config = self._read_treatment_config()
            result = self._compute_treatment(config)
        except ValueError as exc:
            self._show_error("Data Treatment Failed", exc)
            return
        self._present_treatment(result)

    def _run_treatment_async(self) -> None:
        """Run the data treatment off the GUI thread (#9470)."""
        try:
            config = self._read_treatment_config()
        except ValueError as exc:
            self._show_error("Data Treatment Failed", exc)
            return
        self._start_action(
            "Data treatment",
            lambda ctx: self._compute_treatment(config, ctx),
            self._present_treatment,
            "Data Treatment Failed",
        )

    # ---- multivariate (PCA/VIF) -------------------------------------------

    def _compute_multivariate(
        self,
        metrics: tuple[str, ...],
        ctx: WorkerContext | None = None,
    ) -> tuple[PCAResult, VIFResult]:
        """Compute PCA and VIF as plain data (#9470).

        The PCA and VIF stages are separate checkpoints: cancelling between
        them skips the remaining stage.
        """
        if ctx is not None:
            ctx.report(None, "running principal components")
        pca = compute_pca(self.analysis_frame, metrics=metrics)
        if ctx is not None:
            ctx.raise_if_cancelled()
            ctx.report(0.5, "computing variance inflation factors")
        vif = compute_vif(self.analysis_frame, metrics=metrics)
        return pca, vif

    def _present_multivariate(self, pca: PCAResult, vif: VIFResult) -> None:
        """Render PCA loadings and variance ratios. GUI-thread only (#9470)."""
        table = pca.loadings.copy()
        table.insert(0, "VIF", vif.values)
        self.relationship_table.set_frame(table.reset_index(names="metric"))
        axes = self.relationship_plot.reset_axes()
        axes.bar(
            pca.explained_variance_ratio.index,
            pca.explained_variance_ratio.to_numpy(dtype=float),
        )
        axes.set_ylim(0, 1)
        axes.set_ylabel("Explained Variance Ratio")
        axes.set_title(self._title_with_project("Principal-Component Variance"))
        axes.grid(True, axis="y", alpha=0.25)
        self.relationship_plot.draw_idle()
        warning = ", ".join(vif.warning_metrics) or "none"
        self.status_label.setText(
            f"PCA/VIF complete for {pca.sample_count} complete shots. "
            f"VIF >= 5: {warning}."
        )

    def _run_multivariate_ui(self) -> None:
        """Synchronous PCA/VIF analysis (kept pre-migration path)."""
        try:
            pca, vif = self._compute_multivariate(
                _selected_text(self.relationship_metrics)
            )
        except ValueError as exc:
            self._show_error("Multivariate Analysis Failed", exc)
            return
        self._present_multivariate(pca, vif)

    def _run_multivariate_async(self) -> None:
        """Run the PCA/VIF analysis off the GUI thread (#9470)."""
        metrics = _selected_text(self.relationship_metrics)
        self._start_action(
            "Multivariate analysis",
            lambda ctx: self._compute_multivariate(metrics, ctx),
            lambda result: self._present_multivariate(*result),
            "Multivariate Analysis Failed",
        )

    # ---- predictive model -------------------------------------------------

    def _read_model_params(self) -> _ModelParams:
        """Read the model controls on the GUI thread (#9470)."""
        group = self.model_group.currentText()
        return _ModelParams(
            target=self.model_target.currentText(),
            features=_selected_text(self.model_features),
            model=self.model_type.currentText(),
            random_seed=int(self.model_seed.value()),
            group_column=None if group == "(random split)" else group,
        )

    def _compute_model(
        self,
        params: _ModelParams,
        ctx: WorkerContext | None = None,
    ) -> PredictiveModelResult:
        """Fit the predictive model as plain data (#9470).

        One opaque scikit-learn call, so there is no loop boundary to check;
        Cancel takes effect when the fit returns (noted in the progress line).
        """
        if ctx is not None:
            ctx.report(
                None,
                "fitting held-out model; cancel takes effect when the fit ends",
            )
        return fit_predictive_model(
            self.analysis_frame,
            target=params.target,
            features=params.features,
            model=params.model,
            random_seed=params.random_seed,
            group_column=params.group_column,
        )

    def _present_model(self, result: PredictiveModelResult) -> None:
        """Render the model report and prediction scatter. GUI thread only."""
        self.model_report.setPlainText(
            json.dumps(
                {
                    "model": result.model,
                    "target": result.target,
                    "features": result.features,
                    "metrics": result.metrics,
                    "coefficients": result.coefficients,
                    "random_seed": result.random_seed,
                    "train_count": result.train_count,
                    "test_count": result.test_count,
                },
                indent=2,
            )
        )
        axes = self.model_plot.reset_axes()
        axes.scatter(
            result.predictions["actual"], result.predictions["predicted"], alpha=0.75
        )
        bounds = [
            result.predictions[["actual", "predicted"]].min().min(),
            result.predictions[["actual", "predicted"]].max().max(),
        ]
        axes.plot(bounds, bounds, linestyle="--", color="gray")
        axes.set_xlabel("Actual")
        axes.set_ylabel("Predicted")
        axes.set_title(
            self._title_with_project(
                f"Held-Out {result.model.replace('_', ' ').title()} Predictions"
            )
        )
        axes.grid(True, alpha=0.25)
        self.model_plot.draw_idle()
        self.status_label.setText(
            f"Model complete: R2={result.metrics['r2']:.3f}, "
            f"RMSE={result.metrics['rmse']:.3g}."
        )

    def _run_model_ui(self) -> None:
        """Synchronous predictive-model fit (kept pre-migration path)."""
        try:
            result = self._compute_model(self._read_model_params())
        except (ValueError, ImportError) as exc:
            self._show_error("Predictive Model Failed", exc)
            return
        self._present_model(result)

    def _run_model_async(self) -> None:
        """Run the predictive-model fit off the GUI thread (#9470)."""
        params = self._read_model_params()
        self._start_action(
            "Predictive model",
            lambda ctx: self._compute_model(params, ctx),
            self._present_model,
            "Predictive Model Failed",
        )
