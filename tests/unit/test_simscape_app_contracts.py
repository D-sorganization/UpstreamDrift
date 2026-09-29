"""TDD tests for Simscape app false-success reports (issue #2490).

Four bugs:
1. main_golf_analysis_app.m announces success even though Tabs 1 and 2 are placeholders.
2. tab3_visualization.m checks BASEQ.mat exists but blindly loads ZTCFQ.mat and DELTAQ.mat.
3. 2D run_all.m describes a full simulation workflow but is a metadata-only stub.
4. 3D run_all.m claims to run the Golf Model pipeline but runs a hand-coded Euler projectile.
"""

from __future__ import annotations

from pathlib import Path

_BASE = Path("src/engines/Simscape_Multibody_Models")
_2D_APP = _BASE / "2D_Golf_Model/matlab/Integrated_Analysis_App"
_2D_RUN_ALL = _BASE / "2D_Golf_Model/matlab/run_all.m"
_3D_RUN_ALL = _BASE / "3D_Golf_Model/matlab/run_all.m"
_TAB3 = _2D_APP / "tab3_visualization.m"
_MAIN_APP = _2D_APP / "main_golf_analysis_app.m"


class TestMainAppNoFalseSuccessMessage:
    """main_golf_analysis_app.m must not announce success when placeholder tabs are present."""

    def _source(self) -> str:
        return _MAIN_APP.read_text(encoding="utf-8")

    def test_no_unconditional_success_message_for_placeholder_app(self) -> None:
        """App must not print 'initialized successfully' when key tabs are placeholders."""
        source = self._source()
        lines = source.splitlines()
        misleading = [
            line
            for line in lines
            if "initialized successfully" in line.lower()
            and not line.strip().startswith("%")
        ]
        assert not misleading, (
            "main_golf_analysis_app.m prints 'initialized successfully' even though "
            "Tabs 1 and 2 are placeholder stubs. This misleads users and automation. "
            "Replace with a warning that some tabs are not yet implemented.\n"
            "Offending lines:\n" + "\n".join(misleading)
        )


class TestTab3LoadsAllFilesGuarded:
    """tab3_visualization.m must check all required files exist before loading."""

    def _source(self) -> str:
        return _TAB3.read_text(encoding="utf-8")

    def test_ztcfq_existence_checked_before_load(self) -> None:
        """ZTCFQ.mat must be existence-checked before loading."""
        source = self._source()
        lines = source.splitlines()
        # After the fix the file must have an existence check for ztcfq_file
        has_ztcfq_check = any(
            "ztcfq_file" in line and "exist" in line.lower() for line in lines
        )
        assert has_ztcfq_check, (
            "tab3_visualization.m loads ZTCFQ.mat without checking if it exists. "
            "Add: if ~exist(ztcfq_file, 'file') before load(ztcfq_file)."
        )

    def test_deltaq_existence_checked_before_load(self) -> None:
        """DELTAQ.mat must be existence-checked before loading."""
        source = self._source()
        lines = source.splitlines()
        has_deltaq_check = any(
            "deltaq_file" in line and "exist" in line.lower() for line in lines
        )
        assert has_deltaq_check, (
            "tab3_visualization.m loads DELTAQ.mat without checking if it exists. "
            "Add: if ~exist(deltaq_file, 'file') before load(deltaq_file)."
        )


class TestRunAllScriptsNotMisleading:
    """run_all.m scripts must not claim to run the full model when they are stubs."""

    def test_2d_run_all_not_described_as_full_simulation(self) -> None:
        """2D run_all.m must not describe itself as running a full simulation."""
        source = _2D_RUN_ALL.read_text(encoding="utf-8")
        lines = source.splitlines()
        misleading = [
            line
            for line in lines
            if "placeholder for" in line.lower()
            and "simulat" in line.lower()
            and not line.strip().startswith("%")
        ]
        assert not misleading, (
            "2D run_all.m has inline comments like 'Placeholder for simulations' "
            "but then prints 'run_all completed' unconditionally, creating a false "
            "success signal. Remove or replace the misleading output.\n"
            "Offending lines:\n" + "\n".join(misleading)
        )

    def test_2d_run_all_warns_not_fully_implemented(self) -> None:
        """2D run_all.m must warn that full simulation is not yet implemented."""
        source = _2D_RUN_ALL.read_text(encoding="utf-8")
        has_warning = (
            "warning(" in source.lower()
            or "not implemented" in source.lower()
            or "placeholder" in source.lower()
            or "stub" in source.lower()
        )
        assert has_warning, (
            "2D run_all.m does not warn users that the full simulation is not "
            "yet implemented. Add a fprintf warning or error so callers know "
            "the script is a metadata stub."
        )

    def test_3d_run_all_not_described_as_golf_model_pipeline(self) -> None:
        """3D run_all.m must not describe a hand-coded Euler integrator as the 'Golf Model Pipeline'."""
        source = _3D_RUN_ALL.read_text(encoding="utf-8")
        lines = source.splitlines()
        # The description must not claim it orchestrates the full Simscape pipeline
        misleading_header = [
            line
            for line in lines[:15]  # Check only function header
            if "full golf model" in line.lower()
            or ("orchestrates" in line.lower() and "pipeline" in line.lower())
        ]
        assert not misleading_header, (
            "3D run_all.m header claims to orchestrate the 'full Golf Model pipeline' "
            "but actually runs a hand-coded Euler projectile toy. "
            "Update the description to reflect what the script actually does.\n"
            "Offending lines:\n" + "\n".join(misleading_header)
        )
