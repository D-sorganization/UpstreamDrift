"""UpstreamDrift CI/CD verification suite for the v4.0 3D Putting Dynamics Package.

Verifies:
1. Strict AST Architecture Budgets across all package modules:
   - <= 500 LOC per file
   - <= 100 LOC per function
   - <= 8 parameters per function
   - Zero references to banned 'MuJoCo' terminology
2. Issue D-11803 Zero-Allocation @precondition Signature Caching
3. Non-planar undulating green topography, analytical gradient vs finite difference, and micro-roughness
4. 4-point dual-hand grasp matrix, Moore-Penrose task drive, Null-Space Counter-Torque preload, ZTCF/ZVCF
5. Player-specific Bellman Strokes Gained curve (0-45 ft), capture speed strategy (low vs high variance),
   6-channel variance decomposition, and JSON export/load
6. Virtual USGA Stimpmeter ramp rollout and player-specific optimal green reading solver
7. Model Context Protocol (MCP) & JSON-RPC 2.0 server tool discovery and dispatch
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Sequence
import unittest
from unittest.mock import patch

import numpy as np

THIS_DIR = Path(__file__).resolve().parent


class _DirectStatFinder(importlib.abc.MetaPathFinder):
    """9p-safe module finder using direct stat() rather than cached readdir()."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        base = fullname.rpartition(".")[2]
        candidate = THIS_DIR / f"{base}.py"
        if candidate.exists():
            return importlib.util.spec_from_file_location(fullname, candidate)
        return None


sys.meta_path.insert(0, _DirectStatFinder())

from green_surface_undulation import (  # noqa: E402
    GreenUndulationProfile,
    evaluate_green_elevation_m,
    evaluate_green_gradient,
    generate_topographic_grid,
    sample_surface_roughness_acceleration,
)
from grip_nullspace_stabilization import (  # noqa: E402
    build_grasp_matrix_a_couple,
    decompose_ztcf_zvcf_accelerations,
    solve_grip_nullspace_stabilization,
)
from putting_mcp_server import PuttingMCPServer  # noqa: E402
from stimpmeter_reader import roll_stimpmeter_ball, solve_player_optimal_green_read  # noqa: E402
from strokes_gained_analytics import (  # noqa: E402
    PlayerSkillProfile,
    compute_player_expected_strokes_curve,
    decompose_putting_variance_channels,
    export_simulation_state_json,
    load_simulation_state_json,
    optimize_capture_speed_strategy,
)
from turf_properties import compute_rolling_friction  # noqa: E402


class TestUpstreamDriftPuttingCISuite(unittest.TestCase):
    """Comprehensive CI/CD test suite for UpstreamDrift v4.0 putting modules."""

    def test_01_ast_architecture_budgets_and_terminology(self) -> None:
        """All v4.0 modular files must obey <=500 LOC/file, <=100 LOC/func, <=8 params, and zero 'mujoco'."""
        module_names = (
            "turf_properties.py",
            "grip_nullspace_stabilization.py",
            "green_surface_undulation.py",
            "strokes_gained_analytics.py",
            "stimpmeter_reader.py",
            "putting_mcp_server.py",
            "test_putting_ci_suite.py",
        )
        for name in module_names:
            fpath = THIS_DIR / name
            self.assertTrue(fpath.exists(), f"Missing module {name}")
            text = fpath.read_text(encoding="utf-8")
            lines = text.splitlines()
            self.assertLessEqual(len(lines), 500, f"{name} exceeds 500 LOC ({len(lines)})")
            if name != "test_putting_ci_suite.py":
                self.assertNotIn("mujoco", text.lower(), f"Banned term found in {name}")
            tree = ast.parse(text, filename=str(fpath))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_loc = (node.end_lineno or node.lineno) - node.lineno + 1
                    self.assertLessEqual(
                        func_loc, 100, f"{name}:{node.name} exceeds 100 LOC ({func_loc})"
                    )
                    n_params = len(node.args.args) + len(node.args.kwonlyargs)
                    self.assertLessEqual(
                        n_params, 8, f"{name}:{node.name} exceeds 8 params ({n_params})"
                    )

    def test_02_zero_allocation_precondition_caching_issue_d11803(self) -> None:
        """@precondition must never call inspect.signature() at runtime."""
        with patch("inspect.signature", side_effect=AssertionError("inspect.signature called!")):
            mu = compute_rolling_friction(11.5)
            self.assertAlmostEqual(mu, 0.560 / 11.5, places=9)
            z = evaluate_green_elevation_m(1.5, -0.1)
            self.assertTrue(math.isfinite(z))
        with self.assertRaises(ValueError):
            compute_rolling_friction(-3.0)

    def test_03_undulating_green_analytical_gradient_and_roughness(self) -> None:
        """Analytical gradient grad_z(x, y) must match central finite differences to < 1e-7."""
        prof = GreenUndulationProfile(slope_x_pct=0.8, slope_y_pct=2.4, tier_height_m=0.04)
        eps = 1e-6
        for x_m, y_m in [(0.6, -0.2), (1.85, 0.15), (2.7, -0.1)]:
            gx, gy = evaluate_green_gradient(x_m, y_m, prof)
            fd_x = (
                evaluate_green_elevation_m(x_m + eps, y_m, prof)
                - evaluate_green_elevation_m(x_m - eps, y_m, prof)
            ) / (2.0 * eps)
            fd_y = (
                evaluate_green_elevation_m(x_m, y_m + eps, prof)
                - evaluate_green_elevation_m(x_m, y_m - eps, prof)
            ) / (2.0 * eps)
            self.assertAlmostEqual(gx, fd_x, places=7)
            self.assertAlmostEqual(gy, fd_y, places=7)
        grid = generate_topographic_grid((0.0, 4.0), (-1.5, 1.5), nx=15, ny=11, profile=prof)
        self.assertEqual(grid["z_m"].shape, (11, 15))
        ax, ay = sample_surface_roughness_acceleration(1.2, 0.1, 0.8, roughness_rms_mm=0.5, seed=7)
        self.assertGreater(math.hypot(ax, ay), 0.0)

    def test_04_nullspace_stabilization_and_ztcf_zvcf(self) -> None:
        """Null-space preload must produce zero net task torque and ZTCF+ZVCF must superpose."""
        a_mat = build_grasp_matrix_a_couple()
        sol = solve_grip_nullspace_stabilization(
            a_couple=a_mat,
            tau_task=np.array([0.85, -0.15, 0.08]),
            tau_preload=np.array([0.60, -0.60, 0.60, -0.60]),
        )
        np.testing.assert_allclose(a_mat @ sol.u_null, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(a_mat @ sol.u_total, np.array([0.85, -0.15, 0.08]), atol=1e-12)
        m_mat = np.diag([0.052, 0.048, 0.0045])
        c_mat = np.diag([0.02, 0.03, 0.01])
        g_vec = np.array([0.0, 0.45, 0.0])
        q_dot = np.array([1.2, -0.1, 0.05])
        tau_vec = np.array([0.1, -0.2, 0.8])
        decomp = decompose_ztcf_zvcf_accelerations(m_mat, c_mat, g_vec, q_dot, tau_vec)
        self.assertLess(decomp.superposition_residual_norm, 1e-12)

    def test_05_bellman_strokes_gained_and_capture_speed_strategy(self) -> None:
        """High-variance putter must optimize at slower capture speed than stabilized putter."""
        stabilized = PlayerSkillProfile(
            name="Stabilized",
            tau_preload_nm=0.60,
            base_face_sd_deg=0.42,
            base_speed_cv_pct=2.8,
            green_read_sd_pct=0.20,
        )
        unstabilized_poor = PlayerSkillProfile(
            name="Poor Unstabilized",
            tau_preload_nm=0.0,
            base_face_sd_deg=1.35,
            base_speed_cv_pct=6.8,
            green_read_sd_pct=0.65,
        )
        curve_stab = compute_player_expected_strokes_curve(skill=stabilized, n_trials=120)
        curve_poor = compute_player_expected_strokes_curve(skill=unstabilized_poor, n_trials=120)
        self.assertLess(
            curve_stab["expected_strokes_player"][4],
            curve_poor["expected_strokes_player"][4],
        )
        opt_stab = optimize_capture_speed_strategy(12.0, 2.2, stabilized, n_trials=250)
        opt_poor = optimize_capture_speed_strategy(12.0, 2.2, unstabilized_poor, n_trials=250)
        self.assertLessEqual(opt_poor["optimal_entry_speed_m_s"], opt_stab["optimal_entry_speed_m_s"])
        self.assertLess(opt_stab["optimal_expected_strokes"], opt_poor["optimal_expected_strokes"])

    def test_06_variance_decomposition_stimpmeter_and_mcp_server(self) -> None:
        """6-channel variance, Stimpmeter green read, JSON persistence, and MCP server must succeed."""
        var_res = decompose_putting_variance_channels(distance_ft=15.0, n_trials=120)
        self.assertEqual(len(var_res["strokes_gained_per_18_holes_by_channel"]), 6)
        roll = roll_stimpmeter_ball(launch_angle_deg=0.0, stimp_rating=11.0)
        self.assertGreater(roll.stopping_distance_ft, 7.0)
        read_res = solve_player_optimal_green_read(distance_ft=12.0, stimp_rating=11.0)
        self.assertGreater(read_res["expected_make_pct"], 20.0)
        mcp = PuttingMCPServer()
        tools = mcp.list_tools()
        self.assertEqual(len(tools), 5)
        rpc_resp = mcp.handle_jsonrpc_request({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "run_parallel_putt_batch",
                "arguments": {"distance_ft": 10.0, "n_trials": 100},
            },
        })
        self.assertIn("result", rpc_resp)
        self.assertEqual(rpc_resp["result"]["n_trials"], 100)
        exported = export_simulation_state_json(rpc_resp["result"])
        loaded = load_simulation_state_json(exported)
        self.assertEqual(loaded["n_trials"], 100)


if __name__ == "__main__":
    unittest.main()
