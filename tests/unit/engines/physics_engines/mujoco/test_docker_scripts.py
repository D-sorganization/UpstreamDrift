"""Tests for MuJoCo engine docker-related scripts."""

from unittest.mock import MagicMock, patch

import pytest


class TestAddDefusedXml:
    def test_create_minimal_dockerfile(self):
        from src.engines.physics_engines.mujoco.add_defusedxml_to_robotics_env import (
            create_minimal_dockerfile,
        )

        content = create_minimal_dockerfile()
        assert "FROM upstream-drift:engine" in content
        assert "defusedxml>=0.7.1" in content

    @patch(
        "src.engines.physics_engines.mujoco.add_defusedxml_to_robotics_env.subprocess.run"
    )
    def test_update_upstream_drift_success(self, mock_run):
        from src.engines.physics_engines.mujoco.add_defusedxml_to_robotics_env import (
            update_upstream_drift,
        )

        assert update_upstream_drift() is True
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][:4] == [
            "docker",
            "build",
            "-t",
            "upstream-drift:engine",
        ]

    @patch(
        "src.engines.physics_engines.mujoco.add_defusedxml_to_robotics_env.subprocess.run"
    )
    def test_test_updated_environment_success(self, mock_run):
        from src.engines.physics_engines.mujoco.add_defusedxml_to_robotics_env import (
            test_updated_environment,
        )

        mock_run.return_value = MagicMock(
            stdout="✅ defusedxml available\n mujoco 3.0\n"
        )
        assert test_updated_environment() is True
        assert mock_run.call_count == 3

    @patch(
        "src.engines.physics_engines.mujoco.add_defusedxml_to_robotics_env.test_updated_environment"
    )
    @patch(
        "src.engines.physics_engines.mujoco.add_defusedxml_to_robotics_env.update_upstream_drift"
    )
    def test_main(self, mock_update, mock_test):
        from src.engines.physics_engines.mujoco.add_defusedxml_to_robotics_env import (
            main as defusedxml_main,
        )

        mock_update.return_value = True
        mock_test.return_value = True
        assert defusedxml_main() == 0

        mock_update.return_value = False
        assert defusedxml_main() == 1


class TestAddQtDependencies:
    def test_create_qt_dockerfile(self):
        from src.engines.physics_engines.mujoco.add_qt_dependencies import (
            create_qt_dockerfile,
        )

        content = create_qt_dockerfile()
        assert "FROM upstream-drift:engine" in content
        assert "libqt6gui6" in content

    @patch("src.engines.physics_engines.mujoco.add_qt_dependencies.subprocess.run")
    def test_update_upstream_drift_qt_success(self, mock_run):
        from src.engines.physics_engines.mujoco.add_qt_dependencies import (
            update_upstream_drift_qt,
        )

        assert update_upstream_drift_qt() is True
        mock_run.assert_called_once()

    @patch("src.engines.physics_engines.mujoco.add_qt_dependencies.subprocess.run")
    def test_test_qt_environment_success(self, mock_run):
        from src.engines.physics_engines.mujoco.add_qt_dependencies import (
            test_qt_environment,
        )

        mock_run.return_value = MagicMock(stdout="✅ PyQt6 imports successfully\n")
        assert test_qt_environment() is True
        assert mock_run.call_count == 2

    @patch("src.engines.physics_engines.mujoco.add_qt_dependencies.test_qt_environment")
    @patch(
        "src.engines.physics_engines.mujoco.add_qt_dependencies.update_upstream_drift_qt"
    )
    def test_main(self, mock_update, mock_test):
        from src.engines.physics_engines.mujoco.add_qt_dependencies import (
            main as qt_main,
        )

        mock_update.return_value = True
        mock_test.return_value = True
        assert qt_main() == 0

        mock_update.return_value = False
        assert qt_main() == 1


class TestDumpNames:
    def test_dump_names_main(self):
        import sys

        mock_dm_control = MagicMock()
        mock_suite = MagicMock()
        mock_dm_control.suite = mock_suite

        env_mock = MagicMock()
        mock_suite.load.return_value = env_mock

        env_mock.physics.model.ngeom = 2
        env_mock.physics.model.nbody = 1
        env_mock.physics.model.id2name.side_effect = lambda i, type: f"{type}_{i}"

        with patch.dict(sys.modules, {"dm_control": mock_dm_control}):
            from src.engines.physics_engines.mujoco.docker.dump_names import (
                main as dump_names_main,
            )

            dump_names_main()

        mock_suite.load.assert_called_once_with(
            domain_name="humanoid_CMU", task_name="stand"
        )
        assert env_mock.physics.model.id2name.call_count == 3


class TestMeasureHeight:
    def test_measure_height_main(self):
        import sys

        mock_dm_control = MagicMock()
        mock_suite = MagicMock()
        mock_dm_control.suite = mock_suite

        env_mock = MagicMock()
        mock_suite.load.return_value = env_mock

        env_mock.physics.named.data.xpos = {
            "head": [0, 0, 1.5],
            "lfoot": [0, 0.1, 0],
            "rfoot": [0, -0.1, 0.05],
        }

        env_mock.physics.model.ngeom = 1
        env_mock.physics.model.id2name.return_value = "head_geom"
        env_mock.physics.data.geom_xpos = [[0, 0, 1.6]]
        env_mock.physics.model.geom_size = [[0.1, 0, 0]]

        with patch.dict(sys.modules, {"dm_control": mock_dm_control}):
            from src.engines.physics_engines.mujoco.docker.measure_height import (
                main as measure_height_main,
            )

            measure_height_main()

        mock_suite.load.assert_called_once()


class TestExampleDynamicStance:
    def test_main(self):
        import sys

        mock_dm_control = MagicMock()
        mock_suite = MagicMock()
        mock_suite.__file__ = "dummy/path.py"
        mock_mjcf = MagicMock()
        mock_dm_control.suite = mock_suite
        mock_dm_control.mjcf = mock_mjcf

        env_mock = MagicMock()
        mock_suite.load.return_value = env_mock

        physics_mock = MagicMock()
        physics_mock.model.ngeom = 1
        physics_mock.model.id2name.return_value = "head"

        with patch.dict(sys.modules, {"dm_control": mock_dm_control}):
            from src.engines.physics_engines.mujoco.docker.example_dynamic_stance import (
                main,
            )

            with (
                patch(
                    "src.engines.physics_engines.mujoco.docker.example_dynamic_stance.imageio.mimsave"
                ) as mock_mimsave,
                patch("builtins.open"),
                patch("pathlib.Path.exists", return_value=True),
                patch(
                    "src.engines.physics_engines.mujoco.docker.example_dynamic_stance.Path.exists",
                    return_value=True,
                ),
                patch(
                    "src.engines.physics_engines.mujoco.docker.example_dynamic_stance._setup_physics",
                    return_value=physics_mock,
                ),
            ):
                main()
            mock_mimsave.assert_called_once()


class TestExampleGolfSwing:
    def test_main(self):
        import sys

        mock_dm_control = MagicMock()
        mock_suite = MagicMock()
        mock_dm_control.suite = mock_suite

        env_mock = MagicMock()
        mock_suite.load.return_value = env_mock

        env_mock.physics.model.ngeom = 1
        env_mock.physics.model.id2name.return_value = "torso"

        with patch.dict(sys.modules, {"dm_control": mock_dm_control}):
            from src.engines.physics_engines.mujoco.docker.example_golf_swing import (
                main,
            )

            with patch(
                "src.engines.physics_engines.mujoco.docker.example_golf_swing.imageio.mimsave"
            ) as mock_mimsave:
                main()

        mock_suite.load.assert_called_once()
        mock_mimsave.assert_called_once()


class TestExampleHumanoid:
    def test_main(self):
        import sys

        mock_dm_control = MagicMock()
        mock_suite = MagicMock()
        mock_dm_control.suite = mock_suite

        env_mock = MagicMock()
        mock_suite.load.return_value = env_mock

        action_spec_mock = MagicMock()
        action_spec_mock.minimum = -1.0
        action_spec_mock.maximum = 1.0
        action_spec_mock.shape = (10,)
        env_mock.action_spec.return_value = action_spec_mock

        with patch.dict(sys.modules, {"dm_control": mock_dm_control}):
            from src.engines.physics_engines.mujoco.docker.example_humanoid import main

            with patch(
                "src.engines.physics_engines.mujoco.docker.example_humanoid.imageio.mimsave"
            ) as mock_mimsave:
                main()

        mock_suite.load.assert_called_once_with(
            domain_name="humanoid", task_name="walk"
        )
        assert env_mock.step.call_count == 200
        mock_mimsave.assert_called_once()


class TestInspectGeoms:
    def test_main(self):
        import sys

        mock_dm_control = MagicMock()
        mock_suite = MagicMock()
        mock_dm_control.suite = mock_suite

        env_mock = MagicMock()
        mock_suite.load.return_value = env_mock

        env_mock.physics.model.ngeom = 2
        env_mock.physics.model.nbody = 2
        env_mock.physics.model.id2name.side_effect = lambda i, type: f"{type}_{i}"

        with patch.dict(sys.modules, {"dm_control": mock_dm_control}):
            from src.engines.physics_engines.mujoco.docker.inspect_geoms import main

            main()

        mock_suite.load.assert_called_once_with(
            domain_name="humanoid_CMU", task_name="stand"
        )
        assert env_mock.physics.model.id2name.call_count == 4


class TestInspectHumanoid:
    def test_main(self):
        import sys

        mock_dm_control = MagicMock()
        mock_suite = MagicMock()
        mock_dm_control.suite = mock_suite

        env_mock = MagicMock()
        mock_suite.load.return_value = env_mock

        env_mock.physics.model.njnt = 2
        env_mock.physics.model.id2name.side_effect = lambda i, type: f"{type}_{i}"

        env_mock.observation_spec.return_value = {"pos": 1, "vel": 2}

        named_data = MagicMock()
        named_data.qpos.axes.row.names = ["q1", "q2"]
        named_data.ctrl.axes.row.names = ["c1"]
        env_mock.physics.named.data = named_data

        with patch.dict(sys.modules, {"dm_control": mock_dm_control}):
            from src.engines.physics_engines.mujoco.docker.inspect_humanoid import main

            main()

        mock_suite.load.assert_called_once_with(
            domain_name="humanoid", task_name="stand"
        )
        assert env_mock.physics.model.id2name.call_count == 2


class TestVerifyDmControl:
    def test_main_success(self):
        import sys

        mock_dm_control = MagicMock()
        mock_suite = MagicMock()
        mock_dm_control.suite = mock_suite

        env_mock = MagicMock()
        mock_suite.load.return_value = env_mock

        action_spec_mock = MagicMock()
        action_spec_mock.minimum = -1.0
        action_spec_mock.maximum = 1.0
        action_spec_mock.shape = (10,)
        env_mock.action_spec.return_value = action_spec_mock

        env_mock.physics.render.return_value = MagicMock(shape=(480, 640, 3))

        with patch.dict(sys.modules, {"dm_control": mock_dm_control}):
            from src.engines.physics_engines.mujoco.docker.verify_dm_control import main

            main()

        mock_suite.load.assert_called_once_with(
            domain_name="cartpole", task_name="swingup"
        )
        env_mock.reset.assert_called_once()
        env_mock.step.assert_called_once()
        env_mock.physics.render.assert_called_once()

    @patch("sys.exit")
    def test_main_import_error(self, mock_exit):
        mock_exit.side_effect = SystemExit(1)
        import sys

        with patch.dict(sys.modules, {"dm_control": None}):
            from src.engines.physics_engines.mujoco.docker.verify_dm_control import main

            with pytest.raises(SystemExit):
                main()
        mock_exit.assert_called_once_with(1)


class TestRebuildDocker:
    @patch("src.engines.physics_engines.mujoco.rebuild_docker.run_command")
    def test_rebuild_docker_image_success(self, mock_run_command):
        mock_run_command.return_value = MagicMock(returncode=0)
        from src.engines.physics_engines.mujoco.rebuild_docker import (
            rebuild_docker_image,
        )

        with patch(
            "src.engines.physics_engines.mujoco.rebuild_docker.Path.exists",
            return_value=True,
        ):
            assert rebuild_docker_image() is True

    @patch("src.engines.physics_engines.mujoco.rebuild_docker.run_command")
    def test_rebuild_docker_image_failure(self, mock_run_command):
        mock_run_command.return_value = MagicMock(returncode=1)
        from src.engines.physics_engines.mujoco.rebuild_docker import (
            rebuild_docker_image,
        )

        with patch(
            "src.engines.physics_engines.mujoco.rebuild_docker.Path.exists",
            return_value=True,
        ):
            assert rebuild_docker_image() is False

    @patch("src.engines.physics_engines.mujoco.rebuild_docker.rebuild_docker_image")
    def test_main(self, mock_rebuild):
        mock_rebuild.return_value = True
        from src.engines.physics_engines.mujoco.rebuild_docker import main

        assert main() == 0

        mock_rebuild.return_value = False
        assert main() == 1


class TestDockerTestDependencies:
    @patch("src.engines.physics_engines.mujoco.docker_test_dependencies.subprocess.run")
    def test_test_pip_list(self, mock_run):
        mock_run.return_value = MagicMock(stdout="Package Version")
        from src.engines.physics_engines.mujoco.docker_test_dependencies import (
            test_pip_list,
        )

        test_pip_list()
        mock_run.assert_called_once()

    @patch("importlib.util.find_spec")
    def test_test_specific_imports(self, mock_find_spec):
        mock_find_spec.return_value = True
        import sys

        mock_defusedxml = MagicMock()
        mock_defusedxml.__file__ = "path"
        mock_defusedxml.__version__ = "1.0"
        with patch.dict(sys.modules, {"defusedxml": mock_defusedxml}):
            from src.engines.physics_engines.mujoco.docker_test_dependencies import (
                test_specific_imports,
            )

            assert test_specific_imports() is True

    @patch("src.engines.physics_engines.mujoco.docker_test_dependencies.subprocess.run")
    def test_test_environment_activation(self, mock_run):
        mock_run.return_value = MagicMock(stdout="/opt/mujoco-env/bin/python3")
        with patch("os.path.exists", return_value=True):
            from src.engines.physics_engines.mujoco.docker_test_dependencies import (
                test_environment_activation,
            )

            test_environment_activation()
            assert mock_run.call_count == 2

    @patch(
        "src.engines.physics_engines.mujoco.docker_test_dependencies.test_specific_imports"
    )
    @patch("src.engines.physics_engines.mujoco.docker_test_dependencies.test_pip_list")
    @patch(
        "src.engines.physics_engines.mujoco.docker_test_dependencies.test_environment_activation"
    )
    @patch(
        "src.engines.physics_engines.mujoco.docker_test_dependencies.test_python_environment"
    )
    def test_main(self, mock_python, mock_env, mock_pip, mock_imports):
        mock_imports.return_value = True
        from src.engines.physics_engines.mujoco.docker_test_dependencies import main

        assert main() == 0

        mock_imports.return_value = False
        assert main() == 1
