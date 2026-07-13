from __future__ import annotations

import ast
import contextlib
import importlib
import io
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SOURCE = PROJECT_ROOT / "configs"


def _startup_module():
    """Import lazily so static launcher tests still report useful failures."""

    return importlib.import_module("deep_shark_studio.startup")


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class WindowsLauncherContractTests(unittest.TestCase):
    def test_launcher_is_relocatable_and_has_interpreter_fallbacks(self) -> None:
        launcher = (PROJECT_ROOT / "StartDeepSharkViewStudio.cmd").read_text(
            encoding="utf-8"
        )
        folded = launcher.casefold()

        self.assertIn("%~dp0", folded)
        self.assertIsNone(
            re.search(r"(?i)(?<![%a-z])[a-z]:[\\/]", launcher),
            "the distributed launcher must not contain a machine-specific drive path",
        )
        self.assertIn(r".venv\scripts\python.exe", folded)
        self.assertIn("deep_shark_python", folded)
        self.assertIn(
            r"..\..\envs\deep-shark-view-studio\scripts\python.exe",
            folded,
        )
        self.assertIn("import numpy, cv2, yaml, pyside6", folded)
        self.assertIn("py -3", folded)
        self.assertRegex(folded, r"\bpython(?:\.exe)?\b")
        self.assertLess(folded.index("deep_shark_python"), folded.index(r".venv\scripts\python.exe"))
        self.assertLess(folded.index(r".venv\scripts\python.exe"), folded.index("py -3"))
        self.assertLess(folded.index("py -3"), folded.rindex("python"))
        self.assertGreaterEqual(folded.count("app.py %*"), 3)
        self.assertIn('if "%~1"=="" pause', folded)
        self.assertIn("exit /b", folded)

    def test_python_entrypoint_defers_gui_import_to_startup_preflight(self) -> None:
        entrypoint = PROJECT_ROOT / "app.py"
        source = entrypoint.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(entrypoint))
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }

        self.assertIn("deep_shark_studio.startup", imported_modules)
        self.assertFalse(
            any(module.startswith("deep_shark_studio.gui") for module in imported_modules),
            "GUI dependencies must only be imported after dependency/config preflight",
        )

    def test_distribution_docs_do_not_require_a_personal_drive(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotRegex(readme, r"(?i)\b[a-z]:\\develop\\")
        self.assertNotIn("deep-shark-view-studio\\scripts\\python.exe", readme.casefold())


class ConfigPreflightContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temporary_directory.name) / "configs"
        self.config_dir.mkdir()
        shutil.copy2(CONFIG_SOURCE / "calibration.yaml", self.config_dir / "calibration.yaml")
        shutil.copy2(
            CONFIG_SOURCE / "cameras.example.yaml",
            self.config_dir / "cameras.example.yaml",
        )
        shutil.copy2(
            CONFIG_SOURCE / "network.example.yaml",
            self.config_dir / "network.example.yaml",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_default_preflight_is_strictly_read_only_when_configs_are_missing(self) -> None:
        startup = _startup_module()
        before = _tree_bytes(self.config_dir)

        with self.assertRaises(startup.ConfigPreflightError) as raised:
            startup.preflight_configs(self.config_dir)

        self.assertEqual(before, _tree_bytes(self.config_dir))
        self.assertFalse((self.config_dir / "cameras.yaml").exists())
        self.assertFalse((self.config_dir / "network.yaml").exists())
        message = str(raised.exception).casefold()
        self.assertIn("cameras.yaml", message)
        self.assertIn("network.yaml", message)
        self.assertIn("initialize", message)

    def test_explicit_initialization_copies_only_missing_local_configs(self) -> None:
        startup = _startup_module()
        existing_cameras = {
            "active_camera_count": 0,
            "camera_order": [],
            "cameras": {},
            "language": "en_US",
            "performance": {},
        }
        _write_yaml(self.config_dir / "cameras.yaml", existing_cameras)
        cameras_before = (self.config_dir / "cameras.yaml").read_bytes()

        startup.preflight_configs(self.config_dir, initialize=True)

        self.assertEqual(cameras_before, (self.config_dir / "cameras.yaml").read_bytes())
        self.assertEqual(
            yaml.safe_load((self.config_dir / "network.example.yaml").read_text("utf-8")),
            yaml.safe_load((self.config_dir / "network.yaml").read_text("utf-8")),
        )
        self.assertFalse(
            any(path.suffix == ".tmp" for path in self.config_dir.iterdir()),
            "atomic initialization must not leave temporary files behind",
        )

    def test_initialization_never_synthesizes_or_replaces_calibration(self) -> None:
        startup = _startup_module()
        calibration_path = self.config_dir / "calibration.yaml"
        calibration_before = calibration_path.read_bytes()

        startup.preflight_configs(self.config_dir, initialize=True)
        self.assertEqual(calibration_before, calibration_path.read_bytes())

        calibration_path.unlink()
        before = _tree_bytes(self.config_dir)
        with self.assertRaises(startup.ConfigPreflightError) as raised:
            startup.preflight_configs(self.config_dir, initialize=True)

        self.assertEqual(before, _tree_bytes(self.config_dir))
        self.assertIn("calibration.yaml", str(raised.exception))

    def test_invalid_existing_yaml_is_rejected_and_not_overwritten(self) -> None:
        startup = _startup_module()
        cameras_path = self.config_dir / "cameras.yaml"
        cameras_path.write_bytes(b"cameras: [\n")
        corrupt_bytes = cameras_path.read_bytes()

        with self.assertRaises(startup.ConfigPreflightError) as raised:
            startup.preflight_configs(self.config_dir, initialize=True)

        self.assertEqual(corrupt_bytes, cameras_path.read_bytes())
        self.assertIn("cameras.yaml", str(raised.exception))
        self.assertRegex(str(raised.exception).casefold(), r"invalid|yaml|parse")

    def test_invalid_template_cannot_leave_a_half_initialized_config_set(self) -> None:
        startup = _startup_module()
        (self.config_dir / "network.example.yaml").write_bytes(b"output: [\n")
        before = _tree_bytes(self.config_dir)

        with self.assertRaises(startup.ConfigPreflightError):
            startup.preflight_configs(self.config_dir, initialize=True)

        self.assertEqual(before, _tree_bytes(self.config_dir))
        self.assertFalse((self.config_dir / "cameras.yaml").exists())
        self.assertFalse((self.config_dir / "network.yaml").exists())

    def test_publish_failure_rolls_back_every_config_created_by_the_call(self) -> None:
        startup = _startup_module()
        before = _tree_bytes(self.config_dir)
        real_link = startup.os.link
        publish_count = 0

        def fail_second_publish(source: Path, target: Path) -> None:
            nonlocal publish_count
            publish_count += 1
            if publish_count == 2:
                raise OSError("simulated publish failure")
            real_link(source, target)

        with mock.patch(
            "deep_shark_studio.startup.os.link",
            side_effect=fail_second_publish,
        ):
            with self.assertRaises(startup.ConfigPreflightError) as raised:
                startup.preflight_configs(self.config_dir, initialize=True)

        self.assertEqual(before, _tree_bytes(self.config_dir))
        self.assertIn("failed", str(raised.exception).casefold())
        self.assertFalse((self.config_dir / "cameras.yaml").exists())
        self.assertFalse((self.config_dir / "network.yaml").exists())

    def test_rollback_preserves_a_target_replaced_by_a_concurrent_writer(self) -> None:
        startup = _startup_module()
        real_link = startup.os.link
        publish_count = 0
        concurrent_payload = b"concurrent writer owns this path\n"

        def replace_first_target_then_fail(source: Path, target: Path) -> None:
            nonlocal publish_count
            publish_count += 1
            if publish_count == 1:
                real_link(source, target)
                return

            concurrent_path = self.config_dir / ".concurrent-cameras.yaml"
            concurrent_path.write_bytes(concurrent_payload)
            startup.os.replace(concurrent_path, self.config_dir / "cameras.yaml")
            raise OSError("simulated second publish failure")

        with mock.patch(
            "deep_shark_studio.startup.os.link",
            side_effect=replace_first_target_then_fail,
        ):
            with self.assertRaises(startup.ConfigPreflightError) as raised:
                startup.preflight_configs(self.config_dir, initialize=True)

        self.assertEqual(
            concurrent_payload,
            (self.config_dir / "cameras.yaml").read_bytes(),
            "rollback must not delete a path replaced by another writer",
        )
        self.assertFalse((self.config_dir / "network.yaml").exists())
        message = str(raised.exception).casefold()
        self.assertIn("rollback conflict", message)
        self.assertIn("preserved", message)
        self.assertFalse(any(path.suffix == ".tmp" for path in self.config_dir.iterdir()))

    def test_final_validation_failure_rolls_back_published_configs(self) -> None:
        startup = _startup_module()
        before = _tree_bytes(self.config_dir)
        real_validate = startup._validate_config

        def fail_on_published_cameras(
            path: Path,
            *,
            display_name: str | None = None,
        ) -> None:
            candidate = Path(path)
            if candidate == self.config_dir / "cameras.yaml":
                raise startup.ConfigPreflightError("simulated final validation failure")
            real_validate(candidate, display_name=display_name)

        with mock.patch(
            "deep_shark_studio.startup._validate_config",
            side_effect=fail_on_published_cameras,
        ):
            with self.assertRaises(startup.ConfigPreflightError) as raised:
                startup.preflight_configs(self.config_dir, initialize=True)

        self.assertEqual(before, _tree_bytes(self.config_dir))
        self.assertIn("final validation failure", str(raised.exception))
        self.assertFalse((self.config_dir / "cameras.yaml").exists())
        self.assertFalse((self.config_dir / "network.yaml").exists())

    def test_yaml_roots_must_be_mappings(self) -> None:
        startup = _startup_module()
        shutil.copy2(
            self.config_dir / "cameras.example.yaml",
            self.config_dir / "cameras.yaml",
        )
        _write_yaml(self.config_dir / "network.yaml", ["not", "a", "mapping"])

        with self.assertRaises(startup.ConfigPreflightError) as raised:
            startup.preflight_configs(self.config_dir)

        self.assertIn("network.yaml", str(raised.exception))
        self.assertIn("mapping", str(raised.exception).casefold())

    def test_boot_critical_performance_values_are_validated_before_gui_import(self) -> None:
        startup = _startup_module()
        cameras = yaml.safe_load(
            (self.config_dir / "cameras.example.yaml").read_text("utf-8")
        )
        cameras["performance"]["process_fps"] = "fast"
        _write_yaml(self.config_dir / "cameras.yaml", cameras)
        shutil.copy2(
            self.config_dir / "network.example.yaml",
            self.config_dir / "network.yaml",
        )

        with self.assertRaises(startup.ConfigPreflightError) as raised:
            startup.preflight_configs(self.config_dir)

        self.assertIn("process_fps", str(raised.exception))


class CleanCheckoutAssetsTests(unittest.TestCase):
    def test_tracked_config_seed_files_are_valid_and_complete(self) -> None:
        required = {
            "calibration.yaml",
            "cameras.example.yaml",
            "network.example.yaml",
        }
        self.assertTrue(required.issubset({path.name for path in CONFIG_SOURCE.iterdir()}))

        calibration = yaml.safe_load((CONFIG_SOURCE / "calibration.yaml").read_text("utf-8"))
        cameras = yaml.safe_load((CONFIG_SOURCE / "cameras.example.yaml").read_text("utf-8"))
        network = yaml.safe_load((CONFIG_SOURCE / "network.example.yaml").read_text("utf-8"))

        self.assertIsInstance(calibration, dict)
        self.assertIn("canvas", calibration)
        self.assertIn("topologies", calibration)
        self.assertIsInstance(cameras, dict)
        self.assertIn("camera_order", cameras)
        self.assertIn("cameras", cameras)
        self.assertIn("performance", cameras)
        self.assertIsInstance(network, dict)
        self.assertIn("output", network)
        self.assertIn("udp_legacy", network)

    def test_missing_runtime_dependency_has_an_actionable_error(self) -> None:
        startup = _startup_module()
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, *args, **kwargs):
            if name == "PySide6":
                return None
            return real_find_spec(name, *args, **kwargs)

        with mock.patch(
            "deep_shark_studio.startup.importlib.util.find_spec",
            side_effect=fake_find_spec,
        ):
            with self.assertRaises(startup.DependencyPreflightError) as raised:
                startup.check_runtime_dependencies()

        message = str(raised.exception)
        self.assertIn("PySide6", message)
        self.assertIn("requirements.txt", message)
        self.assertNotIn(r"D:\Develop", message)


class StartupMainContractTests(unittest.TestCase):
    def test_missing_configs_fail_before_gui_import(self) -> None:
        startup = _startup_module()
        events: list[str] = []

        def missing_configs(*_args, **_kwargs) -> None:
            events.append("configs")
            raise startup.ConfigPreflightError("missing cameras.yaml")

        with (
            mock.patch.object(
                startup,
                "check_runtime_dependencies",
                side_effect=lambda: events.append("dependencies"),
            ),
            mock.patch.object(startup, "preflight_configs", side_effect=missing_configs),
            mock.patch.object(startup.importlib, "import_module") as import_module,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = startup.main([])

        self.assertEqual(2, result)
        self.assertEqual(["dependencies", "configs"], events)
        import_module.assert_not_called()

    def test_initialization_precedes_lazy_gui_import_and_preserves_qt_args(self) -> None:
        startup = _startup_module()
        events: list[object] = []
        original_argv = list(sys.argv)

        def dependencies() -> None:
            events.append("dependencies")

        def configs(*, initialize: bool = False) -> None:
            events.append(("configs", initialize))

        def gui_main() -> int:
            events.append(("gui", list(sys.argv)))
            return 17

        gui_module = types.SimpleNamespace(main=gui_main)

        def import_module(name: str):
            events.append(("import", name))
            return gui_module

        with (
            mock.patch.object(startup, "check_runtime_dependencies", dependencies),
            mock.patch.object(startup, "preflight_configs", configs),
            mock.patch.object(startup.importlib, "import_module", side_effect=import_module),
        ):
            result = startup.main(
                ["--initialize-configs", "-style", "fusion", "--platform", "offscreen"]
            )

        self.assertEqual(17, result)
        self.assertEqual("dependencies", events[0])
        self.assertEqual(("configs", True), events[1])
        self.assertEqual(
            ("import", "deep_shark_studio.gui.main_window"),
            events[2],
        )
        self.assertEqual(
            ("gui", [original_argv[0], "-style", "fusion", "--platform", "offscreen"]),
            events[3],
        )
        self.assertEqual(original_argv, sys.argv)

    def test_preflight_only_never_imports_gui(self) -> None:
        startup = _startup_module()
        with (
            mock.patch.object(startup, "check_runtime_dependencies"),
            mock.patch.object(startup, "preflight_configs"),
            mock.patch.object(startup.importlib, "import_module") as import_module,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = startup.main(["--preflight-only"])

        self.assertEqual(0, result)
        import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
