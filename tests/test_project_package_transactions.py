from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from deep_shark_studio.config import file_revision, load_yaml, save_yaml
from deep_shark_studio import project_package
from deep_shark_studio.project_package import package as package_module
from deep_shark_studio.startup import preflight_configs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_FILES = (
    "calibration.yaml",
    "cameras.yaml",
    "network.yaml",
    "active_project.yaml",
)
CONFIG_FILES = ACTIVE_FILES[:3]


class ProjectPackageTransactionTests(unittest.TestCase):
    """Transaction and restart contracts for portable project packages.

    Every fixture lives below a TemporaryDirectory.  In particular, these
    tests must never activate a package into the repository's real configs/.
    """

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.source_config_dir = self.root / "source_configs"
        self._write_config_set(self.source_config_dir, marker="package-new")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_export_failure_removes_staging_and_never_leaves_partial_package(self) -> None:
        output_root = self.root / "failed_export"
        output_root.mkdir()
        real_copy2 = shutil.copy2
        config_copy_count = 0

        def fail_on_second_config_copy(source, destination, *args, **kwargs):
            nonlocal config_copy_count
            source_path = Path(source)
            if source_path.parent == self.source_config_dir:
                config_copy_count += 1
                if config_copy_count == 2:
                    raise OSError("injected export copy failure")
            return real_copy2(source, destination, *args, **kwargs)

        with mock.patch.object(package_module.shutil, "copy2", side_effect=fail_on_second_config_copy):
            with self.assertRaisesRegex(OSError, "injected export copy failure"):
                self._export(output_root, project_name="atomic_failure")

        self.assertEqual([], list(output_root.iterdir()))

    def test_successful_export_is_not_published_until_copying_is_complete(self) -> None:
        output_root = self.root / "successful_export"
        output_root.mkdir()
        real_copy2 = shutil.copy2
        visible_final_directories_during_copy: list[Path] = []

        def observe_copy(source, destination, *args, **kwargs):
            visible_final_directories_during_copy.extend(
                child
                for child in output_root.iterdir()
                if child.is_dir() and child.name.startswith("DeepShark_Project_")
            )
            return real_copy2(source, destination, *args, **kwargs)

        with mock.patch.object(package_module.shutil, "copy2", side_effect=observe_copy):
            result = self._export(output_root, project_name="atomic_success")

        self.assertEqual([], visible_final_directories_during_copy)
        self.assertTrue(result.package_root.is_dir())
        self.assertTrue(result.manifest_path.is_file())
        self.assertEqual([result.package_root], list(output_root.iterdir()))

    def test_export_self_validation_failure_never_publishes_a_package(self) -> None:
        output_root = self.root / "invalid_export"
        output_root.mkdir()
        (self.source_config_dir / "network.yaml").write_text(
            "output: [unterminated",
            encoding="utf-8",
        )

        with self.assertRaises(Exception):
            self._export(output_root, project_name="invalid_source")

        self.assertEqual(
            [],
            list(output_root.iterdir()),
            "an invalid staging tree must never become a visible package",
        )

    def test_export_rejects_machine_specific_absolute_config_paths(self) -> None:
        output_root = self.root / "absolute_config_export"
        output_root.mkdir()
        cameras = load_yaml(self.source_config_dir / "cameras.yaml")
        cameras["cameras"]["front"]["source_type"] = "video_file"
        cameras["cameras"]["front"]["source"] = "D:/private/capture.mp4"
        save_yaml(self.source_config_dir / "cameras.yaml", cameras)

        with self.assertRaises(project_package.ProjectPackageExportError):
            self._export(output_root, project_name="absolute_config")

        self.assertEqual([], list(output_root.iterdir()))

    def test_export_rejects_b2_dependency_escape_before_any_outside_copy(self) -> None:
        output_root = self.root / "escape_export"
        output_root.mkdir()
        candidate_root = self.root / "nested" / "source" / "malicious_b2"
        candidate_root.mkdir(parents=True)
        outside_source = self.root / "outside_source.npz"
        self._write_valid_remap(outside_source)
        save_yaml(
            candidate_root / "candidate.yaml",
            {
                "format": "DeepSharkFisheyeCalibrationCandidate",
                "resolution": [1920, 1080],
                "virtual_panorama": {
                    "canvas_size": [4, 3],
                    "files": {
                        "remaps": {
                            "front_left": "../../../outside_source.npz",
                        }
                    },
                },
            },
        )
        outside_before = outside_source.read_bytes()

        with self.assertRaises(Exception):
            self._export(
                output_root,
                project_name="dependency_escape",
                b2_candidate_path=candidate_root / "candidate.yaml",
            )

        self.assertEqual(outside_before, outside_source.read_bytes())
        self.assertEqual([], list(output_root.iterdir()))

    def test_activation_backs_up_all_four_files_before_staging_and_never_copies_directly(self) -> None:
        exported = self._export(self.root / "exports", project_name="backup_first")
        active_dir = self.root / "active"
        self._write_config_set(active_dir, marker="active-old")
        self._write_old_active_state(active_dir)
        old_bytes = self._read_active_bytes(active_dir)
        backup_root = self.root / "backups"
        real_copy2 = shutil.copy2
        active_snapshots_during_package_copy: list[dict[str, bytes]] = []

        def observe_copy(source, destination, *args, **kwargs):
            source_path = Path(source).resolve()
            result = real_copy2(source, destination, *args, **kwargs)
            if source_path.parent == (exported.package_root / "configs").resolve():
                backups = [path for path in backup_root.iterdir() if path.is_dir()]
                self.assertEqual(1, len(backups), "backup must exist before package config staging")
                for name in ACTIVE_FILES:
                    self.assertEqual(old_bytes[name], (backups[0] / name).read_bytes())
                active_snapshots_during_package_copy.append(
                    self._read_active_bytes(active_dir)
                )
            return result

        with mock.patch.object(package_module.shutil, "copy2", side_effect=observe_copy):
            activation = project_package.activate_project_package(
                exported.manifest_path,
                active_config_dir=active_dir,
                backup_root=backup_root,
            )

        self.assertTrue(active_snapshots_during_package_copy)
        self.assertTrue(
            all(snapshot == old_bytes for snapshot in active_snapshots_during_package_copy),
            "package configs must be copied to staging, never directly over active files",
        )
        self.assertEqual(
            (active_dir / "active_project.yaml").resolve(),
            activation.active_state_path.resolve(),
        )
        for name in CONFIG_FILES:
            self.assertEqual(
                load_yaml(exported.package_root / "configs" / name),
                load_yaml(active_dir / name),
            )

    def test_activation_rolls_back_every_file_when_second_or_third_publish_fails(self) -> None:
        exported = self._export(self.root / "exports", project_name="rollback")

        for fail_at in (2, 3):
            with self.subTest(fail_at=fail_at):
                active_dir = self.root / f"active_fail_{fail_at}"
                self._write_config_set(active_dir, marker=f"active-old-{fail_at}")
                self._write_old_active_state(active_dir)
                old_bytes = self._read_active_bytes(active_dir)
                backup_root = self.root / f"backups_fail_{fail_at}"
                real_replace = os.replace
                publish_count = 0
                failure_injected = False

                def fail_during_publish(source, destination, *args, **kwargs):
                    nonlocal publish_count, failure_injected
                    destination_path = Path(destination)
                    if (
                        destination_path.parent.resolve() == active_dir.resolve()
                        and destination_path.name in ACTIVE_FILES
                    ):
                        publish_count += 1
                        if publish_count == fail_at and not failure_injected:
                            failure_injected = True
                            raise OSError(f"injected publish failure {fail_at}")
                    return real_replace(source, destination, *args, **kwargs)

                with mock.patch("os.replace", side_effect=fail_during_publish):
                    # The public API may wrap the low-level OSError in a
                    # package-specific activation error; rollback semantics,
                    # not the wrapper exception type, are the contract here.
                    with self.assertRaises(Exception):
                        project_package.activate_project_package(
                            exported.manifest_path,
                            active_config_dir=active_dir,
                            backup_root=backup_root,
                        )

                self.assertTrue(failure_injected, "test must reach the transactional publish phase")
                self.assertEqual(old_bytes, self._read_active_bytes(active_dir))
                backups = [path for path in backup_root.iterdir() if path.is_dir()]
                self.assertEqual(1, len(backups))
                for name in ACTIVE_FILES:
                    self.assertEqual(old_bytes[name], (backups[0] / name).read_bytes())
                self.assertEqual(
                    set(ACTIVE_FILES),
                    {path.name for path in active_dir.iterdir()},
                    "failed activation must not leave transaction debris or a mixed config set",
                )

    def test_activation_rolls_back_when_active_state_was_previously_absent(self) -> None:
        exported = self._export(self.root / "exports", project_name="absent_state")
        active_dir = self.root / "active_without_state"
        self._write_config_set(active_dir, marker="old-without-state")
        old_bytes = {
            name: (active_dir / name).read_bytes()
            for name in CONFIG_FILES
        }
        real_replace = os.replace
        failure_injected = False

        def fail_on_active_state_publish(source, destination, *args, **kwargs):
            nonlocal failure_injected
            destination_path = Path(destination)
            if (
                destination_path.parent.resolve() == active_dir.resolve()
                and destination_path.name == "active_project.yaml"
                and not failure_injected
            ):
                failure_injected = True
                raise OSError("injected active state publish failure")
            return real_replace(source, destination, *args, **kwargs)

        with mock.patch("os.replace", side_effect=fail_on_active_state_publish):
            with self.assertRaisesRegex(
                OSError,
                "injected active state publish failure",
            ):
                project_package.activate_project_package(
                    exported.manifest_path,
                    active_config_dir=active_dir,
                    backup_root=self.root / "absent_state_backups",
                )

        self.assertTrue(failure_injected)
        self.assertFalse((active_dir / "active_project.yaml").exists())
        self.assertEqual(
            old_bytes,
            {name: (active_dir / name).read_bytes() for name in CONFIG_FILES},
        )
        self.assertEqual(
            set(CONFIG_FILES),
            {path.name for path in active_dir.iterdir()},
        )

    def test_backup_failure_cannot_touch_active_files_or_publish_partial_backup(self) -> None:
        exported = self._export(self.root / "exports", project_name="backup_failure")
        active_dir = self.root / "active_backup_failure"
        self._write_config_set(active_dir, marker="must-survive")
        self._write_old_active_state(active_dir)
        old_bytes = self._read_active_bytes(active_dir)
        backup_root = self.root / "failed_backups"
        real_copy2 = shutil.copy2
        backup_copy_count = 0

        def fail_during_backup(source, destination, *args, **kwargs):
            nonlocal backup_copy_count
            destination_path = Path(destination)
            try:
                destination_path.resolve().relative_to(backup_root.resolve())
            except ValueError:
                return real_copy2(source, destination, *args, **kwargs)
            backup_copy_count += 1
            if backup_copy_count == 2:
                raise OSError("injected backup copy failure")
            return real_copy2(source, destination, *args, **kwargs)

        with mock.patch.object(
            package_module.shutil,
            "copy2",
            side_effect=fail_during_backup,
        ):
            with self.assertRaisesRegex(OSError, "injected backup copy failure"):
                project_package.activate_project_package(
                    exported.manifest_path,
                    active_config_dir=active_dir,
                    backup_root=backup_root,
                )

        self.assertEqual(old_bytes, self._read_active_bytes(active_dir))
        self.assertEqual([], list(backup_root.iterdir()) if backup_root.exists() else [])

    def test_custom_active_directory_still_gets_a_default_backup(self) -> None:
        exported = self._export(self.root / "exports", project_name="default_backup")
        active_dir = self.root / "custom_active"
        self._write_config_set(active_dir, marker="old-default-backup")

        activation = project_package.activate_project_package(
            exported.manifest_path,
            active_config_dir=active_dir,
        )

        self.assertIsNotNone(activation.backup_path)
        self.assertTrue(activation.backup_path.is_dir())
        for name in CONFIG_FILES:
            self.assertTrue((activation.backup_path / name).is_file())

    def test_moved_package_activation_restores_runtime_defaults_after_restart(self) -> None:
        candidates = self._write_runtime_candidates()
        exported = self._export(
            self.root / "exports",
            project_name="portable_runtime",
            far_field_candidate_path=candidates["far"],
            near_field_candidate_path=candidates["near"],
            b2_candidate_path=candidates["b2"],
            fisheye_intrinsics_path=candidates["fisheye"],
            default_stitch_mode="near_field",
            live_stitch_strategy="template",
            use_far_field_custom=True,
            near_field_projection_source="fisheye_rectilinear_candidate",
        )
        moved_root = self.root / "another_drive" / exported.package_root.name
        moved_root.parent.mkdir(parents=True)
        shutil.copytree(exported.package_root, moved_root)
        shutil.rmtree(exported.package_root)

        validation = project_package.validate_project_package(moved_root, write_report=False)
        self.assertTrue(validation.valid, validation.errors)
        package_digest_before = self._tree_digest(moved_root)
        active_dir = self.root / "restart_active"
        self._write_config_set(active_dir, marker="before-moved-activation")
        activation = project_package.activate_project_package(
            moved_root,
            active_config_dir=active_dir,
            backup_root=self.root / "restart_backups",
        )

        self.assertEqual(package_digest_before, self._tree_digest(moved_root))
        state_path = active_dir / "active_project.yaml"
        self.assertEqual(state_path.resolve(), activation.active_state_path.resolve())
        state = load_yaml(state_path)
        moved_manifest = (moved_root / "project.dcsvs.yaml").resolve()
        self.assertEqual("DeepSharkActiveProject", state["format"])
        self.assertEqual(1, state["schema_version"])
        self.assertEqual(str(moved_manifest), state["manifest_path"])
        self.assertEqual(file_revision(moved_manifest), state["manifest_sha256"])
        preflight_configs(active_dir)

        # This public loader is the restart boundary: it may use only the
        # persisted active_project.yaml and the still-portable package.
        restored = project_package.load_active_project_runtime_defaults(
            active_config_dir=active_dir
        )
        self.assertIsNotNone(restored)
        self.assertEqual(moved_manifest, restored.manifest_path.resolve())
        self.assertEqual(moved_root.resolve(), restored.package_root.resolve())
        self.assertEqual("near_field", restored.default_stitch_mode)
        self.assertEqual("template", restored.live_stitch_strategy)
        self.assertTrue(restored.use_far_field_custom)
        self.assertEqual(
            "fisheye_rectilinear_candidate",
            restored.near_field_projection_source,
        )
        expected_paths = {
            "far_field_layout_candidate": moved_root / "candidates" / "far_field_layout" / "candidate.yaml",
            "near_field_layout_candidate": moved_root / "candidates" / "near_field_layout" / "candidate.yaml",
            "b2_candidate": moved_root / "candidates" / "b2_candidate" / "candidate.yaml",
            "fisheye_intrinsics_source": moved_root / "candidates" / "fisheye_intrinsics" / "candidate.yaml",
        }
        for attribute, expected in expected_paths.items():
            actual = getattr(restored, attribute)
            self.assertEqual(expected.resolve(), actual.resolve())
            self.assertTrue(actual.is_file())

        restarted = project_package.load_active_project_runtime_defaults(
            active_config_dir=active_dir
        )
        self.assertEqual(restored, restarted)

    def test_restart_loader_rejects_manifest_changed_after_activation(self) -> None:
        exported = self._export(self.root / "exports", project_name="state_hash")
        active_dir = self.root / "hash_active"
        self._write_config_set(active_dir, marker="old")
        project_package.activate_project_package(
            exported.manifest_path,
            active_config_dir=active_dir,
            backup_root=self.root / "hash_backups",
        )
        manifest = load_yaml(exported.manifest_path)
        manifest["project"]["name"] = "tampered-after-activation"
        save_yaml(exported.manifest_path, manifest)

        with self.assertRaises(project_package.ActiveProjectStateError):
            project_package.load_active_project_runtime_defaults(
                active_config_dir=active_dir
            )

    def _export(self, output_root: Path, project_name: str, **kwargs):
        return project_package.export_project_package(
            output_root,
            project_name=project_name,
            calibration_config_path=self.source_config_dir / "calibration.yaml",
            cameras_config_path=self.source_config_dir / "cameras.yaml",
            network_config_path=self.source_config_dir / "network.yaml",
            **kwargs,
        )

    @staticmethod
    def _repository_config(name: str) -> dict:
        source_name = {
            "calibration.yaml": "calibration.yaml",
            "cameras.yaml": "cameras.example.yaml",
            "network.yaml": "network.example.yaml",
        }[name]
        return load_yaml(REPOSITORY_ROOT / "configs" / source_name)

    def _write_config_set(self, directory: Path, marker: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in CONFIG_FILES:
            data = self._repository_config(name)
            data["transaction_test_marker"] = marker
            save_yaml(directory / name, data)

    @staticmethod
    def _write_old_active_state(active_dir: Path) -> None:
        save_yaml(
            active_dir / "active_project.yaml",
            {
                "format": "DeepSharkActiveProject",
                "schema_version": 1,
                "manifest_path": "C:/old/location/project.dcsvs.yaml",
                "manifest_sha256": "old-manifest-revision",
            },
        )

    @staticmethod
    def _read_active_bytes(active_dir: Path) -> dict[str, bytes]:
        return {name: (active_dir / name).read_bytes() for name in ACTIVE_FILES}

    @staticmethod
    def _tree_digest(root: Path) -> str:
        digest = sha256()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _write_valid_remap(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            map_x=np.zeros((3, 4), dtype=np.float32),
            map_y=np.zeros((3, 4), dtype=np.float32),
            valid_mask=np.ones((3, 4), dtype=np.uint8),
        )

    def _write_runtime_candidates(self) -> dict[str, Path]:
        b2 = self.root / "runtime_candidates" / "b2"
        intrinsics = b2 / "intrinsics"
        intrinsics.mkdir(parents=True)
        canvas_width, canvas_height = 1800, 600
        map_x = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        map_y = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        valid_mask = np.ones((canvas_height, canvas_width), dtype=np.uint8)
        remaps: dict[str, str] = {}
        transforms: dict[str, dict] = {}
        for camera in ("front_left", "front", "front_right"):
            relative = f"intrinsics/{camera}_remap.npz"
            np.savez_compressed(
                b2 / relative,
                map_x=map_x,
                map_y=map_y,
                valid_mask=valid_mask,
            )
            remaps[camera] = relative
            transforms[camera] = {
                "rotation_camera_to_front": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            }
        save_yaml(
            b2 / "candidate.yaml",
            {
                "format": "DeepSharkFisheyeCalibrationCandidate",
                "schema_version": 1,
                "topology": "triple_front_panorama",
                "resolution": [1920, 1080],
                "experimental": True,
                "report_only": True,
                "recommended": False,
                "apply_allowed": False,
                "rig": {"complete": True, "transforms": transforms},
                "virtual_panorama": {
                    "canvas_size": [canvas_width, canvas_height],
                    "files": {"remaps": remaps},
                },
            },
        )
        save_yaml(
            b2 / "report.yaml",
            {
                "format": "DeepSharkFisheyeCalibrationCandidateReport",
                "schema_version": 1,
                "candidate": "valid-b2",
            },
        )

        fisheye = self.root / "runtime_candidates" / "fisheye"
        fisheye.mkdir()
        camera_intrinsics = {
            "status": "success",
            "model": "opencv_fisheye",
            "camera_matrix": [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]],
            "distortion_coefficients": [0.1, 0.01, 0.001, 0.0001],
            "resolution": [1920, 1080],
            "rms_px": 0.5,
            "accepted_input_count": 10,
        }
        save_yaml(
            fisheye / "candidate.yaml",
            {
                "format": "DeepSharkFisheyeCalibrationCandidate",
                "topology": "triple_front_panorama",
                "resolution": [1920, 1080],
                "experimental": False,
                "apply_allowed": True,
                "intrinsics": {
                    camera: dict(camera_intrinsics)
                    for camera in ("front_left", "front", "front_right")
                },
            },
        )
        save_yaml(fisheye / "report.yaml", {"candidate": "valid-fisheye"})

        far = self.root / "runtime_candidates" / "far"
        far.mkdir()
        save_yaml(
            far / "candidate.yaml",
            {
                "schema_version": 1,
                "candidate_type": "far_field_layout",
                "profile_id": "triple_front_panorama",
                "projection": {
                    "source": "b2_far_field_candidate",
                    "candidate_directory": str(b2.resolve()),
                },
                "camera_adjust": self._camera_adjust(),
                "output": {"width_px": 1800, "height_px": 600},
                "far_field_blend": {"mode": "b2_weight_selection"},
                "formal_profile_modified": False,
                "writes_calibration_yaml": False,
            },
        )
        save_yaml(far / "report.yaml", {"candidate_type": "far_field_layout"})

        near = self.root / "runtime_candidates" / "near"
        near.mkdir()
        save_yaml(
            near / "candidate.yaml",
            {
                "schema_version": 3,
                "candidate_type": "front_priority_layout",
                "profile_id": "triple_front_panorama",
                "projection": {
                    "source": "fisheye_rectilinear_candidate",
                    "intrinsics_source_path": str((fisheye / "candidate.yaml").resolve()),
                    "balance": 0.6,
                    "fov_scale": 1.0,
                },
                "camera_adjust": self._camera_adjust(),
                "left_pair": {
                    "side_shift_px": 0,
                    "side_visible_fraction": 0.3,
                    "feather_width_px": 24,
                },
                "right_pair": {
                    "side_shift_px": 0,
                    "side_visible_fraction": 0.3,
                    "feather_width_px": 24,
                },
                "output": {"width_px": 1800, "height_px": 700},
                "vertical_safety": {
                    "enabled": True,
                    "vertical_safe_ratio": 0.9,
                    "side_vertical_fade_px": 16,
                },
                "formal_profile_modified": False,
                "writes_calibration_yaml": False,
            },
        )
        save_yaml(near / "report.yaml", {"candidate_type": "front_priority_layout"})
        return {
            "far": far / "candidate.yaml",
            "near": near / "candidate.yaml",
            "b2": b2 / "candidate.yaml",
            "fisheye": fisheye / "candidate.yaml",
        }

    @staticmethod
    def _camera_adjust() -> dict[str, dict[str, float | int]]:
        return {
            camera: {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0}
            for camera in ("front_left", "front", "front_right")
        }


if __name__ == "__main__":
    unittest.main()
