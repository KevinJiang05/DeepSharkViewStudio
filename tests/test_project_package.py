from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from deep_shark_studio.config import file_revision, load_yaml, save_yaml
from deep_shark_studio.project_package import (
    activate_project_package,
    export_project_package,
    find_absolute_path_strings,
    resolve_project_path,
    validate_project_package,
)
from deep_shark_studio.project_package.path_resolver import ProjectPathError


class ProjectPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "configs"
        self.config_dir.mkdir()
        save_yaml(
            self.config_dir / "calibration.yaml",
            {
                "stitch_topology": "triple_front_panorama",
                "canvas": {"width": 2200, "height": 700},
            },
        )
        save_yaml(self.config_dir / "cameras.yaml", {"cameras": {}, "performance": {}})
        save_yaml(self.config_dir / "network.yaml", {"network": {}})
        self.before_calibration_hash = file_revision(self.config_dir / "calibration.yaml")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_package_uses_relative_manifest_and_copies_dependencies(self) -> None:
        far = self._make_far_field_candidate()
        near = self._make_near_field_candidate()
        b2 = self._make_b2_candidate()
        fisheye = self._make_fisheye_intrinsics_candidate()

        result = export_project_package(
            self.temp_dir / "exports",
            project_name="portable_demo",
            calibration_config_path=self.config_dir / "calibration.yaml",
            cameras_config_path=self.config_dir / "cameras.yaml",
            network_config_path=self.config_dir / "network.yaml",
            far_field_candidate_path=far,
            near_field_candidate_path=near,
            b2_candidate_path=b2,
            fisheye_intrinsics_path=fisheye,
            use_far_field_custom=True,
            near_field_projection_source="fisheye_rectilinear_candidate",
        )

        manifest = load_yaml(result.manifest_path)
        self.assertFalse(find_absolute_path_strings(manifest))
        self.assertEqual("configs/calibration.yaml", manifest["configs"]["calibration"])
        self.assertTrue((result.package_root / "configs" / "cameras.yaml").exists())
        self.assertTrue((result.package_root / "candidates" / "far_field_layout" / "candidate.yaml").exists())
        self.assertTrue((result.package_root / "candidates" / "near_field_layout" / "candidate.yaml").exists())
        self.assertTrue((result.package_root / "candidates" / "b2_candidate" / "intrinsics" / "front_left_remap.npz").exists())
        self.assertTrue((result.package_root / "candidates" / "fisheye_intrinsics" / "candidate.yaml").exists())

        far_copy = load_yaml(result.package_root / "candidates" / "far_field_layout" / "candidate.yaml")
        self.assertEqual("../b2_candidate", far_copy["projection"]["candidate_directory"])
        near_copy = load_yaml(result.package_root / "candidates" / "near_field_layout" / "candidate.yaml")
        self.assertEqual(
            "../fisheye_intrinsics/candidate.yaml",
            near_copy["projection"]["intrinsics_source_path"],
        )
        validation = validate_project_package(result.manifest_path)
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(self.before_calibration_hash, file_revision(self.config_dir / "calibration.yaml"))

    def test_package_can_move_to_another_directory_and_validate(self) -> None:
        result = export_project_package(
            self.temp_dir / "exports",
            project_name="move_demo",
            calibration_config_path=self.config_dir / "calibration.yaml",
            cameras_config_path=self.config_dir / "cameras.yaml",
            network_config_path=self.config_dir / "network.yaml",
            b2_candidate_path=self._make_b2_candidate(),
        )
        moved_root = self.temp_dir / "moved" / result.package_root.name
        moved_root.parent.mkdir(parents=True)
        shutil.copytree(result.package_root, moved_root)

        validation = validate_project_package(moved_root)

        self.assertTrue(validation.valid, validation.errors)

    def test_validation_finds_missing_file_and_absolute_manifest_path(self) -> None:
        result = export_project_package(
            self.temp_dir / "exports",
            project_name="invalid_demo",
            calibration_config_path=self.config_dir / "calibration.yaml",
            cameras_config_path=self.config_dir / "cameras.yaml",
            network_config_path=self.config_dir / "network.yaml",
        )
        (result.package_root / "configs" / "network.yaml").unlink()
        validation = validate_project_package(result.manifest_path)
        self.assertFalse(validation.valid)
        self.assertTrue(any("network" in error for error in validation.errors))

        manifest = load_yaml(result.manifest_path)
        manifest["configs"]["network"] = "D:/external/network.yaml"
        save_yaml(result.manifest_path, manifest)
        validation = validate_project_package(result.manifest_path)
        self.assertFalse(validation.valid)
        self.assertTrue(any("absolute" in error.lower() for error in validation.errors))

    def test_path_resolver_blocks_escape_and_absolute_paths(self) -> None:
        with self.assertRaises(ProjectPathError):
            resolve_project_path(self.temp_dir, "../outside.yaml")
        with self.assertRaises(ProjectPathError):
            resolve_project_path(self.temp_dir, "D:/outside.yaml")

    def test_activate_project_package_requires_explicit_call_and_backs_up(self) -> None:
        result = export_project_package(
            self.temp_dir / "exports",
            project_name="activate_demo",
            calibration_config_path=self.config_dir / "calibration.yaml",
            cameras_config_path=self.config_dir / "cameras.yaml",
            network_config_path=self.config_dir / "network.yaml",
        )
        active_dir = self.temp_dir / "active_configs"
        active_dir.mkdir()
        save_yaml(active_dir / "calibration.yaml", {"before": True})
        save_yaml(active_dir / "cameras.yaml", {"before": True})
        save_yaml(active_dir / "network.yaml", {"before": True})

        self.assertEqual({"before": True}, load_yaml(active_dir / "calibration.yaml"))
        activation = activate_project_package(
            result.manifest_path,
            active_config_dir=active_dir,
            backup_root=self.temp_dir / "backups",
        )

        self.assertTrue(activation.backup_path and activation.backup_path.exists())
        self.assertEqual(
            "triple_front_panorama",
            load_yaml(active_dir / "calibration.yaml")["stitch_topology"],
        )

    def _make_far_field_candidate(self) -> Path:
        root = self.temp_dir / "far_candidate"
        root.mkdir()
        save_yaml(
            root / "candidate.yaml",
            {
                "schema_version": 1,
                "candidate_type": "far_field_layout",
                "profile_id": "triple_front_panorama",
                "projection": {
                    "source": "b2_far_field_candidate",
                    "candidate_directory": str((self.temp_dir / "b2_candidate").resolve()),
                },
                "camera_adjust": self._camera_adjust(),
                "output": {"width_px": 2200, "height_px": 700},
                "far_field_blend": {"mode": "b2_weight_selection"},
                "formal_profile_modified": False,
                "writes_calibration_yaml": False,
            },
        )
        save_yaml(root / "report.yaml", {"candidate_type": "far_field_layout"})
        return root / "candidate.yaml"

    def _make_near_field_candidate(self) -> Path:
        root = self.temp_dir / "near_candidate"
        root.mkdir()
        save_yaml(
            root / "candidate.yaml",
            {
                "schema_version": 3,
                "candidate_type": "front_priority_layout",
                "profile_id": "triple_front_panorama",
                "projection": {
                    "source": "fisheye_rectilinear_candidate",
                    "intrinsics_source_path": str((self.temp_dir / "fisheye_candidate" / "candidate.yaml").resolve()),
                    "balance": 0.6,
                    "fov_scale": 1.0,
                },
                "camera_adjust": self._camera_adjust(),
                "left_pair": {"side_shift_px": 0, "side_visible_fraction": 0.3, "feather_width_px": 24},
                "right_pair": {"side_shift_px": 0, "side_visible_fraction": 0.3, "feather_width_px": 24},
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
        save_yaml(root / "report.yaml", {"candidate_type": "front_priority_layout"})
        return root / "candidate.yaml"

    def _make_b2_candidate(self) -> Path:
        root = self.temp_dir / "b2_candidate"
        (root / "intrinsics").mkdir(parents=True, exist_ok=True)
        for camera in ("front_left", "front", "front_right"):
            (root / "intrinsics" / f"{camera}_remap.npz").write_bytes(b"fake")
        save_yaml(
            root / "candidate.yaml",
            {
                "format": "DeepSharkCalibrationCandidate",
                "resolution": [1280, 720],
                "rig": {
                    "complete": True,
                    "transforms": {
                        camera: {"rotation_camera_to_front": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
                        for camera in ("front_left", "front", "front_right")
                    },
                },
                "virtual_panorama": {
                    "canvas_size": [2200, 700],
                    "files": {
                        "remaps": {
                            camera: f"intrinsics/{camera}_remap.npz"
                            for camera in ("front_left", "front", "front_right")
                        }
                    },
                },
            },
        )
        save_yaml(root / "report.yaml", {"candidate": "b2"})
        return root / "candidate.yaml"

    def _make_fisheye_intrinsics_candidate(self) -> Path:
        root = self.temp_dir / "fisheye_candidate"
        root.mkdir(exist_ok=True)
        camera = {
            "status": "success",
            "model": "opencv_fisheye",
            "camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            "distortion_coefficients": [0.1, 0.01, 0.001, 0.0001],
            "resolution": [640, 480],
            "rms_px": 0.5,
            "accepted_input_count": 10,
        }
        save_yaml(
            root / "candidate.yaml",
            {
                "format": "DeepSharkFisheyeCalibrationCandidate",
                "intrinsics": {
                    "front_left": dict(camera),
                    "front": dict(camera),
                    "front_right": dict(camera),
                },
            },
        )
        save_yaml(root / "report.yaml", {"candidate": "fisheye"})
        return root / "candidate.yaml"

    @staticmethod
    def _camera_adjust() -> dict[str, dict[str, float]]:
        return {
            camera: {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0}
            for camera in ("front_left", "front", "front_right")
        }


if __name__ == "__main__":
    unittest.main()
