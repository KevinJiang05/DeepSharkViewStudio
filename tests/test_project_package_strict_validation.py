from __future__ import annotations

from hashlib import sha256
import io
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from deep_shark_studio.calibration_candidate import load_calibration_candidate
from deep_shark_studio.config import load_yaml, save_yaml
from deep_shark_studio.project_package import validate_project_package
from deep_shark_studio.projection.intrinsics_runtime_loader import (
    load_fisheye_intrinsics_source,
)
from deep_shark_studio.seam.far_field_layout_candidate_runtime import (
    load_far_field_layout_candidate,
)
from deep_shark_studio.seam.layout_candidate_runtime import (
    load_layout_candidate_for_runtime,
)


CAMERAS = ("front_left", "front", "front_right")


class ProjectPackageStrictValidationTests(unittest.TestCase):
    """Security and portability contract for a directory project package."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.temp_root = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_complete_package_validates_after_copy_to_unrelated_directory(self) -> None:
        source = self._make_complete_package("source")
        moved = self.temp_root / "unrelated" / "nested" / "portable_project"
        moved.parent.mkdir(parents=True)
        shutil.copytree(source, moved)

        result = validate_project_package(moved, write_report=False)

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(moved.resolve(), result.package_root.resolve())

    def test_complete_fixture_satisfies_all_runtime_candidate_loaders(self) -> None:
        package = self._make_complete_package("loader_contract")
        calibration = package / "configs" / "calibration.yaml"

        far = load_far_field_layout_candidate(
            package / "candidates" / "far_field_layout" / "candidate.yaml",
            formal_calibration_path=calibration,
        )
        near = load_layout_candidate_for_runtime(
            package / "candidates" / "near_field_layout" / "candidate.yaml",
            formal_calibration_path=calibration,
        )
        b2, _report = load_calibration_candidate(
            package / "candidates" / "b2_candidate"
        )
        fisheye = load_fisheye_intrinsics_source(
            package / "candidates" / "fisheye_intrinsics" / "candidate.yaml"
        )

        self.assertEqual("b2_far_field_candidate", far.projection_source)
        self.assertEqual("fisheye_rectilinear_candidate", near.projection.source.value)
        self.assertEqual([1920, 1080], b2["resolution"])
        self.assertEqual(set(CAMERAS), set(fisheye.cameras))

    def test_validate_is_read_only_by_default_for_valid_and_invalid_packages(self) -> None:
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                package = self._make_complete_package(f"readonly_{invalid}")
                if invalid:
                    (package / "configs" / "network.yaml").write_text(
                        "network: [unterminated",
                        encoding="utf-8",
                    )
                before = self._tree_digest(package)

                result = validate_project_package(package)

                self.assertEqual(before, self._tree_digest(package))
                self.assertFalse((package / "project_validation_report.yaml").exists())
                self.assertIsNone(result.report_path)

    def test_malformed_yaml_is_an_error_and_never_escapes_validation(self) -> None:
        cases = (
            ("manifest", "project.dcsvs.yaml"),
            ("calibration config", "configs/calibration.yaml"),
            ("far candidate", "candidates/far_field_layout/candidate.yaml"),
            ("unreferenced package yaml", "notes/broken.yaml"),
        )
        for label, relative in cases:
            with self.subTest(label=label):
                package = self._make_complete_package(f"malformed_{label}")
                target = package / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("broken: [unterminated", encoding="utf-8")

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid)
                self.assertTrue(
                    any("yaml" in error.lower() for error in result.errors),
                    result.errors,
                )

    def test_yaml_roots_and_manifest_sections_must_have_expected_types(self) -> None:
        mutations = {
            "manifest root list": lambda package: save_yaml(
                package / "project.dcsvs.yaml",  # type: ignore[arg-type]
                ["not", "a", "mapping"],  # type: ignore[arg-type]
            ),
            "calibration root list": lambda package: save_yaml(
                package / "configs" / "calibration.yaml",  # type: ignore[arg-type]
                ["not", "a", "mapping"],  # type: ignore[arg-type]
            ),
            "configs section list": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest.__setitem__("configs", ["bad"]),
            ),
            "candidate entry list": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["candidates"].__setitem__(
                    "far_field_layout", ["bad"]
                ),
            ),
            "manifest candidate type mismatch": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["candidates"][
                    "far_field_layout"
                ].__setitem__("candidate_type", "not_far_field_layout"),
            ),
            "candidate yaml root list": lambda package: save_yaml(
                package / "candidates" / "near_field_layout" / "candidate.yaml",  # type: ignore[arg-type]
                ["not", "a", "mapping"],  # type: ignore[arg-type]
            ),
            "runtime defaults list": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest.__setitem__("runtime_defaults", ["bad"]),
            ),
            "paths flag string": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest.__setitem__(
                    "paths", {"use_relative_paths": "true"}
                ),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                package = self._make_complete_package(f"types_{label}")
                mutate(package)

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid, (label, result.errors))

    def test_configs_must_satisfy_the_minimum_runtime_schema(self) -> None:
        cases = {
            "calibration topology missing": lambda package: self._mutate_yaml(
                package / "configs" / "calibration.yaml",
                lambda config: config.pop("stitch_topology"),
            ),
            "selected topology absent": lambda package: self._mutate_yaml(
                package / "configs" / "calibration.yaml",
                lambda config: config.__setitem__("stitch_topology", "missing"),
            ),
            "calibration canvas nonpositive": lambda package: self._mutate_yaml(
                package / "configs" / "calibration.yaml",
                lambda config: config["topologies"]["triple_front_panorama"][
                    "canvas"
                ].__setitem__("width", 0),
            ),
            "global calibration canvas nonpositive": lambda package: self._mutate_yaml(
                package / "configs" / "calibration.yaml",
                lambda config: config["canvas"].__setitem__("height", 0),
            ),
            "active cameras wrong type": lambda package: self._mutate_yaml(
                package / "configs" / "calibration.yaml",
                lambda config: config["topologies"]["triple_front_panorama"].__setitem__(
                    "active_cameras", "front"
                ),
            ),
            "camera count wrong type": lambda package: self._mutate_yaml(
                package / "configs" / "cameras.yaml",
                lambda config: config.__setitem__("active_camera_count", "3"),
            ),
            "camera order wrong type": lambda package: self._mutate_yaml(
                package / "configs" / "cameras.yaml",
                lambda config: config.__setitem__("camera_order", "front"),
            ),
            "cameras section wrong type": lambda package: self._mutate_yaml(
                package / "configs" / "cameras.yaml",
                lambda config: config.__setitem__("cameras", []),
            ),
            "performance section wrong type": lambda package: self._mutate_yaml(
                package / "configs" / "cameras.yaml",
                lambda config: config.__setitem__("performance", []),
            ),
            "performance value is not numeric": lambda package: self._mutate_yaml(
                package / "configs" / "cameras.yaml",
                lambda config: config["performance"].__setitem__(
                    "process_fps", "fast"
                ),
            ),
            "network section wrong type": lambda package: self._mutate_yaml(
                package / "configs" / "network.yaml",
                lambda config: config.__setitem__("output", []),
            ),
            "config path is directory": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["configs"].__setitem__(
                    "network", "configs"
                ),
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                package = self._make_complete_package(f"config_schema_{label}")
                mutate(package)

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid, (label, result.errors))

    def test_absolute_parent_and_internal_dependency_escapes_are_rejected(self) -> None:
        outside_yaml = self.temp_root / "outside.yaml"
        save_yaml(outside_yaml, {"outside": True})
        outside_npz = self.temp_root / "outside.npz"
        self._write_valid_remap(outside_npz)
        cases = {
            "absolute manifest config": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["configs"].__setitem__(
                    "network", str(outside_yaml.resolve())
                ),
            ),
            "parent manifest config": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["configs"].__setitem__(
                    "network", "../outside.yaml"
                ),
            ),
            "b2 remap escape": lambda package: self._mutate_b2(
                package,
                lambda candidate: candidate["virtual_panorama"]["files"][
                    "remaps"
                ].__setitem__("front_left", "../../../outside.npz"),
            ),
            "far projection escape": lambda package: self._mutate_yaml(
                package / "candidates" / "far_field_layout" / "candidate.yaml",
                lambda candidate: candidate["projection"].__setitem__(
                    "candidate_directory", "../../../outside"
                ),
            ),
            "near intrinsics escape": lambda package: self._mutate_yaml(
                package / "candidates" / "near_field_layout" / "candidate.yaml",
                lambda candidate: candidate["projection"].__setitem__(
                    "intrinsics_source_path", "../../../outside.yaml"
                ),
            ),
            "runtime default escape": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["runtime_defaults"].__setitem__(
                    "far_field_layout_candidate", "../outside.yaml"
                ),
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                package = self._make_complete_package(f"escape_{label}")
                mutate(package)

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid, (label, result.errors))
                self.assertTrue(
                    any(
                        marker in error.lower()
                        for error in result.errors
                        for marker in ("absolute", "escape", "outside", "relative")
                    ),
                    result.errors,
                )

    def test_symlink_to_outside_package_is_rejected(self) -> None:
        package = self._make_complete_package("symlink_escape")
        outside = self.temp_root / "outside_network.yaml"
        save_yaml(outside, {"network": {}})
        link = package / "configs" / "linked_network.yaml"
        try:
            os.symlink(outside, link)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"Symlink creation is unavailable: {exc}")
        self._mutate_manifest(
            package,
            lambda manifest: manifest["configs"].__setitem__(
                "network", "configs/linked_network.yaml"
            ),
        )

        result = self._validate_without_exception(package)

        self.assertFalse(result.valid, result.errors)

    def test_b2_remap_rejects_forged_and_truncated_npz(self) -> None:
        for label, payload_factory in (
            ("forged", lambda valid: b"this is not an npz archive"),
            ("truncated", lambda valid: valid[: max(1, len(valid) // 3)]),
        ):
            with self.subTest(label=label):
                package = self._make_complete_package(f"npz_{label}")
                remap = self._remap_path(package, "front_left")
                valid_bytes = remap.read_bytes()
                remap.write_bytes(payload_factory(valid_bytes))

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid, result.errors)
                self.assertTrue(
                    any("npz" in error.lower() or "remap" in error.lower() for error in result.errors),
                    result.errors,
                )

    def test_b2_remap_requires_exact_keys_shapes_and_dtypes(self) -> None:
        shape = (3, 4)
        cases = {
            "missing valid_mask": {
                "map_x": np.zeros(shape, dtype=np.float32),
                "map_y": np.zeros(shape, dtype=np.float32),
            },
            "map shape mismatch": {
                "map_x": np.zeros(shape, dtype=np.float32),
                "map_y": np.zeros((2, 4), dtype=np.float32),
                "valid_mask": np.ones(shape, dtype=np.uint8),
            },
            "common shape differs from canvas": {
                "map_x": np.zeros((2, 4), dtype=np.float32),
                "map_y": np.zeros((2, 4), dtype=np.float32),
                "valid_mask": np.ones((2, 4), dtype=np.uint8),
            },
            "map dtype mismatch": {
                "map_x": np.zeros(shape, dtype=np.float64),
                "map_y": np.zeros(shape, dtype=np.float32),
                "valid_mask": np.ones(shape, dtype=np.uint8),
            },
            "mask dtype mismatch": {
                "map_x": np.zeros(shape, dtype=np.float32),
                "map_y": np.zeros(shape, dtype=np.float32),
                "valid_mask": np.ones(shape, dtype=np.float32),
            },
            "nonfinite map": {
                "map_x": np.full(shape, np.nan, dtype=np.float32),
                "map_y": np.zeros(shape, dtype=np.float32),
                "valid_mask": np.ones(shape, dtype=np.uint8),
            },
        }
        for label, arrays in cases.items():
            with self.subTest(label=label):
                package = self._make_complete_package(f"npz_contract_{label}")
                np.savez_compressed(self._remap_path(package, "front_left"), **arrays)

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid, (label, result.errors))
                self.assertTrue(
                    any(
                        marker in error.lower()
                        for error in result.errors
                        for marker in ("key", "shape", "dtype", "remap", "npz")
                    ),
                    result.errors,
                )

    def test_b2_remap_rejects_huge_npy_shape_before_array_allocation(self) -> None:
        package = self._make_complete_package("npz_huge_header")
        remap = self._remap_path(package, "front_left")

        def npy_header(dtype: str, shape: tuple[int, int]) -> bytes:
            stream = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                stream,
                {
                    "descr": dtype,
                    "fortran_order": False,
                    "shape": shape,
                },
            )
            return stream.getvalue()

        with zipfile.ZipFile(remap, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "map_x.npy",
                npy_header("<f4", (1_000_000, 1_000_000)),
            )
            archive.writestr("map_y.npy", npy_header("<f4", (3, 4)))
            archive.writestr("valid_mask.npy", npy_header("|u1", (3, 4)))

        result = self._validate_without_exception(package)

        self.assertFalse(result.valid, result.errors)
        self.assertTrue(
            any("header shape" in error.lower() for error in result.errors),
            result.errors,
        )

    def test_b2_candidate_requires_complete_three_camera_runtime_contract(self) -> None:
        cases = {
            "wrong manifest source type": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["candidates"]["b2_candidate"].__setitem__(
                    "source_type", "not_b2"
                ),
            ),
            "wrong candidate format": lambda package: self._mutate_b2(
                package,
                lambda candidate: candidate.__setitem__("format", "forged"),
            ),
            "rig incomplete": lambda package: self._mutate_b2(
                package,
                lambda candidate: candidate["rig"].__setitem__("complete", False),
            ),
            "missing transform": lambda package: self._mutate_b2(
                package,
                lambda candidate: candidate["rig"]["transforms"].pop("front_right"),
            ),
            "invalid transform shape": lambda package: self._mutate_b2(
                package,
                lambda candidate: candidate["rig"]["transforms"]["front"].__setitem__(
                    "rotation_camera_to_front", [[1.0, 0.0], [0.0, 1.0]]
                ),
            ),
            "missing remap declaration": lambda package: self._mutate_b2(
                package,
                lambda candidate: candidate["virtual_panorama"]["files"][
                    "remaps"
                ].pop("front_right"),
            ),
            "far custom missing b2 manifest dependency": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["candidates"].pop("b2_candidate"),
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                package = self._make_complete_package(f"b2_{label}")
                mutate(package)

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid, (label, result.errors))

    def test_packaged_far_layout_cannot_defer_a_legacy_projection_bypass(self) -> None:
        package = self._make_complete_package("legacy_far_bypass")
        self._mutate_yaml(
            package / "candidates" / "far_field_layout" / "candidate.yaml",
            lambda candidate: (
                candidate["projection"].clear(),
                candidate["projection"].__setitem__(
                    "source", "current_perspective"
                ),
                candidate["far_field_blend"].__setitem__(
                    "mode", "current_horizontal_feather"
                ),
            ),
        )
        self._mutate_manifest(
            package,
            lambda manifest: manifest["runtime_defaults"].__setitem__(
                "use_far_field_custom", False
            ),
        )

        result = self._validate_without_exception(package)

        self.assertFalse(result.valid, result.errors)
        self.assertTrue(
            any("b-2" in error.lower() for error in result.errors),
            result.errors,
        )

    def test_fisheye_intrinsics_requires_complete_numeric_runtime_contract(self) -> None:
        candidate_path = (
            Path("candidates") / "fisheye_intrinsics" / "candidate.yaml"
        )
        cases = {
            "wrong manifest source type": lambda package: self._mutate_manifest(
                package,
                lambda manifest: manifest["candidates"]["fisheye_intrinsics"].__setitem__(
                    "source_type", "not_fisheye"
                ),
            ),
            "missing runtime camera": lambda package: self._mutate_yaml(
                package / candidate_path,
                lambda candidate: candidate["intrinsics"].pop("front_right"),
            ),
            "missing camera matrix": lambda package: self._mutate_yaml(
                package / candidate_path,
                lambda candidate: candidate["intrinsics"]["front"].pop(
                    "camera_matrix"
                ),
            ),
            "nonnumeric camera matrix": lambda package: self._mutate_yaml(
                package / candidate_path,
                lambda candidate: candidate["intrinsics"]["front"].__setitem__(
                    "camera_matrix", [["bad", 0, 0], [0, 1, 0], [0, 0, 1]]
                ),
            ),
            "invalid resolution": lambda package: self._mutate_yaml(
                package / candidate_path,
                lambda candidate: candidate["intrinsics"]["front"].__setitem__(
                    "resolution", [0, -1]
                ),
            ),
            "nonfinite distortion": lambda package: self._mutate_yaml(
                package / candidate_path,
                lambda candidate: candidate["intrinsics"]["front"].__setitem__(
                    "distortion_coefficients", [0.1, float("nan"), 0.0, 0.0]
                ),
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                package = self._make_complete_package(f"fisheye_{label}")
                mutate(package)

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid, (label, result.errors))

    def test_runtime_defaults_are_typed_and_reference_declared_candidates(self) -> None:
        cases = {
            "mode type": ("default_stitch_mode", 1),
            "unknown mode": ("default_stitch_mode", "not_a_mode"),
            "custom flag type": ("use_far_field_custom", "true"),
            "projection source type": ("near_field_projection_source", 1),
            "unknown projection source": (
                "near_field_projection_source",
                "not_a_projection",
            ),
            "candidate reference type": ("far_field_layout_candidate", ["bad"]),
            "candidate reference mismatch": (
                "far_field_layout_candidate",
                "candidates/near_field_layout/candidate.yaml",
            ),
        }
        for label, (key, value) in cases.items():
            with self.subTest(label=label):
                package = self._make_complete_package(f"defaults_{label}")
                self._mutate_manifest(
                    package,
                    lambda manifest, key=key, value=value: manifest[
                        "runtime_defaults"
                    ].__setitem__(key, value),
                )

                result = self._validate_without_exception(package)

                self.assertFalse(result.valid, (label, result.errors))

    def _make_complete_package(self, name: str) -> Path:
        package = self.temp_root / name.replace(" ", "_")
        package.mkdir(parents=True)
        config_dir = package / "configs"
        config_dir.mkdir()
        save_yaml(
            config_dir / "calibration.yaml",
            {
                "stitch_topology": "triple_front_panorama",
                "canvas": {"width": 4, "height": 3},
                "topologies": {
                    "triple_front_panorama": {
                        "active_cameras": list(CAMERAS),
                        "canvas": {"width": 4, "height": 3},
                        "stitch_points": {
                            "front_left_front": [[1.0, 0.0], [1.0, 3.0]],
                            "front_front_right": [[3.0, 0.0], [3.0, 3.0]],
                        },
                        "overlaps": [
                            {
                                "name": "front_left_front",
                                "cameras": ["front_left", "front"],
                                "x_range": [0.0, 2.0],
                                "seam": "front_left_front",
                            },
                            {
                                "name": "front_front_right",
                                "cameras": ["front", "front_right"],
                                "x_range": [2.0, 4.0],
                                "seam": "front_front_right",
                            },
                        ],
                    }
                },
            },
        )
        save_yaml(
            config_dir / "cameras.yaml",
            {
                "active_camera_count": 3,
                "camera_order": list(CAMERAS),
                "cameras": {
                    camera: {
                        "display_name": camera,
                        "enabled": True,
                        "source_type": "image_dir",
                        "source": "",
                    }
                    for camera in CAMERAS
                },
                "performance": {"process_fps": 15, "preview_fps": 15},
            },
        )
        save_yaml(
            config_dir / "network.yaml",
            {
                "output": {"enabled": False},
                "udp_legacy": {"enabled": False},
            },
        )

        b2_root = package / "candidates" / "b2_candidate"
        (b2_root / "intrinsics").mkdir(parents=True)
        remaps: dict[str, str] = {}
        for camera in CAMERAS:
            relative = f"intrinsics/{camera}_remap.npz"
            self._write_valid_remap(b2_root / relative)
            remaps[camera] = relative
        save_yaml(
            b2_root / "candidate.yaml",
            {
                "format": "DeepSharkFisheyeCalibrationCandidate",
                "topology": "triple_front_panorama",
                "resolution": [1920, 1080],
                "experimental": True,
                "apply_allowed": False,
                "rig": {
                    "complete": True,
                    "transforms": {
                        camera: {
                            "rotation_camera_to_front": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ]
                        }
                        for camera in CAMERAS
                    },
                },
                "virtual_panorama": {
                    "canvas_size": [4, 3],
                    "files": {"remaps": remaps},
                },
            },
        )
        save_yaml(
            b2_root / "report.yaml",
            {
                "format": "DeepSharkFisheyeCalibrationCandidateReport",
                "schema_version": 1,
            },
        )

        far_root = package / "candidates" / "far_field_layout"
        far_root.mkdir(parents=True)
        save_yaml(
            far_root / "candidate.yaml",
            {
                "schema_version": 1,
                "candidate_type": "far_field_layout",
                "profile_id": "triple_front_panorama",
                "projection": {
                    "source": "b2_far_field_candidate",
                    "candidate_directory": "../b2_candidate",
                },
                "camera_adjust": self._camera_adjust(),
                "output": {"width_px": 4, "height_px": 3},
                "far_field_blend": {"mode": "b2_weight_selection"},
                "formal_profile_modified": False,
                "writes_calibration_yaml": False,
            },
        )

        fisheye_root = package / "candidates" / "fisheye_intrinsics"
        fisheye_root.mkdir(parents=True)
        camera_intrinsics = {
            "status": "success",
            "model": "opencv_fisheye",
            "camera_matrix": [
                [2.0, 0.0, 2.0],
                [0.0, 2.0, 1.5],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [0.1, 0.01, 0.001, 0.0001],
            "resolution": [4, 3],
            "rms_px": 0.5,
            "accepted_input_count": 10,
        }
        save_yaml(
            fisheye_root / "candidate.yaml",
            {
                "format": "DeepSharkFisheyeCalibrationCandidate",
                "topology": "triple_front_panorama",
                "intrinsics": {
                    camera: dict(camera_intrinsics) for camera in CAMERAS
                },
            },
        )

        near_root = package / "candidates" / "near_field_layout"
        near_root.mkdir(parents=True)
        save_yaml(
            near_root / "candidate.yaml",
            {
                "schema_version": 3,
                "candidate_type": "front_priority_layout",
                "profile_id": "triple_front_panorama",
                "projection": {
                    "source": "fisheye_rectilinear_candidate",
                    "intrinsics_source_path": "../fisheye_intrinsics/candidate.yaml",
                    "balance": 0.6,
                    "fov_scale": 1.0,
                },
                "camera_adjust": self._camera_adjust(),
                "left_pair": {
                    "side_shift_px": 0,
                    "side_visible_fraction": 0.3,
                    "feather_width_px": 2,
                },
                "right_pair": {
                    "side_shift_px": 0,
                    "side_visible_fraction": 0.3,
                    "feather_width_px": 2,
                },
                "output": {"width_px": 4, "height_px": 3},
                "vertical_safety": {
                    "enabled": True,
                    "vertical_safe_ratio": 0.9,
                    "side_vertical_fade_px": 1,
                },
                "formal_profile_modified": False,
                "writes_calibration_yaml": False,
            },
        )

        save_yaml(
            package / "project.dcsvs.yaml",
            {
                "schema_version": 1,
                "package_type": "deep_shark_project",
                "project": {
                    "name": name,
                    "created_at": "2026-07-10T12:00:00+08:00",
                    "exported_by": "DeepSharkViewStudio",
                },
                "configs": {
                    "calibration": "configs/calibration.yaml",
                    "cameras": "configs/cameras.yaml",
                    "network": "configs/network.yaml",
                },
                "runtime_defaults": {
                    "default_stitch_mode": "far_field",
                    "use_far_field_custom": True,
                    "far_field_layout_candidate": "candidates/far_field_layout/candidate.yaml",
                    "near_field_layout_candidate": "candidates/near_field_layout/candidate.yaml",
                    "near_field_projection_source": "fisheye_rectilinear_candidate",
                    "fisheye_intrinsics_source": "candidates/fisheye_intrinsics/candidate.yaml",
                },
                "candidates": {
                    "far_field_layout": {
                        "path": "candidates/far_field_layout/candidate.yaml",
                        "candidate_type": "far_field_layout",
                    },
                    "near_field_layout": {
                        "path": "candidates/near_field_layout/candidate.yaml",
                        "candidate_type": "front_priority_layout",
                    },
                    "b2_candidate": {
                        "path": "candidates/b2_candidate/candidate.yaml",
                        "source_type": "b2_calibration_candidate",
                    },
                    "fisheye_intrinsics": {
                        "path": "candidates/fisheye_intrinsics/candidate.yaml",
                        "source_type": "opencv_fisheye_intrinsics",
                    },
                },
                "paths": {"use_relative_paths": True},
            },
        )
        return package

    def _validate_without_exception(self, package: Path):
        try:
            return validate_project_package(package, write_report=False)
        except Exception as exc:  # pragma: no cover - assertion reports old unsafe behavior.
            self.fail(f"Validation raised {type(exc).__name__}: {exc}")

    @staticmethod
    def _write_valid_remap(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = (3, 4)
        np.savez_compressed(
            path,
            map_x=np.zeros(shape, dtype=np.float32),
            map_y=np.zeros(shape, dtype=np.float32),
            valid_mask=np.ones(shape, dtype=np.uint8),
        )

    @staticmethod
    def _camera_adjust() -> dict[str, dict[str, float]]:
        return {
            camera: {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0}
            for camera in CAMERAS
        }

    @staticmethod
    def _remap_path(package: Path, camera: str) -> Path:
        return (
            package
            / "candidates"
            / "b2_candidate"
            / "intrinsics"
            / f"{camera}_remap.npz"
        )

    @classmethod
    def _mutate_manifest(cls, package: Path, mutation) -> None:
        cls._mutate_yaml(package / "project.dcsvs.yaml", mutation)

    @classmethod
    def _mutate_b2(cls, package: Path, mutation) -> None:
        cls._mutate_yaml(
            package / "candidates" / "b2_candidate" / "candidate.yaml",
            mutation,
        )

    @staticmethod
    def _mutate_yaml(path: Path, mutation) -> None:
        data = load_yaml(path)
        mutation(data)
        save_yaml(path, data)

    @staticmethod
    def _tree_digest(root: Path) -> str:
        digest = sha256()
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            if path.is_symlink():
                payload = os.readlink(path).encode("utf-8")
                digest.update(b"L")
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
            elif path.is_file():
                payload = path.read_bytes()
                digest.update(b"F")
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
            else:
                digest.update(b"D")
        return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
