# Project Package Export / Import

DeepShark project packages are directory packages for moving a tuned project to
another machine without keeping local absolute paths such as `D:\Develop\...`.

## Package Layout

```text
DeepShark_Project_<name>_<timestamp>/
  project.dcsvs.yaml
  export_report.yaml
  project_validation_report.yaml
  README_使用说明.md
  configs/
    calibration.yaml
    cameras.yaml
    network.yaml
  candidates/
    far_field_layout/
      candidate.yaml
      report.yaml
      preview.png
    near_field_layout/
      candidate.yaml
      report.yaml
      preview.png
    b2_candidate/
      candidate.yaml
      report.yaml
      intrinsics/*_remap.npz
    fisheye_intrinsics/
      candidate.yaml
      report.yaml
```

Preview assets are optional. Debug assets are excluded by default.

## Manifest Schema

The package manifest is `project.dcsvs.yaml`.

```yaml
schema_version: 1
package_type: deep_shark_project
project:
  name: triple_front_panorama
  created_at: ...
  exported_by: DeepSharkViewStudio
configs:
  calibration: configs/calibration.yaml
  cameras: configs/cameras.yaml
  network: configs/network.yaml
runtime_defaults:
  default_stitch_mode: far_field
  use_far_field_custom: true
  far_field_layout_candidate: candidates/far_field_layout/candidate.yaml
  near_field_layout_candidate: candidates/near_field_layout/candidate.yaml
  near_field_projection_source: current_perspective
  fisheye_intrinsics_source: candidates/fisheye_intrinsics/candidate.yaml
candidates:
  far_field_layout:
    path: candidates/far_field_layout/candidate.yaml
    candidate_type: far_field_layout
  near_field_layout:
    path: candidates/near_field_layout/candidate.yaml
    candidate_type: front_priority_layout
  b2_candidate:
    path: candidates/b2_candidate/candidate.yaml
    source_type: b2_calibration_candidate
  fisheye_intrinsics:
    path: candidates/fisheye_intrinsics/candidate.yaml
    source_type: opencv_fisheye_intrinsics
paths:
  use_relative_paths: true
```

All manifest paths must be package-relative and use `/` separators. Import
validation rejects absolute paths and path escape such as `../outside.yaml`.

## Export Flow

`Export Project Package` copies the active `configs/*.yaml` files and currently
loaded candidates into a new package directory. The exporter rewrites known
candidate references:

- Far-field Layout Candidate `projection.candidate_directory` points to
  `../b2_candidate` inside the package.
- Near-field fisheye candidate `projection.intrinsics_source_path` points to
  `../fisheye_intrinsics/candidate.yaml` inside the package.

The exporter does not modify the original configs or original candidates. It
does not write `configs/calibration.yaml`.

## Import Flow

`Import Project Package` first validates the selected package. Activation is a
separate confirmation step. When activated, the current `configs/` files are
backed up before package configs are copied into the active config directory.

`Validate Project Package` only checks the package and writes
`project_validation_report.yaml`; it does not activate or overwrite anything.

## Validation

Validation checks:

- manifest schema and package type;
- required config files;
- candidate paths and candidate types;
- B-2 runtime remap dependencies;
- fisheye intrinsics source completeness through the runtime loader;
- absence of absolute paths in package YAML files;
- package-relative path safety.

## Candidate Types

- `far_field_layout`: Far-field Custom layout. It is B-2 per-camera projection
  plus manual camera adjust plus B-2 weight selection.
- `front_priority_layout`: Near-field Layout Candidate V2/V3.
- `b2_calibration_candidate`: B-2 calibration candidate used as a per-camera
  projection source.
- `opencv_fisheye_intrinsics`: experimental fisheye intrinsics source for
  Near-field Fisheye Rectilinear.

## Not Included

The exporter does not include raw videos, full calibration sessions, temporary
runtime reports, or large debug folders unless explicitly requested by the
caller. Runtime algorithms and formal calibration/profile schemas are not
changed by project package export/import.
