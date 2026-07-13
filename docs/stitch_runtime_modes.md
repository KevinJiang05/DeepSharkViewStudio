# Stitch Runtime Mode Architecture

This document records the runtime stitching boundaries for DeepSharkViewStudio.

## UI Workflow

The main workspaces are organized by professional workflow:

- Realtime Monitor
- Layout & Projection Lab
- Calibration & Candidates
- Project Management
- Diagnostics & Logs

See `docs/ui_workflow.md` for the UI-level responsibilities. Runtime stitching
rules below remain the source of truth for Far-field, Near-field, and projection
behavior.

The Realtime Monitor maps the underlying fields to five task-facing presets in
`deep_shark_studio/gui/runtime_workflow.py`: Far Default, B-2 View, Far Custom,
Near Current, and Near Fisheye. This view-model is the only GUI selection
source; it does not alter the processing contracts below.

## PreviewLayoutMode

`PreviewLayoutMode` is the UI display layout state in `deep_shark_studio/gui/main_window.py`.

- `Grid`: show multiple raw camera views.
- `Focus`: show one enlarged raw camera view.
- `Stitched`: show the stitched canvas.

It only decides what the user sees in the preview area. It does not choose the stitching algorithm.

## StitchRuntimeMode

`StitchRuntimeMode` is the stitching algorithm mode in `deep_shark_studio/stitch_runtime_modes.py`.

- `Far-field`: keeps the existing formal runtime path by default, using `SurroundStitcher.process()` and the formal `compose_horizontal_feather()` path. This remains the formal fallback. An optional Far-field Custom Layout candidate uses the B-2 candidate per-camera projection as input, then adds post-warp `x/y/scale` camera adjustment before Far-field-style horizontal feather composition.
- `Near-field`: uses a `ProjectionProvider` to produce per-camera `warped_images` and `valid_masks`, then applies a read-only Layout Candidate with the front-priority compositor.
- `Auto`: placeholder only. It does not perform automatic conflict detection or switching.

`PreviewLayoutMode` and `StitchRuntimeMode` must stay separate.

## ProjectionProvider Layer

The ProjectionProvider layer isolates how each camera is projected or warped into a shared canvas.

The common result object is:

```python
ProjectionResult:
    warped_images: dict[str, np.ndarray]
    valid_masks: dict[str, np.ndarray]
    metadata: dict
    timings: dict
```

Camera keys are `front_left`, `front`, and `front_right`.

Far-field Default currently does not use ProjectionProvider:

```text
frames
  -> SurroundStitcher.process()
  -> SurroundStitcher.warp_all()
  -> compose_horizontal_feather()
  -> canvas
```

Near-field uses ProjectionProvider:

```text
frames
  -> ProjectionProvider.project()
  -> ProjectionResult(warped_images, valid_masks, metadata, timings)
  -> Layout Candidate
  -> front-priority compositor
  -> canvas
```

Near-field must not reuse the Far-field final blended canvas. That canvas has already applied horizontal feather blending and loses the independent per-camera pixel sources needed by the Near-field compositor.

## Far-field Custom Layout

Far-field Custom Layout is an optional candidate path for manually tuning far-scene alignment without changing the formal profile.

Far-field Default remains:

```text
frames
  -> SurroundStitcher.process()
  -> current horizontal feather canvas
```

Far-field Custom Layout is:

```text
frames
  -> B2CandidateProjectionProvider.project()
  -> B-2 per-camera warped_images / valid_masks
  -> post-warp camera_adjust
  -> B-2 weight-selection compositor
  -> optional centered output crop
  -> canvas
```

The custom path intentionally uses B-2 per-camera warped images, not the
flattened B-2 final canvas. Manual `x/y/scale` tuning still needs independent
camera layers.

The shared `camera_adjust` block supports `front_left`, `front`, and `front_right`:

```yaml
camera_adjust:
  front_left: {x_offset_px: 0, y_offset_px: 0, scale: 1.0}
  front: {x_offset_px: 0, y_offset_px: 0, scale: 1.0}
  front_right: {x_offset_px: 0, y_offset_px: 0, scale: 1.0}
```

This is a preview/runtime-only post-warp affine display adjustment. It is not camera extrinsics calibration and does not write `configs/calibration.yaml`.

Far-field Custom Layout does not use Near-field front-priority logic:

- no `side_visible_fraction`
- no `front_preserved_mask`
- no side suppression
- no front-first compositor

It preserves the B-2 candidate's far-field camera selection model after B-2
projection. It is intended for far-scene manual tuning, such as ceiling grid or
distant board alignment, when the B-2 candidate is close but needs small
post-warp offsets or scale changes.

## Near-field Projection Source

Near-field currently supports these projection sources:

- `Current Perspective`: default. This wraps `SurroundStitcher.warp_all()` through `CurrentPerspectiveProjectionProvider`.
- `Fisheye Rectilinear [Experimental]`: optional Near-field-only source. It uses OpenCV fisheye remap first, then reuses the current template perspective warp into the same canvas.

Far-field is not affected by this setting and continues to use the current formal runtime.

The Fisheye Rectilinear pipeline is:

```text
raw fisheye frame
  -> cv2.fisheye rectilinear remap
  -> rectified frame
  -> template perspective warp
  -> per-camera warped canvas
  -> Near-field front-priority compositor
```

This is not equirectangular projection and not a full physical panorama model. Because the template source points were tuned on the current perspective/template path, users should retune Layout Tuner parameters after switching projection source.

Fisheye Rectilinear requires a read-only intrinsics source with all three cameras and `model: opencv_fisheye`. It does not read from or write to `configs/calibration.yaml`.

## Research-only Projection Lines

`Equirectangular Candidate` remains research-only and is not selectable for runtime.

Future provider ideas such as `CurrentUndistortProjectionProvider` or equirectangular runtime should be added behind the same ProjectionProvider boundary, but they should not change Far-field behavior by default.

## Project Packages

Project package export/import is a distribution layer only. It copies configs
and candidate resources into a relative-path package so another machine can
load the same Far-field Default, Far-field Custom, Near-field, and projection
research inputs.

It does not change runtime algorithms:

- Far-field Default remains the original runtime path.
- Far-field Custom remains B-2 per-camera projection plus camera adjust plus
  B-2 weight selection.
- Near-field remains front-priority layout candidate based.
- Fisheye Rectilinear remains experimental and Near-field only.
- Equirectangular remains research-only.

## Candidate Versions

- Layout Candidate V2 has no projection block and is interpreted as `current_perspective`.
- Layout Candidate V3 may include a projection block.
- V3 `fisheye_rectilinear` candidates must include an intrinsics source path and are rejected at runtime if that source cannot be loaded.
- Far-field Layout Candidate V1 has `candidate_type: far_field_layout` and contains only `projection`, `camera_adjust`, `output`, and `far_field_blend`. New candidates should use `projection.source: b2_far_field_candidate` with a read-only B-2 `candidate_directory`, and `far_field_blend.mode: b2_weight_selection`. It must not contain Near-field pair suppression fields.

Candidate files remain preview/runtime inputs only. They do not modify the formal profile or `configs/calibration.yaml`.

## Safety Boundaries

- Do not write `configs/calibration.yaml` from runtime projection providers.
- Do not modify topology/profile/calibration schema for projection experiments.
- Do not route Far-field through Near-field providers.
- Do not use the Far-field final canvas as Near-field input.
- Do not enable equirectangular runtime until a separate implementation and validation pass is complete.
