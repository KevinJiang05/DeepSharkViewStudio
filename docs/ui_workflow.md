# UI Workflow

DeepSharkViewStudio uses workflow-oriented workspaces. The goal is to keep
runtime operation, layout tuning, calibration, projection research, project
management, and diagnostics separate without hiding expert controls.

## Realtime Monitor

Runtime workspace for connecting cameras, viewing raw streams, viewing stitched
output, and selecting the current stitching mode.

- Far-field Default keeps the formal runtime path.
- Far-field Custom uses B-2 per-camera projection plus manual camera layout
  adjustment and B-2 weight selection.
- Near-field uses Layout Candidate V2/V3 with the front-priority compositor.
- Apply/use actions are preview/runtime-only and do not write
  `configs/calibration.yaml`.

## Layout Tuning

Candidate authoring workspace.

- Far-field Layout tunes three post-warp camera layers on top of the B-2
  per-camera projection basis. It does not use Near-field side suppression.
- Near-field Layout tunes front-priority pair parameters, camera adjustment,
  projection source, and vertical safety settings.
- Saving a candidate writes a separate candidate file. It does not modify the
  formal profile or calibration file.

## Calibration & Candidates

Calibration workspace for chessboard/session sampling, B-2 calibration
candidates, reports, seam/profile diagnostics, and pairwise diagnostics.

Use precise names in UI copy:

- B-2 Calibration Candidate
- B-2 Candidate View
- Far-field Layout Candidate
- Near-field Layout Candidate
- Projection Candidate

## Projection Research

Research workspace for projection experiments and A/B comparison notes.

- Near-field Fisheye Rectilinear is experimental and only affects Near-field
  preview/runtime when explicitly selected.
- Equirectangular remains research-only.
- Projection research is not Far-field Default.
- Far-field Custom currently uses the B-2 candidate projection basis.

## Project Management

Configuration and distribution workspace.

- Camera and runtime configuration remains available here.
- Project files bundle configuration YAMLs.
- Project packages export configs and currently loaded candidates into a
  directory with package-relative paths for team handoff.
- Import validates first. Activation backs up current configs before replacing
  them with package configs.
- Runtime export remains an experimental/config distribution path and does not
  deploy or apply a runtime automatically.

## Diagnostics & Logs

Troubleshooting workspace for logs and status evidence. Live camera status,
frame age, FPS, stitch timing, warnings, and candidate/provider metadata are
still surfaced near the runtime views where they are most actionable.
