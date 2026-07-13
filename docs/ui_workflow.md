# UI Workflow

DeepSharkViewStudio exposes five task-oriented workspaces. Runtime operation,
candidate authoring, calibration, project transfer, and diagnostics have one
primary home each. Advanced source controls are contextual and collapsed until
the selected task requires them.

## 1. Realtime Monitor

Use this workspace to load still images, start or stop camera preview, switch
between multi-camera and stitched display, and select one concrete runtime
view. Display layout does not change the processing algorithm.

The **Runtime View** selector is the only runtime-mode entry:

| View | Processing path | Required source |
| --- | --- | --- |
| **Far Default** | Current formal profile and perspective path | None |
| **B-2 View** | Advanced diagnostic view of the B-2 per-camera candidate | B-2 candidate |
| **Far Custom** | B-2 per-camera projection, manual camera adjustment, and `b2_weight_selection` | B-2 candidate plus Far Custom layout |
| **Near Current** | Current Perspective projection plus the front-priority Near layout | Near layout candidate |
| **Near Fisheye** | Fisheye Rectilinear projection plus the front-priority Near layout | Near layout candidate plus fisheye intrinsics |

The production application starts live preview automatically with the current
Runtime View. Changing the selector applies a valid choice immediately. If
preview is already live, only the stitch worker is reconfigured; the three
RTSP capture workers are not restarted. If preview is stopped, the valid
selection starts it. A selection that lacks a required candidate or intrinsics
source remains **Selected, not applied**, while the previous **Effective** view
keeps running. Candidate and projection sources expand only for the selected
view and never write `configs/calibration.yaml`.

## 2. Layout & Projection Lab

Use this workspace to author read-only layout candidates from captured frames.
Projection research lives here because it feeds candidate comparison rather
than acting as a separate runtime workspace.

- **Near Layout** tunes front-priority pair parameters, camera adjustment,
  vertical safety, and either Current Perspective or experimental Fisheye
  Rectilinear projection.
- **Far Layout** starts from the B-2 per-camera projection basis, then tunes
  camera adjustment and B-2 weight selection. It does not use Near side
  suppression.
- Saving creates a candidate file. It does not modify the formal profile or
  `calibration.yaml`.
- Equirectangular and other future providers remain research-only until they
  have an explicit runtime contract and tests.

## 3. Calibration & Candidates

The default **Calibration Wizard** guides device preparation, board definition,
intrinsic sampling, pair sampling, readiness review, and B-2 candidate
generation. Perspective, seam, image-set, geometry, pairwise, and session-detail
tools remain in tabs explicitly labelled **Advanced** or **Experimental**.

Canonical candidate names are:

- B-2 Calibration Candidate
- Far Custom Layout Candidate
- Near Layout Candidate
- Fisheye Intrinsics Candidate

## 4. Project Management

The first subtab is **Project & Portability**.

- **Portable Project Package (recommended)** includes configs and active
  Far/Near/B-2/fisheye candidates using package-relative paths.
- **Validate Package (Read-only)** does not modify the package or active config.
- **Validate & Activate Package** validates, backs up active configs, activates
  the package, and restores runtime defaults on restart.
- The current active package and the most recent project action are shown as
  separate states.

**Legacy Project File (`.dsvs.yaml`)** is hidden under **Legacy & Advanced
Tools**. It contains only the three config YAML documents, carries no candidate
assets, and is retained for compatibility rather than team transfer. Config
backup and experimental Runtime Snapshot export are maintenance tools, not
portable-project formats.

Camera sources remain under **Camera Setup**. QGC video output remains under an
explicitly advanced subtab and is outside the normal project-transfer flow.

## 5. Diagnostics & Logs

This read-only workspace holds application log evidence, copy/clear-view tools,
and the log path. The Realtime Monitor keeps the effective view, projection,
camera health, warning count, and candidate state next to the live images where
they are actionable; internal enum names remain in detailed logs only.

## Window and test behavior

- Root tabs use scroll buttons when the window is narrow.
- Layout controls have their own vertical scroll area.
- Runtime sources are collapsed and contextual.
- Calibration uses a scroll area for advanced pages.
- GUI tests use `QT_QPA_PLATFORM=offscreen`.
- Screenshot QA uses the Windows Qt platform with
  `WA_DontShowOnScreen`; the Qt offscreen plugin exposes no Windows font
  database and otherwise renders Chinese as missing-glyph boxes.
- Real-window smoke screenshots use Win32 `PrintWindow`, so another foreground
  window cannot cover the captured Studio content and Chinese uses the normal
  Windows font stack.
