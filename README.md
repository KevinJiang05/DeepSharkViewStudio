# DeepShark View Studio

DeepShark View Studio is a standalone surround-view development tool extracted from the legacy `Code_End` prototype. It is intended to become the lab tool for camera preview, camera-source configuration, chessboard calibration, perspective tuning, and surround-view stitching before any QGC integration. The legacy source tree is not required to install or run this repository.

## Current Features

- Loads camera calibration from `configs/calibration.yaml`
- Supports 2, 3, 4, or 5 active camera channels
- Supports English and Chinese UI language switching
- Warps five camera views into a shared bird's-eye canvas
- Applies seam masks based on the legacy AVM stitching rules
- Produces a stitched surround-view canvas
- Provides a PySide6 GUI with five task-oriented workspaces
- Detects the lab chessboard calibration board
- Keeps advanced QGC/video output separate from the normal preview workflow

## Workspaces

### Realtime Monitor

- Load a still-image directory for quick stitching checks
- Start live preview from configured USB, RTSP, or video-file sources
- Display raw camera frames, warped bird's-eye frames, and final stitched canvas
- Show per-camera enabled state, source, status, and frame counts
- Select one explicit runtime view: Far Default, B-2 View, Far Custom, Near
  Current, or Near Fisheye
- Distinguish a staged selection from the effective worker state

### Layout & Projection Lab

- Author Near front-priority and Far B-2-backed layout candidates
- Compare Current Perspective and experimental Near Fisheye projection
- Keep projection research and candidate authoring out of runtime operation
- Save candidate files without changing `configs/calibration.yaml`

### Camera Setup

- Set active camera count from 2 to 5
- Configure each channel name, source type, and source value
- Supported source types: `image_dir`, `usb`, `video_file`, `rtsp`
- Tune realtime performance:
  - `Stitch FPS`: how often the surround canvas is recomputed
  - `Preview FPS`: how often raw preview thumbnails are refreshed
  - `Max input width`: downscale high-resolution RTSP frames before stitching
  - `Refresh warped previews`: disable during live testing for smoother UI
  - `Use undistort live`: disable during link testing, enable later for final quality
- Save camera settings to `configs/cameras.yaml`
- Save canvas size to `configs/calibration.yaml`

### Calibration

- Lab board defaults:
  - Total squares: `12 x 9`
  - Square size: `25 mm`
  - OpenCV inner-corner pattern: `11 x 8`
- Load a calibration image and run chessboard detection
- Calibrate camera intrinsics from a folder of chessboard images
- Save camera matrix, distortion coefficients, RMS, detection summary, and reprojection errors
- Preview undistortion using saved camera intrinsics
- Export a Markdown calibration quality report
- Drag the four source perspective points directly on the calibration image
- Edit source and target perspective points in the synchronized table
- Use `Preview Warp` to inspect the current perspective transform before saving
- Drag seam endpoints directly on the stitched surround canvas
- Capture calibration frames from the currently selected preview camera
- Manage a calibration image folder and scan chessboard detection status
- Save calibration data to `configs/calibration.yaml`

### Project Management

- Export, read-only validate, and activate a Portable Project Package containing
  configs plus active candidates
- Use legacy `.dsvs.yaml` config-only files only from the collapsed compatibility
  section
- Create timestamped backups before risky calibration or seam edits
- Export an explicitly experimental Runtime Snapshot for a future service bridge

### Diagnostics & Logs

- Inspect and copy the application log without changing runtime state
- Keep effective view, projection, health, and warnings beside the realtime view

## Project Layout

```text
DeepSharkViewStudio/
  app.py
  StartDeepSharkViewStudio.cmd
  requirements.txt
  configs/
    calibration.yaml
    cameras.example.yaml
    network.example.yaml
    cameras.yaml          # local, initialized explicitly and ignored by Git
    network.yaml          # local, initialized explicitly and ignored by Git
  deep_shark_studio/
    config.py
    camera.py
    masks.py
    stitcher.py
    streaming.py
    gui/
      main_window.py
      runtime_workflow.py
      panels/
  tools/
    run_image_demo.py
  samples/
    input/
    output/
```

## Install

From a fresh checkout, create a repository-local environment and install the tracked dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The launcher first honors an optional `DEEP_SHARK_PYTHON` interpreter path, then tries `.venv\Scripts\python.exe`, a shared workspace environment at `..\..\envs\deep-shark-view-studio`, `py -3`, and `python`. Every candidate must provide the full runtime dependency set. It contains no machine-specific drive path, and a failed no-argument launch keeps the error window open.

## Local Configuration And First Start

`configs/calibration.yaml` is tracked because it contains device-specific calibration data. Local camera and network settings are deliberately not tracked. Initialize only the missing local files from the safe examples:

```powershell
.\.venv\Scripts\python.exe app.py --initialize-configs --preflight-only
```

Initialization is explicit and transactional: it creates missing `cameras.yaml` and `network.yaml`, never overwrites an existing file, and never generates or replaces `calibration.yaml`. Review camera sources before starting live preview. Re-run the read-only preflight at any time with:

```powershell
.\.venv\Scripts\python.exe app.py --preflight-only
```

Both commands can also be passed through `StartDeepSharkViewStudio.cmd`. A normal launch remains read-only and reports a clear error when required local configs are absent or invalid.

## Image Demo

Put images into `samples/input`. Filenames should contain these camera keys:

```text
front_left
front_right
behind
left
right
```

Then run:

```powershell
python tools\run_image_demo.py
```

Results are written to:

```text
samples/output/canvas.jpg
samples/output/*_warped.jpg
```

You can also point the demo at another image directory:

```powershell
python tools\run_image_demo.py --input <path-to-images>
```

## GUI

Run:

```powershell
StartDeepSharkViewStudio.cmd
```

Or, from an activated environment:

```powershell
python app.py
```

The GUI currently supports:

- English/Chinese language switching from the top language selector
- Realtime Monitor workspace with one five-view runtime selector
- Layout & Projection Lab workspace
- Calibration & Candidates workspace with wizard-first navigation
- Project Management workspace with portable-first semantics
- Diagnostics & Logs workspace
- Image-directory and live-source preview paths
- Chessboard detection and intrinsics export
- Calibration quality reports and undistortion preview
- Portable package export/validate/activate plus collapsed legacy tools
- Perspective point editing
- Stitched result saving

## Development Roadmap

1. Add robust live camera sources for USB, RTSP, and video files.
2. Add draggable source and seam points directly on images.
3. Add frame-rate, latency, camera health, and reconnect status.
4. Add video recording and snapshot export.
5. Add RTSP/GStreamer output for QGC consumption.
6. Consider native C++/Qt integration after the standalone tool is stable.

## Notes

The default calibration values are taken from the more complete five-camera legacy prototype in `Code_End\Part12`. They are device-specific and should be recalibrated for a different camera rig or mounting geometry.
