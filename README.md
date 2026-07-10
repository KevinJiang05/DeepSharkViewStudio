# DeepShark View Studio

DeepShark View Studio is a standalone surround-view development tool extracted from the legacy `Code_End` prototype. It is intended to become the lab tool for camera preview, camera-source configuration, chessboard calibration, perspective tuning, and surround-view stitching before any QGC integration. The legacy source tree is not required to install or run this repository.

## Current Features

- Loads camera calibration from `configs/calibration.yaml`
- Supports 2, 3, 4, or 5 active camera channels
- Supports English and Chinese UI language switching
- Warps five camera views into a shared bird's-eye canvas
- Applies seam masks based on the legacy AVM stitching rules
- Produces a stitched surround-view canvas
- Provides a PySide6 GUI with three core workspaces
- Detects the lab chessboard calibration board
- Keeps placeholders for future video input and streaming output

## Workspaces

### Realtime Preview

- Load a still-image directory for quick stitching checks
- Start live preview from configured USB, RTSP, or video-file sources
- Display raw camera frames, warped bird's-eye frames, and final stitched canvas
- Show per-camera enabled state, source, status, and frame counts

### Camera Config

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

### Project

- Save all editable configs into one `.dsvs.yaml` project file
- Open a `.dsvs.yaml` project file and restore configs
- Create timestamped backups before risky calibration or seam edits
- Export a compact runtime config for a future service/QGC bridge

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

The launcher first uses `.venv\Scripts\python.exe`, then falls back to `py -3` and `python`. It contains no machine-specific drive or environment path.

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
- Realtime Preview workspace
- Camera Config workspace
- Calibration workspace
- Project workspace
- Image-directory and live-source preview paths
- Chessboard detection and intrinsics export
- Calibration quality reports and undistortion preview
- Project save/open/backup/export actions
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
