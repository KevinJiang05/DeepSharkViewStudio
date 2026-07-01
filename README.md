# DeepShark View Studio

DeepShark View Studio is a standalone surround-view development tool extracted from the legacy `Code_End` prototype. It is intended to become the lab tool for camera preview, camera-source configuration, chessboard calibration, perspective tuning, and surround-view stitching before any QGC integration.

The legacy project remains untouched at:

```text
D:\Develop\image_mosaic\Code_End
```

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
    cameras.yaml
    network.yaml
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

Create or activate a Python environment, then install dependencies:

```powershell
cd D:\Develop\image_mosaic\DeepSharkViewStudio
python -m pip install -r requirements.txt
```

This workstation also has a prepared shared environment:

```text
D:\Develop\envs\deep-shark-view-studio
```

If a dependency is missing and installation is not available, check the shared environment directory first:

```text
D:\Develop\envs
```

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
cd D:\Develop\image_mosaic\DeepSharkViewStudio
python tools\run_image_demo.py
```

With the shared environment:

```powershell
D:\Develop\envs\deep-shark-view-studio\Scripts\python.exe tools\run_image_demo.py
```

Results are written to:

```text
samples/output/canvas.jpg
samples/output/*_warped.jpg
```

You can also point the demo at the legacy sample images:

```powershell
python tools\run_image_demo.py --input D:\Develop\image_mosaic\Code_End\3D_opengl_bowl_combined_mode2\Img
```

## GUI

Run:

```powershell
cd D:\Develop\image_mosaic\DeepSharkViewStudio
python app.py
```

With the shared environment:

```powershell
D:\Develop\envs\deep-shark-view-studio\Scripts\python.exe app.py
```

Or double-click:

```text
D:\Develop\image_mosaic\DeepSharkViewStudio\StartDeepSharkViewStudio.cmd
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
