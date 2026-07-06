# QGC Video Payload Service

DeepSharkViewStudio can run as a headless stitched-video payload service. The
first integration stage keeps QGroundControl simple: QGC consumes a standard
video stream, while DeepShark owns camera capture, stitching, and candidate
selection.

## Architecture

```text
front_left / front / front_right cameras
  -> DeepShark GUI Realtime Monitor or headless runtime service
  -> stitched BGR canvas
  -> FFmpeg H.264 stream
  -> UDP MPEG-TS or RTSP
  -> QGC standard video source
```

This does not embed Python/PySide into QGC and does not make QGC perform
stitching.

DeepShark supports two QGC output modes:

```text
UDP Local Low Latency:
  udp://127.0.0.1:5600?pkt_size=1316

RTSP Standard Service:
  rtsp://127.0.0.1:8554/deepshark
```

Use UDP for same-machine low-latency bridge testing. Use RTSP when a local or
onboard RTSP server, for example MediaMTX, should accept the published stream
and QGC should open a standard RTSP URL.

FFmpeg uses wall-clock timestamps for raw stdin frames and disables MPEG-TS
mux buffering where possible. This keeps QGC latency reporting closer to the
actual DeepShark frame production cadence when the stitcher cannot sustain the
configured nominal output FPS.

For B-2 Candidate View, DeepShark can optionally use an experimental OpenCL
processor. It keeps B-2 remap and weight-selection composition on the OpenCL
device and downloads only the final stitched canvas for FFmpeg. This is enabled
from the UI performance settings or by setting
`configs/cameras.yaml -> performance.b2_candidate_opencl: true`; it remains
off by default and does not modify `calibration.yaml`.

When the command line does not provide `--output-kind` / `--output-url`, the
headless service reads the saved UI settings from
`configs/cameras.yaml -> qgc_video_output`. If that block is missing, it falls
back to the UDP local bridge default above.

## Runtime Modes

Far-field Default:

```text
frames -> SurroundStitcher.process() -> current formal runtime canvas
```

Far-field Custom:

```text
frames
  -> B-2 candidate per-camera projection
  -> manual x/y/scale camera layout adjustment
  -> B-2 weight selection
  -> canvas
```

Near-field:

```text
frames
  -> ProjectionProvider
  -> Layout Candidate V2/V3
  -> front-priority compositor
  -> canvas
```

## Example Commands

Far-field Default to local QGC:

```powershell
D:\Develop\envs\deep-shark-view-studio\Scripts\python.exe -m deep_shark_studio.qgc.runtime_service `
  --mode far_field `
  --output-kind udp_mpegts `
  --output-url "udp://127.0.0.1:5600?pkt_size=1316" `
  --process-fps 15 `
  --output-fps 15
```

Far-field Custom with a saved Far-field Layout Candidate:

```powershell
D:\Develop\envs\deep-shark-view-studio\Scripts\python.exe -m deep_shark_studio.qgc.runtime_service `
  --mode far_field `
  --far-field-layout-candidate "D:\path\to\far_field_layout\candidate.yaml" `
  --output-kind rtsp `
  --output-url "rtsp://<rtsp-server-ip>:8554/deepshark"
```

Near-field with a saved Layout Candidate:

```powershell
D:\Develop\envs\deep-shark-view-studio\Scripts\python.exe -m deep_shark_studio.qgc.runtime_service `
  --mode near_field `
  --near-field-layout-candidate "D:\path\to\near_field_layout\candidate.yaml" `
  --output-kind rtsp `
  --output-url "rtsp://<rtsp-server-ip>:8554/deepshark"
```

RTSP service output:

```powershell
D:\Develop\envs\deep-shark-view-studio\Scripts\python.exe -m deep_shark_studio.qgc.runtime_service `
  --mode far_field `
  --output-kind rtsp `
  --output-url "rtsp://<rtsp-server-ip>:8554/deepshark"
```

## QGC Side

For UDP local bridge mode, configure QGC to receive UDP/H.264 or MPEG-TS video
on port 5600.

For RTSP service mode, configure QGC to open the RTSP stream URL, for example:

```text
rtsp://<rtsp-server-ip>:8554/deepshark
```

The exact QGC labels depend on the QGC build, but keep the DeepShark output mode
and QGC video source type matched: UDP with UDP, RTSP with RTSP.

## UI Settings

The desktop UI stores QGC video output settings under:

```text
Project Management -> QGC Video Output
```

These settings are saved in `configs/cameras.yaml` under `qgc_video_output`.
They do not modify `configs/calibration.yaml`.

The same UI page also provides:

```text
Start QGC Output Service
Stop QGC Output Service
```

These buttons open or close an FFmpeg video sink inside the GUI process. Each
new stitched canvas produced by Realtime Monitor is written to the QGC output
stream, so the QGC stream follows the same Far-field Default, Far-field Custom,
Near-field, or B-2 Candidate View canvas that the GUI is already producing. The
buttons do not start a second camera capture or stitching pipeline.

The command-line headless service remains available for deployment scenarios
where DeepShark runs without the PySide UI.

For headless B-2 candidate output, pass `--candidate-opencl` to request the
experimental OpenCL processor. If OpenCL is not available in the current OpenCV
runtime, DeepShark falls back to the CPU candidate processor.

## Safety Boundaries

- The service reads `configs/cameras.yaml` and `configs/calibration.yaml`.
- The service does not write `configs/calibration.yaml`.
- The service does not modify topology/profile/calibration schema.
- QGC receives video only in V1.
- MAVLink camera/status control is a later integration stage.
- Auto switching, equirectangular runtime, and new stitching algorithms remain
  out of scope.
