# QGC Video Payload Service

DeepSharkViewStudio can run as a headless stitched-video payload service. The
first integration stage keeps QGroundControl simple: QGC consumes a standard
video stream, while DeepShark owns camera capture, stitching, and candidate
selection.

## Architecture

```text
front_left / front / front_right cameras
  -> DeepShark headless runtime service
  -> RuntimeStitchController
  -> stitched BGR canvas
  -> FFmpeg H.264 RTSP publish stream
  -> RTSP server
  -> QGC standard RTSP video source
```

This does not embed Python/PySide into QGC and does not make QGC perform
stitching.

The default output is an RTSP publish URL:

```text
rtsp://127.0.0.1:8554/deepshark
```

DeepShark publishes to that URL. A local or onboard RTSP server, for example
MediaMTX, must accept the published stream. QGC should then open the same RTSP
stream URL as a standard video source. UDP MPEG-TS remains available as a
compatibility option for QGC builds or test networks that prefer `udp://` input.

When the command line does not provide `--output-kind` / `--output-url`, the
headless service reads the saved UI settings from
`configs/cameras.yaml -> qgc_video_output`. If that block is missing, it falls
back to the RTSP default above.

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
  --output-kind rtsp `
  --output-url "rtsp://127.0.0.1:8554/deepshark" `
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

UDP MPEG-TS compatibility output:

```powershell
D:\Develop\envs\deep-shark-view-studio\Scripts\python.exe -m deep_shark_studio.qgc.runtime_service `
  --mode far_field `
  --output-kind udp_mpegts `
  --output-url "udp://<qgc-ip>:5600?pkt_size=1316"
```

## QGC Side

For the default path, configure QGC to open the RTSP stream URL, for example:

```text
rtsp://<rtsp-server-ip>:8554/deepshark
```

If using the compatibility UDP MPEG-TS output, configure QGC to receive the UDP
video stream on the same port, commonly 5600. The exact QGC label depends on the
QGC build, but use the standard UDP/H.264 or MPEG-TS video source option and
point it at the selected port.

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

These buttons launch or stop the headless DeepShark payload service as a
separate process using the current Far-field / Near-field runtime selection and
the configured QGC output URL. The GUI remains responsive while the service is
running.

## Safety Boundaries

- The service reads `configs/cameras.yaml` and `configs/calibration.yaml`.
- The service does not write `configs/calibration.yaml`.
- The service does not modify topology/profile/calibration schema.
- QGC receives video only in V1.
- MAVLink camera/status control is a later integration stage.
- Auto switching, equirectangular runtime, and new stitching algorithms remain
  out of scope.
