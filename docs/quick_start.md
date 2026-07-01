# DeepShark View Studio Quick Start

This page is for project members who only need to run the three-camera
preview, collect calibration samples, and understand common warnings.

## 1. Start Three-Camera Live Preview

1. Open **Camera Config**.
2. Confirm the active topology is the three-camera profile used by the project.
3. Enable `front_left`, `front`, and `front_right`.
4. Set each source to the correct RTSP input, then save the camera config.
5. Return to **Realtime Preview** and click **Start Live**.

The preview status summary shows the current topology, active camera count,
Live / Connecting / Failed / Stopped counts, layout mode, content mode, and
the current calibration status.

## 2. Switch Views

- **Multi-camera View** shows the raw camera tiles.
- Double-click a raw tile to enter single-camera Focus view.
- **Back to Grid** returns from Focus to the multi-camera grid.
- **Stitched View** shows the stitched canvas.

If the current stitching mode is experimental or rotation-only, the stitched
view favors distant content. Nearby people, boxes, or equipment can still
show double images because the three camera optical centers are not at the
same point.

## 3. Collect Calibration Samples

Open **Calibration and Tuning -> Calibration Wizard**.

Follow the five steps in order:

1. Prepare devices: keep the three cameras fixed at 1920 x 1080.
2. Confirm the board: for the lab chessboard, use 11 x 8 inner corners and
   25 mm square size, then explicitly confirm the parameters.
3. Collect each single camera: `front_left`, then `front`, then
   `front_right`. Move the board through center, edges, corners, different
   distances, and different tilts.
4. Collect adjacent pairs: first `front_left <-> front`, then
   `front <-> front_right`. The same physical board must be visible in both
   cameras of the selected pair.
5. Review readiness before entering B-2.

The wizard never captures continuously. Every saved sample is triggered by
one user click.

## 4. When Can B-2 Run?

The formal readiness gate requires:

- `front_left`, `front`, and `front_right`: at least 20 accepted samples each.
- `front_left <-> front`: at least 15 accepted pair samples.
- `front <-> front_right`: at least 15 accepted pair samples.
- Confirmed board definition.
- Session resolution fixed at 1920 x 1080.

If counts are sufficient but coverage is weak, the wizard may allow
**experimental candidate solving (report only)**. Experimental candidates can
produce reports and previews, but cannot be applied to the formal
`calibration.yaml`.

## 5. Why Nearby Objects Can Still Double

The current three-camera rig is not a single optical center. Adjacent camera
optical centers are separated by roughly 22 cm and the side cameras look
nearly sideways relative to the front camera. A rotation-only panorama can
align distant content, but nearby objects have physical parallax and may not
line up across the seam.

For close-range inspection, use Grid or Focus. Treat the experimental stitched
view as a diagnostic panorama, not as a final seamless result.

## 6. What To Do When Something Fails

- **Camera Connecting**: wait briefly; if it does not change, check RTSP URL,
  network, power, and credentials.
- **Camera Failed**: fix the camera source first. Other live cameras can keep
  running.
- **Board not detected**: make the full board visible, reduce blur and glare,
  and verify the board type and corner count.
- **Board area too small**: move the board closer.
- **Blurred sample**: stop the board for about one second before capturing.
- **Duplicate pose**: move to another edge, corner, distance, or tilt.
- **Time delta risk**: keep the same physical board still before capturing
  the pair.
- **B-2 candidate cannot be applied**: collect stronger pair coverage and
  regenerate the candidate. The formal configuration is unchanged.
