# Contributing

Thanks for helping improve DeepShark View Studio. Small, focused changes with reproducible validation are preferred.

## Development setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the test suite before submitting a change:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q
```

## Change boundaries

- Keep camera, projection, composition, and output responsibilities in their existing modules when possible.
- Treat experimental calibration, projection, layout, and seam candidates as staged artifacts. Do not silently promote them into `configs/calibration.yaml`.
- Do not commit local camera or network configuration, captures, logs, generated projects, or UI artifacts.
- Preserve explicit selected, applied, and effective runtime states in GUI changes.
- Include a focused regression test for bug fixes and contract changes.

## Issues and pull requests

When reporting a problem, include the runtime view, source type, input resolution, relevant non-sensitive logs, and a minimal reproduction. Remove RTSP credentials, private addresses, local paths, faces, and workplace imagery before attaching artifacts.

Pull requests should explain the user-visible behavior, the validation performed, and any remaining hardware-specific limitation.
