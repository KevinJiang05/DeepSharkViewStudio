"""Reserved launcher for live video stitching.

This file intentionally stays small in the first studio version. The next step is
to connect the VideoSource abstraction to the SurroundStitcher and GUI runtime.
"""

from __future__ import annotations


def main() -> int:
    print("Live video demo is not implemented yet.")
    print("Use tools\\run_image_demo.py or app.py for the current still-image workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
