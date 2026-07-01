"""Run a still-image surround-view stitch demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deep_shark_studio.config import load_config
from deep_shark_studio.stitcher import SurroundStitcher, load_images_from_directory, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepShark View Studio image stitching demo.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "samples" / "input"),
        help="Directory containing camera images named with front_left/front_right/behind/left/right.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "samples" / "output" / "canvas.jpg"),
        help="Output stitched canvas path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = load_images_from_directory(args.input)
    if not frames:
        print(f"No input images found in {args.input}")
        return 2

    stitcher = SurroundStitcher(load_config("calibration.yaml"))
    warped, canvas = stitcher.process(frames)
    save_image(args.output, canvas)

    output_dir = Path(args.output).parent
    for camera_name, image in warped.items():
        save_image(output_dir / f"{camera_name}_warped.jpg", image)

    print(f"Loaded cameras: {', '.join(sorted(frames))}")
    print(f"Saved stitched canvas: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
