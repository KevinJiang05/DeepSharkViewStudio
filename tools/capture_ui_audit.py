"""Capture deterministic, non-visible screenshots for UI workflow review.

Qt's ``offscreen`` Windows plugin does not expose the system font database in
our test environment, so CJK text is rendered as missing-glyph boxes.  The
Windows plugin does expose the normal font fallback stack.  Combining it with
``WA_DontShowOnScreen`` keeps the audit non-visible while preserving real text
rendering.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = os.environ.get(
    "DEEP_SHARK_CAPTURE_PLATFORM",
    "windows",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from deep_shark_studio.gui.main_window import MainWindow


OUTPUT_DIR = PROJECT_ROOT / "reports" / "ui_audit"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="capture")
    parser.add_argument("--width", type=int, default=1366)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--runtime-view",
        choices=(
            "far_default",
            "b2_view",
            "far_custom",
            "near_current",
            "near_fisheye",
        ),
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show one real Windows window while cycling through screenshots.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = QApplication.instance() or QApplication([])
    with patch(
        "deep_shark_studio.gui.main_window.create_application_logger",
        return_value=MagicMock(),
    ):
        window = MainWindow()
        if not args.visible:
            window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.resize(args.width, args.height)
        window.show()
        app.processEvents()
        if args.runtime_view:
            index = window.runtime_view_combo.findData(args.runtime_view)
            window.runtime_view_combo.setCurrentIndex(index)
            app.processEvents()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for index in range(window.root_tabs.count()):
            window.root_tabs.setCurrentIndex(index)
            app.processEvents()
            title = window.root_tabs.tabText(index)
            slug = "".join(
                character if character.isalnum() else "_"
                for character in title
            )
            path = OUTPUT_DIR / f"{args.prefix}_{index + 1}_{slug}.png"
            if not window.grab().save(str(path)):
                raise RuntimeError(f"Failed to save {path}")
            print(path)

        window.close()
        app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
