from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from deep_shark_studio.config import (
    ConfigConflictError,
    load_yaml,
    save_yaml,
)


class AtomicConfigPersistenceTests(unittest.TestCase):
    def test_stale_writer_cannot_overwrite_newer_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.yaml"
            stale_revision = save_yaml(path, {"active_camera_count": 2})
            save_yaml(path, {"active_camera_count": 3})

            with self.assertRaises(ConfigConflictError):
                save_yaml(
                    path,
                    {"active_camera_count": 2},
                    expected_revision=stale_revision,
                )

            self.assertEqual(
                {"active_camera_count": 3},
                load_yaml(path),
            )
            self.assertEqual([], list(path.parent.glob("*.tmp")))

    def test_atomic_save_returns_revision_for_next_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.yaml"
            first_revision = save_yaml(path, {"enabled": False})
            second_revision = save_yaml(
                path,
                {"enabled": True},
                expected_revision=first_revision,
            )

            self.assertNotEqual(first_revision, second_revision)
            self.assertEqual({"enabled": True}, load_yaml(path))


if __name__ == "__main__":
    unittest.main()
