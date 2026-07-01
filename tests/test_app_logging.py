from __future__ import annotations

import unittest

from deep_shark_studio.app_logging import redact_log_message


class ApplicationLoggingTests(unittest.TestCase):
    def test_rtsp_credentials_are_redacted(self) -> None:
        message = (
            "Open failed: "
            "rtsp://192.168.1.12:554/user=admin&password=secret&channel=1 "
            "password=second-secret"
        )

        redacted = redact_log_message(message)

        self.assertNotIn("192.168.1.12", redacted)
        self.assertNotIn("secret", redacted)
        self.assertIn("rtsp://<redacted>", redacted)
        self.assertIn("password=<redacted>", redacted)


if __name__ == "__main__":
    unittest.main()
